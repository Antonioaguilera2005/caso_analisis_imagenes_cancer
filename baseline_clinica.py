"""Referencia de regresión logística por paciente: edad y menopausia, cinco folds, sin test."""
import argparse
from pathlib import Path
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from clinica import ajustar_clinica, transformar_clinica
from analizar_resultados import metricas, guardar_json, umbrales, paneles


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--datos", type=Path, default=Path("breastdcedl"))
    p.add_argument("--salida", type=Path, required=True)
    a = p.parse_args()
    # Filtrar train antes de consultar variables o resultados.
    pacientes = pd.read_csv(a.datos / "metadata/patients.csv")
    pacientes = pacientes[pacientes.split == "train"].copy()
    muestras = pd.read_csv(a.datos / "metadata/samples.csv")
    folds = muestras[muestras.split == "train"][["patient_id", "fold"]].drop_duplicates()
    if folds.patient_id.duplicated().any():
        raise ValueError("Paciente en varios folds")
    pacientes = pacientes.merge(folds, left_on="pid", right_on="patient_id", validate="one_to_one")
    if len(pacientes) != len(folds) or pacientes.pid.duplicated().any():
        raise ValueError("Cobertura de pacientes incompleta")
    a.salida.mkdir(parents=True, exist_ok=False)
    faltantes = pacientes.groupby("dataset").agg(pacientes=("pid", "size"), edad_ausente=("age", lambda s: s.isna().sum()), menopausia_ausente=("menopause", lambda s: s.isna().sum()))
    faltantes.to_csv(a.salida / "ausentes_por_cohorte.csv")
    resultados, tablas, resumen = [], {}, {}
    for nombre, ponderacion in (("normal", None), ("ponderada", "balanced")):
        partes = []
        for fold in range(5):
            tr, va = pacientes[pacientes.fold != fold], pacientes[pacientes.fold == fold]
            estado = ajustar_clinica(tr)
            modelo = LogisticRegression(C=1.0, class_weight=ponderacion, max_iter=1000, random_state=42)
            modelo.fit(transformar_clinica(tr, estado), tr.pCR)
            prob = modelo.predict_proba(transformar_clinica(va, estado))[:, 1]
            t = va[["patient_id", "pCR", "fold", "dataset"]].copy()
            t["prob"] = prob
            partes.append(t)
            m = metricas(va.pCR, prob, 0.5)
            resultados.append(dict(perdida=nombre, fold=fold, **m, average_precision=float(average_precision_score(va.pCR, prob))))
            guardar_json(a.salida / f"preprocesado_fold{fold}_{nombre}.json", estado)
            guardar_json(a.salida / f"modelo_fold{fold}_{nombre}.json", dict(coef=modelo.coef_.tolist(), intercept=modelo.intercept_.tolist(), C=1.0, class_weight=ponderacion))
        tabla = pd.concat(partes, ignore_index=True)
        if tabla.patient_id.duplicated().any() or len(tabla) != len(pacientes):
            raise ValueError("OOF incompleta")
        tabla.to_csv(a.salida / f"predicciones_validacion_{nombre}.csv", index=False)
        tablas[nombre] = tabla
        exploratorio = umbrales(tabla, nombre, a.salida)
        resumen[nombre] = dict(referencia=metricas(tabla.pCR, tabla.prob, 0.5), umbral_exploratorio=exploratorio)
    plano = pd.DataFrame([{k: v for k, v in m.items() if not isinstance(v, (dict, list))} for m in resultados])
    plano.to_csv(a.salida / "metricas_por_fold.csv", index=False)
    plano.groupby("perdida")[["auc", "average_precision", "f1_macro"]].agg(["mean", "std"]).to_csv(a.salida / "resumen.csv")
    guardar_json(a.salida / "metricas.json", resultados)
    paneles(tablas, a.salida, resumen)
    print(plano.groupby("perdida")[["auc", "average_precision", "f1_macro"]].agg(["mean", "std"]).to_string())


if __name__ == "__main__":
    main()
