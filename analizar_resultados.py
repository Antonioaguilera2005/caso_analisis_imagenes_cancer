"""Genera PNG y métricas de validación interna; nunca carga ni evalúa test.

Ejemplo: python analizar_resultados.py --runs runs/fold0_semilla42 --salida runs/analisis_fold0
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (average_precision_score, brier_score_loss, confusion_matrix,
                             precision_recall_curve, precision_recall_fscore_support,
                             roc_auc_score, roc_curve)

sys.path.insert(0, str(Path(__file__).resolve().parent / "breastdcedl"))
import utils_caso as uc

COLORES = {"normal": "#2474b5", "ponderada": "#dd6c2f"}


def leer_json(ruta):
    return json.loads(ruta.read_text(encoding="utf-8-sig"))


def guardar_json(ruta, datos):
    ruta.write_text(json.dumps(datos, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def metricas(y, prob, umbral):
    pred = np.asarray(prob) >= umbral
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    precision, recall, f1, soporte = precision_recall_fscore_support(y, pred, labels=[0, 1], zero_division=0)
    return dict(umbral=float(umbral), pacientes=len(y), auc=float(roc_auc_score(y, prob)),
                accuracy=float((tn + tp) / len(y)), sensibilidad=float(recall[1]),
                especificidad=float(recall[0]), balanced_accuracy=float(np.mean(recall)),
                precision_pcr=float(precision[1]) if tp + fp else None,
                f1_no_pcr=float(f1[0]), f1_pcr=float(f1[1]), f1_macro=float(np.mean(f1)),
                brier=float(brier_score_loss(y, prob)),
                confusion=[[int(tn), int(fp)], [int(fn), int(tp)]],
                por_clase={str(c): dict(precision=float(precision[c]) if np.sum(pred == c) else None,
                                       recall=float(recall[c]), f1=float(f1[c]), soporte=int(soporte[c])) for c in (0, 1)})


def intervalo_auc(y, prob, repeticiones, semilla=42):
    """Bootstrap estratificado por paciente; incertidumbre condicional al modelo elegido."""
    y, prob = np.asarray(y), np.asarray(prob)
    rng = np.random.default_rng(semilla)
    negativos, positivos = np.flatnonzero(y == 0), np.flatnonzero(y == 1)
    valores = []
    for _ in range(repeticiones):
        indices = np.r_[rng.choice(negativos, len(negativos), replace=True),
                        rng.choice(positivos, len(positivos), replace=True)]
        valores.append(roc_auc_score(y[indices], prob[indices]))
    return [float(v) for v in np.quantile(valores, [0.025, 0.975])]


def cargar_ejecucion(directorio, samples, pacientes):
    config = leer_json(directorio / "config.json")
    if config.get("prueba") or config.get("uso") != "validacion_interna":
        raise ValueError(f"No es un experimento completo de validación: {directorio}")
    fold = int(config["fold"])
    if fold not in range(5):
        raise ValueError("Fold inválido")
    tr, va = uc.particion(samples, fold)
    for nombre, esperado in (("train", tr), ("val", va)):
        particion = pd.read_csv(directorio / f"particion_{nombre}.csv")
        if particion.sample_id.duplicated().any() or set(particion.sample_id) != set(esperado.sample_id):
            raise ValueError(f"Partición {nombre} incompleta o diferente del fold: {directorio}")
        unido = particion.merge(esperado[["sample_id", "patient_id", "pCR", "fold"]], on="sample_id", suffixes=("", "_ref"), validate="one_to_one")
        if any(not unido[c].equals(unido[c + "_ref"]) for c in ("patient_id", "pCR", "fold")):
            raise ValueError(f"Partición {nombre} con etiquetas o pacientes incoherentes")
    cortes = pd.read_csv(directorio / "predicciones_cortes_val.csv")
    if cortes.sample_id.duplicated().any() or set(cortes.sample_id) != set(va.sample_id):
        raise ValueError("Predicciones de cortes incompletas o duplicadas")
    unido = cortes.merge(va[["sample_id", "patient_id", "pCR"]], on="sample_id", suffixes=("", "_ref"), validate="one_to_one")
    if any(not unido[c].equals(unido[c + "_ref"]) for c in ("patient_id", "pCR")):
        raise ValueError("Predicciones con paciente o etiqueta incorrectos")
    if not np.isfinite(cortes.prob).all() or not cortes.prob.between(0, 1).all():
        raise ValueError("Probabilidades inválidas")
    tabla = cortes.groupby("patient_id").agg(prob=("prob", "mean"), pCR=("pCR", "first")).reset_index()
    guardada = pd.read_csv(directorio / "predicciones_pacientes_mean.csv").set_index("patient_id").sort_index()
    calculada = tabla.set_index("patient_id").sort_index()
    if not calculada.index.equals(guardada.index) or not np.allclose(calculada.prob, guardada.prob, atol=1e-7):
        raise ValueError("La agregación guardada no coincide con la media de cortes")
    tabla = tabla.merge(pacientes[["pid", "dataset"]], left_on="patient_id", right_on="pid", validate="one_to_one").drop(columns="pid")
    if len(tabla) != va.patient_id.nunique():
        raise ValueError("Faltan pacientes en las predicciones")
    ejecucion = leer_json(directorio / "ejecucion.json")
    historial = leer_json(directorio / "historial.json")
    elegida = int(ejecucion["epoca_elegida"])
    if elegida != max(historial, key=lambda h: h["auc"])["epoca"]:
        raise ValueError("Época elegida no coincide con el primer máximo de AUC")
    auc = roc_auc_score(tabla.pCR, tabla.prob)
    fila = next(h for h in historial if h["epoca"] == elegida)
    if not np.isclose(auc, fila["auc"]):
        raise ValueError("AUC exportada no corresponde a la época elegida")
    checkpoint = directorio / ejecucion.get("checkpoint_elegido", "mejor.pt")
    existe_checkpoint = checkpoint.is_file()
    if existe_checkpoint and hashlib.sha256(checkpoint.read_bytes()).hexdigest() != ejecucion["sha256_checkpoint"]:
        raise ValueError("Checksum del checkpoint no coincide; revisar la copia de resultados")
    return dict(directorio=directorio, config=config, tabla=tabla, historial=historial,
                ejecucion=ejecucion, fold=fold, epoca=elegida, checkpoint_presente=existe_checkpoint)


def guardar_figura(fig, ruta):
    fig.tight_layout()
    fig.savefig(ruta, dpi=170, bbox_inches="tight")
    plt.close(fig)


def curvas(ejecucion, destino):
    h = pd.DataFrame(ejecucion["historial"])
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.3))
    if {"loss_train_eval", "loss_val"}.issubset(h.columns):
        ax[0].plot(h.epoca, h.loss_train_eval, label="Train, sin aumentos")
        ax[0].plot(h.epoca, h.loss_val, label="Validación")
        ax[0].set_ylabel("BCE sin ponderar por corte")
    else:
        ax[0].plot(h.epoca, h.loss_train, label="Train, pérdida online")
        ax[0].text(0.04, 0.1, "Pérdida de validación no registrada\nen este entrenamiento antiguo", transform=ax[0].transAxes)
        ax[0].set_ylabel("Pérdida objetivo de entrenamiento")
    ax[1].plot(h.epoca, h.auc, label="Validación, media por paciente", color="#dd6c2f")
    if "auc_train" in h:
        ax[1].plot(h.epoca, h.auc_train, label="Train, media por paciente")
    else:
        ax[1].text(0.04, 0.1, "AUC train no registrada; no se puede\ncalcular la brecha por época", transform=ax[1].transAxes)
    ax[1].axhline(0.5, color="gray", linestyle=":", label="Azar")
    ax[1].set_ylim(0, 1)
    ax[1].set_ylabel("ROC-AUC por paciente")
    for eje in ax:
        eje.axvline(ejecucion["epoca"], color="black", linestyle="--", alpha=0.5, label="Época elegida")
        eje.set_xlabel("Época")
        eje.legend(fontsize=8)
        eje.grid(alpha=0.2)
    fig.suptitle(f"Fold {ejecucion['fold']} · pérdida {ejecucion['config']['perdida']} · validación interna")
    guardar_figura(fig, destino)


def paneles(tablas, destino, resumen):
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    for nombre, tabla in tablas.items():
        color = COLORES[nombre]
        y, p = tabla.pCR.to_numpy(), tabla.prob.to_numpy()
        fpr, tpr, _ = roc_curve(y, p)
        precision, recall, _ = precision_recall_curve(y, p)
        ax[0].plot(fpr, tpr, color=color, label=f"{nombre}: AUC={roc_auc_score(y,p):.3f}")
        ax[1].plot(recall, precision, color=color, label=f"{nombre}: AP={average_precision_score(y,p):.3f}")
        observado, predicho = calibration_curve(y, p, n_bins=8, strategy="uniform")
        ax[2].plot(predicho, observado, "o-", color=color, label=f"{nombre}: Brier={brier_score_loss(y,p):.3f}")
    prevalencia = float(next(iter(tablas.values())).pCR.mean())
    ax[0].plot([0, 1], [0, 1], ":", color="gray", label="Azar")
    ax[1].axhline(prevalencia, linestyle=":", color="gray", label=f"Prevalencia={prevalencia:.3f}")
    ax[2].plot([0, 1], [0, 1], ":", color="gray", label="Calibración ideal")
    titulos = ["ROC", "Precision–recall", "Calibración (8 bins)"]
    etiquetas = [("1 − especificidad", "Sensibilidad"), ("Recall pCR", "Precision pCR"), ("Probabilidad media", "Fracción observada de pCR")]
    for eje, titulo, (x, y) in zip(ax, titulos, etiquetas):
        eje.set(title=titulo, xlabel=x, ylabel=y, xlim=(0, 1), ylim=(0, 1))
        eje.legend(fontsize=8)
        eje.grid(alpha=0.2)
    fig.suptitle("Validación interna por paciente · media de cortes · no es test")
    guardar_figura(fig, destino / "roc_pr_calibracion.png")
    fig, ax = plt.subplots(1, len(tablas), figsize=(6 * len(tablas), 4), squeeze=False)
    for eje, (nombre, tabla) in zip(ax[0], tablas.items()):
        for clase, color in ((0, "#2474b5"), (1, "#dd6c2f")):
            eje.hist(tabla.loc[tabla.pCR == clase, "prob"], bins=np.linspace(0, 1, 26), alpha=0.65,
                     color=color, label=f"Real pCR={clase}")
        eje.axvline(0.5, color="black", linestyle="--", label="Umbral 0.5")
        eje.set(title=nombre, xlabel="Probabilidad media por paciente", ylabel="Pacientes")
        eje.legend()
    guardar_figura(fig, destino / "distribucion_probabilidades.png")
    fig, ax = plt.subplots(len(tablas), 2, figsize=(10, 4 * len(tablas)), squeeze=False)
    for fila, (nombre, r) in zip(ax, resumen.items()):
        for eje, clave, titulo in zip(fila, ("referencia", "umbral_exploratorio"), ("Referencia 0.5", "Umbral exploratorio de validación")):
            m = np.asarray(r[clave]["confusion"])
            eje.imshow(m, cmap="Blues", vmin=0)
            for i in range(2):
                for j in range(2):
                    eje.text(j, i, str(m[i, j]), ha="center", va="center", fontsize=16)
            eje.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["no pCR", "pCR"], yticklabels=["no pCR", "pCR"],
                    xlabel="Predicción", ylabel="Real", title=f"{nombre} · {titulo}\nt={r[clave]['umbral']:.3f}, F1 macro={r[clave]['f1_macro']:.3f}")
    guardar_figura(fig, destino / "matrices_confusion.png")


def umbrales(tabla, nombre, destino):
    # Candidatos predeterminados, evita precisión ficticia de un umbral con tres decimales ajustados.
    valores = np.linspace(0.01, 0.99, 99)
    filas = [metricas(tabla.pCR, tabla.prob, t) for t in valores]
    mejor = max(filas, key=lambda f: (f["f1_macro"], -abs(f["umbral"] - 0.5)))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for clave, etiqueta in (("f1_macro", "F1 macro"), ("sensibilidad", "Sensibilidad pCR"), ("especificidad", "Especificidad")):
        ax.plot(valores, [f[clave] for f in filas], label=etiqueta)
    ax.axvline(0.5, color="gray", linestyle=":", label="Referencia 0.5")
    ax.axvline(mejor["umbral"], color="black", linestyle="--", label=f"Exploratorio t={mejor['umbral']:.2f}")
    ax.set(xlabel="Umbral", ylabel="Métrica por paciente", ylim=(0, 1), title=f"{nombre} · ajuste exploratorio en validación\nLas métricas ajustadas son optimistas; no es una decisión clínica")
    ax.legend()
    ax.grid(alpha=0.2)
    guardar_figura(fig, destino / f"umbrales_{nombre}.png")
    pd.DataFrame([{k: f[k] for k in ("umbral", "f1_macro", "sensibilidad", "especificidad", "accuracy")} for f in filas]).to_csv(destino / f"umbrales_{nombre}.csv", index=False)
    return mejor


def casos_fallados(tabla, nombre, samples, raiz, destino, umbral=0.5):
    fallos = tabla.assign(pred=(tabla.prob >= umbral).astype(int))
    fallos = fallos[fallos.pCR != fallos.pred].copy()
    fallos["tipo"] = np.where(fallos.pred == 1, "FP", "FN")
    fallos["distancia_umbral"] = abs(fallos.prob - umbral)
    fallos.sort_values("distancia_umbral", ascending=False).to_csv(destino / f"fallos_{nombre}.csv", index=False)
    seleccion = fallos.sort_values("distancia_umbral", ascending=False).groupby("tipo").head(2)
    if seleccion.empty:
        return
    fig, ax = plt.subplots(len(seleccion), 4, figsize=(12, 3 * len(seleccion)), squeeze=False)
    for ejes, paciente in zip(ax, seleccion.itertuples()):
        filas = samples[samples.patient_id == paciente.patient_id].sort_values("slice_index")
        fila = filas.iloc[len(filas) // 2]
        x = uc.cargar_imagen(fila, raiz)
        for eje, fase, etiqueta in zip(ejes[:3], x, uc.FASES):
            eje.imshow(fase, cmap="gray", vmin=0, vmax=1)
            eje.set_title(etiqueta)
            eje.axis("off")
        realce = x[1] - x[0]
        limite = max(float(abs(realce).max()), 1 / 255)
        ejes[3].imshow(realce, cmap="coolwarm", vmin=-limite, vmax=limite)
        ejes[3].set_title("EARLY − PRE")
        ejes[3].axis("off")
        ejes[0].set_title(f"{paciente.patient_id} · {paciente.tipo}\nreal={paciente.pCR}, prob={paciente.prob:.3f} · PRE", fontsize=9)
    fig.suptitle(f"Errores {nombre}, t={umbral:.2f} · un corte representativo, predicción con todos los cortes")
    guardar_figura(fig, destino / f"casos_fallados_{nombre}.png")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", type=Path, nargs="+", required=True)
    p.add_argument("--datos", type=Path, default=Path("breastdcedl"))
    p.add_argument("--salida", type=Path, required=True)
    p.add_argument("--bootstrap", type=int, default=2000)
    args = p.parse_args()
    if args.bootstrap < 100:
        p.error("Usar al menos 100 réplicas de bootstrap")
    if args.salida.exists():
        p.error("La carpeta de análisis ya existe; elige otra para conservar resultados")
    samples = uc.cargar_samples(args.datos)
    pacientes = uc.cargar_patients(args.datos)
    muestras_train = samples[samples.split == "train"]
    pacientes_train = pacientes[pacientes.split == "train"]
    ejecuciones, claves = [], set()
    for raiz_run in args.runs:
        for nombre in ("normal", "ponderada"):
            directorio = raiz_run / nombre
            if not directorio.is_dir():
                continue
            ejecucion = cargar_ejecucion(directorio, samples, pacientes_train)
            clave = (nombre, ejecucion["fold"])
            if clave in claves:
                p.error(f"Fold duplicado para la misma pérdida: {clave}")
            claves.add(clave)
            ejecuciones.append(ejecucion)
    if not ejecuciones:
        p.error("No hay experimentos completos")
    campos_receta = ("semilla", "lr", "dropout", "weight_decay", "batch_size", "epocas", "paciencia", "parametros", "sha256_modelo")
    recetas = {tuple(e["config"].get(c) for c in campos_receta) for e in ejecuciones}
    if len(recetas) != 1:
        p.error("Las ejecuciones tienen recetas o arquitecturas distintas; analizarlas por separado")
    args.salida.mkdir(parents=True)
    tablas, resumen, filas_cv, filas_cohorte = {}, {}, [], []
    for e in ejecuciones:
        nombre = e["config"]["perdida"]
        curvas(e, args.salida / f"curvas_fold{e['fold']}_{nombre}.png")
        m = metricas(e["tabla"].pCR, e["tabla"].prob, 0.5)
        elegida = next(h for h in e["historial"] if h["epoca"] == e["epoca"])
        auc_train = elegida.get("auc_train")
        filas_cv.append(dict(perdida=nombre, fold=e["fold"], epoca=e["epoca"],
                             auc_train_elegida=auc_train,
                             brecha_auc_train_val=(auc_train - m["auc"]) if auc_train is not None else None,
                             loss_train_eval_elegida=elegida.get("loss_train_eval"),
                             loss_val_elegida=elegida.get("loss_val"),
                             **{k: m[k] for k in ("auc", "accuracy", "sensibilidad", "especificidad", "f1_macro", "balanced_accuracy")}))
    for nombre in sorted({e["config"]["perdida"] for e in ejecuciones}):
        es = [e for e in ejecuciones if e["config"]["perdida"] == nombre]
        firmas = {(e["config"]["semilla"], e["config"]["lr"], e["config"]["dropout"], e["config"]["batch_size"]) for e in es}
        if len(firmas) != 1:
            raise ValueError("No combinar folds con recetas distintas")
        tabla = pd.concat([e["tabla"].assign(fold=e["fold"]) for e in es], ignore_index=True)
        if tabla.patient_id.duplicated().any():
            raise ValueError("Paciente repetida en predicciones fuera de fold")
        tablas[nombre] = tabla
        tabla.to_csv(args.salida / f"predicciones_validacion_{nombre}.csv", index=False)
        referencia = metricas(tabla.pCR, tabla.prob, 0.5)
        referencia["auc_ic95_bootstrap"] = intervalo_auc(tabla.pCR, tabla.prob, args.bootstrap)
        exploratorio = umbrales(tabla, nombre, args.salida)
        resumen[nombre] = dict(folds=sorted(e["fold"] for e in es), cv_completa=len(es) == 5,
                               referencia=referencia, umbral_exploratorio=exploratorio,
                               advertencia_umbral="Maximiza F1 macro sobre la misma validación; métricas ajustadas optimistas. No es umbral final ni clínico.",
                               checkpoints_presentes=all(e["checkpoint_presente"] for e in es),
                               baseline_siempre_no_pcr=metricas(tabla.pCR, np.zeros(len(tabla)), 0.5))
        for cohorte, grupo in tabla.groupby("dataset"):
            r = metricas(grupo.pCR, grupo.prob, 0.5) if grupo.pCR.nunique() == 2 else None
            filas_cohorte.append(dict(perdida=nombre, cohorte=cohorte, pacientes=len(grupo),
                                      auc=r["auc"] if r else None, sensibilidad=r["sensibilidad"] if r else None,
                                      especificidad=r["especificidad"] if r else None))
        casos_fallados(tabla, nombre, muestras_train, args.datos, args.salida)
    if len(tablas) == 2 and set(tablas["normal"].patient_id) != set(tablas["ponderada"].patient_id):
        raise ValueError("Las pérdidas deben compararse en las mismas pacientes/folds")
    paneles(tablas, args.salida, resumen)
    cv = pd.DataFrame(filas_cv)
    cv.to_csv(args.salida / "metricas_por_fold.csv", index=False)
    estadisticas = cv.groupby("perdida")[["auc", "accuracy", "sensibilidad", "especificidad", "f1_macro"]].agg(["mean", "std", "count"])
    estadisticas.to_csv(args.salida / "resumen_folds.csv")
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    for posicion, nombre in enumerate(tablas):
        f = cv[cv.perdida == nombre].sort_values("fold")
        ax[0].plot(f.fold, f.auc, "o-", label=nombre, color=COLORES[nombre])
        dispersion = float(f.auc.std(ddof=1)) if len(f) > 1 else None
        ax[1].bar(posicion, f.auc.mean(), yerr=dispersion, capsize=5, color=COLORES[nombre])
        ax[1].text(posicion, f.auc.mean() + (dispersion or 0) + 0.025, f"n={len(f)} folds", ha="center")
    ax[0].set(xticks=range(5), xlabel="Fold", ylabel="ROC-AUC por paciente", ylim=(0, 1), title="Resultados disponibles a umbral 0.5")
    ax[0].legend()
    ax[1].set(xticks=range(len(tablas)), xticklabels=list(tablas), ylabel="ROC-AUC por paciente", ylim=(0, 1),
              title="Media ± desviación entre folds\nUn único fold no tiene desviación estimable")
    guardar_figura(fig, args.salida / "auc_folds_media_desviacion.png")
    cohortes = pd.DataFrame(filas_cohorte)
    cohortes.to_csv(args.salida / "metricas_por_cohorte.csv", index=False)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for nombre in tablas:
        c = cohortes[cohortes.perdida == nombre]
        ax.plot(c.cohorte, c.auc, "o-", color=COLORES[nombre], label=nombre)
    ax.axhline(0.5, color="gray", linestyle=":")
    ax.set(ylabel="ROC-AUC por paciente", ylim=(0, 1), title="Validación por cohorte · grupos pequeños, sin comparación causal")
    ax.legend()
    guardar_figura(fig, args.salida / "auc_por_cohorte.png")
    guardar_json(args.salida / "resumen.json", resumen)
    notas = ["# Análisis de validación interna", "", "Agregación: media de cortes. Test no cargado ni evaluado.", ""]
    for nombre, r in resumen.items():
        m, intervalo = r["referencia"], r["referencia"]["auc_ic95_bootstrap"]
        notas.extend([f"## {nombre}", "", f"Folds disponibles: {r['folds']}; CV completa: {r['cv_completa']}.",
                      f"AUC {m['auc']:.3f}, IC95% bootstrap por paciente [{intervalo[0]:.3f}, {intervalo[1]:.3f}].",
                      f"Umbral 0.5: sensibilidad {m['sensibilidad']:.3f}, especificidad {m['especificidad']:.3f}, F1 macro {m['f1_macro']:.3f}.",
                      f"Checkpoint disponible: {r['checkpoints_presentes']}.", ""])
    notas.extend(["## Límites", "", "Con un fold no existe media ± desviación de cinco folds; la desviación de un único fold queda vacía.",
                  "Los intervalos describen incertidumbre de pacientes para modelos ya elegidos; no incluyen la selección de época ni variabilidad del entrenamiento.",
                  "El umbral exploratorio optimiza F1 macro en la misma validación y no aporta una evaluación independiente.",
                  "F1 macro es la media del F1 de ambas clases. No hay un F1 clínicamente óptimo universal.",
                  "Los historiales antiguos no permiten reconstruir pérdidas de validación o AUC train por época.",
                  "No deducir sobreajuste demostrado sin las curvas comparables que faltan; una AUC val que cae mientras baja train loss es una señal para investigar.",
                  "Resultados y figuras locales: mantener fuera de Git."])
    (args.salida / "INFORME.md").write_text("\n".join(notas) + "\n", encoding="utf-8")
    print(f"Análisis y PNG generados en {args.salida}")


if __name__ == "__main__":
    main()
