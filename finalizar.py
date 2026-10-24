"""Cierre con CV, reentrenamiento en todo train y evaluación final sin reajustes."""
import argparse
import hashlib
from importlib.metadata import version
import json
import math
import platform
import time
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.calibration import calibration_curve
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_curve, f1_score

from analizar_resultados import cargar_ejecucion, guardar_figura, intervalo_auc, metricas
from entrenar import (uc, fijar_semilla, guardar_json, evaluar_epoca, loader,
                      limitar_para_prueba, predecir, sincronizar)
from modelo import CNNBreastDCE

REPO = Path(__file__).resolve().parent


def sha(ruta):
    return hashlib.sha256(Path(ruta).read_bytes()).hexdigest()


def sha_modelo():
    return hashlib.sha256((REPO / "modelo.py").read_text(encoding="utf-8").replace("\r\n", "\n").encode()).hexdigest()


def leer_json(ruta):
    return json.loads(Path(ruta).read_text(encoding="utf-8-sig"))


def validar_receta(r):
    if r.get("version") != 1 or r.get("modo") != "cnn_unica_todo_train":
        raise ValueError("Receta final no reconocida")
    if r.get("agregacion") != "mean" or r.get("canales") != ["PRE", "EARLY", "LATE"]:
        raise ValueError("Agregación o fases distintas a las del cierre")
    if r.get("normalizacion") != "float32 / 255":
        raise ValueError("Normalización distinta a la del cierre")
    if r.get("perdida") not in ("normal", "ponderada"):
        raise ValueError("Pérdida inválida")
    if not isinstance(r.get("epocas"), int) or r["epocas"] < 1:
        raise ValueError("Número de épocas inválido")
    if not 0 < r.get("umbral", -1) < 1 or not math.isfinite(r["umbral"]):
        raise ValueError("Umbral inválido")
    if r.get("sha256_modelo_texto") != sha_modelo():
        raise ValueError("El código de la CNN ha cambiado respecto a la receta")


def cargar_receta(ruta):
    ruta = Path(ruta)
    esperado = ruta.with_suffix(".sha256").read_text(encoding="utf-8").strip()
    if sha(ruta) != esperado:
        raise ValueError("La receta se ha modificado después del cierre")
    r = leer_json(ruta)
    validar_receta(r)
    return r


def preparar(args):
    samples = uc.cargar_samples(args.datos)
    pacientes = uc.cargar_patients(args.datos)
    pacientes = pacientes[pacientes.split == "train"]
    ejecuciones = [cargar_ejecucion(args.cv / f"fold{f}_semilla42" / perdida, samples, pacientes)
                   for f in range(5) for perdida in ("normal", "ponderada")]
    if any(e["config"].get("clinica") for e in ejecuciones):
        raise ValueError("La comparación clínica no es compatible con el cierre de la CNN solo imagen. No evaluar test.")
    if any(e["config"].get("arquitectura", "simple") != "simple" for e in ejecuciones):
        raise ValueError("El cierre histórico solo admite la CNN simple; comparar la nueva arquitectura en validación, sin test.")
    if not all(e["checkpoint_presente"] for e in ejecuciones):
        raise ValueError("Faltan pesos de la validación cruzada")
    campos = ("lr", "dropout", "weight_decay", "batch_size", "semilla", "parametros", "sha256_modelo", "sha256_samples")
    if len({tuple(e["config"].get(c) for c in campos) for e in ejecuciones}) != 1:
        raise ValueError("No mezclar recetas o arquitecturas distintas entre folds")
    resumen, tablas, registros = {}, {}, []
    for perdida in ("normal", "ponderada"):
        es = [e for e in ejecuciones if e["config"]["perdida"] == perdida]
        tabla = pd.concat([e["tabla"].assign(fold=e["fold"]) for e in es], ignore_index=True)
        if tabla.patient_id.duplicated().any() or set(tabla.patient_id) != set(pacientes.pid):
            raise ValueError("Las predicciones fuera de fold no cubren train una única vez")
        tablas[perdida] = tabla
        aucs = []
        for e in es:
            m = metricas(e["tabla"].pCR, e["tabla"].prob, 0.5)
            h = next(h for h in e["historial"] if h["epoca"] == e["epoca"])
            if "auc_train" not in h or "loss_val" not in h:
                raise ValueError("Se requieren las curvas completas de cv5_v2")
            aucs.append(m["auc"])
            registros.append(dict(perdida=perdida, fold=e["fold"], epoca=e["epoca"], auc=m["auc"],
                                  auc_train=h["auc_train"], brecha_auc=h["auc_train"] - m["auc"],
                                  f1_macro=m["f1_macro"], sensibilidad=m["sensibilidad"], especificidad=m["especificidad"]))
        resumen[perdida] = dict(auc_media=float(np.mean(aucs)), auc_std=float(np.std(aucs, ddof=1)),
                                epocas=[e["epoca"] for e in es])
    elegida = max(("normal", "ponderada"), key=lambda l: resumen[l]["auc_media"])
    tabla = tablas[elegida]
    candidatos = np.linspace(0.01, 0.99, 99)
    umbral = max(candidatos, key=lambda t: (f1_score(tabla.pCR, tabla.prob >= t, average="macro", zero_division=0), -abs(t - 0.5)))
    base = ejecuciones[0]["config"]
    if base["sha256_modelo"] not in {sha(REPO / "modelo.py"), sha_modelo()}:
        raise ValueError("La arquitectura actual no coincide con el código utilizado en CV")
    epocas = int(np.median(resumen[elegida]["epocas"]))
    receta = dict(version=1, modo="cnn_unica_todo_train", perdida=elegida, epocas=epocas,
                  umbral=float(umbral), agregacion="mean", canales=["PRE", "EARLY", "LATE"], normalizacion="float32 / 255",
                  **{c: base[c] for c in ("lr", "dropout", "weight_decay", "batch_size", "semilla", "parametros")},
                  pacientes_train=len(pacientes), sha256_samples=sha(args.datos / "metadata/samples.csv"),
                  sha256_modelo_texto=sha_modelo(), resumen_cv=resumen,
                  criterio_perdida="Mayor ROC-AUC media de cinco folds; diferencia no implica superioridad estadística",
                  criterio_epocas="Mediana de las épocas elegidas en los cinco folds de la pérdida seleccionada",
                  criterio_umbral="Máximo F1 macro en OOF, rejilla 0.01..0.99; empate más próximo a 0.5",
                  limites="Las métricas OOF ajustadas son optimistas; el umbral se transfiere al reentrenamiento y no se reajusta con test.")
    validar_receta(receta)
    args.salida.mkdir(parents=True, exist_ok=False)
    guardar_json(args.salida / "receta.json", receta)
    (args.salida / "receta.sha256").write_text(sha(args.salida / "receta.json") + "\n", encoding="utf-8")
    pd.DataFrame(registros).to_csv(args.salida / "comparacion_cv.csv", index=False)
    tabla.to_csv(args.salida / "oof_perdida_elegida.csv", index=False)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    for perdida in resumen:
        filas = [r for r in registros if r["perdida"] == perdida]
        ax[0].plot([r["fold"] for r in filas], [r["auc"] for r in filas], "o-", label=perdida)
    ax[0].set(xlabel="Fold", ylabel="ROC-AUC por paciente", ylim=(0.45, 0.8), title="Cinco folds: comparación de pérdidas")
    ax[0].legend()
    ax[1].bar(range(5), resumen[elegida]["epocas"])
    ax[1].axhline(epocas, color="orange", linestyle="--", label=f"Mediana: {epocas} épocas")
    ax[1].set(xlabel="Fold", ylabel="Época elegida", title=f"Regla de entrenamiento final · {elegida}")
    ax[1].legend()
    guardar_figura(fig, args.salida / "cierre_cv.png")
    texto = (f"# Cierre anterior a test\n\nCNN única con todo train. Pérdida: {elegida}. "
             f"Épocas: {epocas}. Umbral: {umbral:.2f}. Agregación: media de cortes.\n\n"
             "La pérdida se elige por AUC media; el umbral maximiza F1 macro en OOF. "
             "No son métricas independientes del ajuste. No existe un coste clínico validado: "
             "F1 macro da el mismo peso a las dos clases para esta comparación educativa.\n\n"
             "La AUC media entre folds y la AUC OOF combinada son medidas diferentes. "
             "El modelo reentrenado puede cambiar la distribución de scores; no ajustar el umbral con test.\n\n"
             "La receta y su checksum deben viajar juntos a la universidad o regenerarse con el mismo CV. "
             "No hay entrenamiento ni evaluación de test en esta preparación.\n")
    (args.salida / "CIERRE.md").write_text(texto, encoding="utf-8")
    print(f"Receta cerrada: perdida={elegida}, epocas={epocas}, umbral={umbral:.2f}. Guardada en {args.salida}")


def entrenar_final(args):
    r = cargar_receta(args.receta)
    if args.dispositivo == "cuda" and not torch.cuda.is_available():
        raise ValueError("La GPU no está disponible; conservar y diagnosticar el entorno de clase")
    if sha(args.datos / "metadata/samples.csv") != r["sha256_samples"]:
        raise ValueError("El índice del dataset no coincide con el de la receta")
    if (REPO / "runs/test_una_vez.json").exists():
        raise ValueError("Test ya se ha iniciado en este proyecto; no reentrenar después de verlo")
    samples = uc.cargar_samples(args.datos)
    tr = samples[samples.split == "train"].copy()
    if tr.patient_id.nunique() != r["pacientes_train"] or (tr.groupby("patient_id").pCR.nunique() != 1).any():
        raise ValueError("Pacientes de entrenamiento inconsistentes")
    if args.prueba:
        tr = limitar_para_prueba(tr, negativos=3)
    args.salida.mkdir(parents=True, exist_ok=False)
    (args.salida / "checkpoints").mkdir()
    guardar_json(args.salida / "receta.json", r)
    fijar_semilla(r["semilla"])
    dispositivo = torch.device(args.dispositivo)
    modelo = CNNBreastDCE(r["dropout"]).to(dispositivo)
    peso = uc.pos_weight(tr) if r["perdida"] == "ponderada" else 1.0
    criterio = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(peso, device=dispositivo))
    optimizador = torch.optim.Adam(modelo.parameters(), lr=r["lr"], weight_decay=r["weight_decay"])
    dl_args = SimpleNamespace(batch_size=4 if args.prueba else r["batch_size"], workers=args.workers,
                              semilla=r["semilla"], dispositivo=args.dispositivo)
    dl = loader(tr, args.datos, dl_args, True)
    dl_eval = loader(tr, args.datos, dl_args, False)
    tr[["sample_id", "patient_id", "pCR", "fold"]].to_csv(args.salida / "particion_train.csv", index=False)
    entorno = dict(torch=str(torch.__version__), hip=torch.version.hip, python=platform.python_version(),
                   dispositivo=str(dispositivo), gpu=torch.cuda.get_device_name(0) if dispositivo.type == "cuda" else None)
    entorno["paquetes"] = {nombre: version(nombre) for nombre in ("numpy", "pandas", "pillow", "scikit-learn")}
    entorno["sha256_finalizar"] = sha(Path(__file__))
    guardar_json(args.salida / "entorno.json", entorno)
    if dispositivo.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dispositivo)
    epocas = min(2, r["epocas"]) if args.prueba else r["epocas"]
    historial, mensajes = [], []

    def informar(texto):
        print(texto, flush=True)
        mensajes.append(texto)
        (args.salida / "entrenamiento.log").write_text("\n".join(mensajes) + "\n", encoding="utf-8")

    informar(f"CNN final | pacientes={tr.patient_id.nunique()} | parametros={r['parametros']} | perdida={r['perdida']} | epocas_fijas={epocas} | dispositivo={dispositivo} | gpu={entorno['gpu']} | prueba={args.prueba}")
    informar("Sin validacion durante el reentrenamiento; test no se utiliza.")
    sincronizar(dispositivo)
    total = time.perf_counter()
    for epoca in range(1, epocas + 1):
        inicio, acumulada, n = time.perf_counter(), 0.0, 0
        modelo.train()
        for x, y in dl:
            x, y = x.to(dispositivo), y.to(dispositivo)
            optimizador.zero_grad(set_to_none=True)
            loss = criterio(modelo(x), y)
            if not torch.isfinite(loss):
                raise ValueError("Pérdida no finita")
            loss.backward()
            optimizador.step()
            acumulada += loss.item() * len(y)
            n += len(y)
        evaluacion = evaluar_epoca(modelo, dl_eval, tr, dispositivo, peso)
        sincronizar(dispositivo)
        historial.append(dict(epoca=epoca, loss_online=acumulada/n, segundos=time.perf_counter()-inicio, **evaluacion))
        guardar_json(args.salida / "historial.json", historial)
        checkpoint = args.salida / "checkpoints" / f"epoca_{epoca:03d}.pt"
        temporal = checkpoint.with_suffix(".pt.tmp")
        torch.save(dict(tipo="cnn_final", prueba=args.prueba, receta=r, epoca=epoca,
                        pacientes_train=tr.patient_id.nunique(), state_dict=modelo.state_dict(),
                        optimizer_state_dict=optimizador.state_dict()), temporal)
        temporal.replace(checkpoint)
        informar(f"epoch={epoca:03d} train_loss={acumulada/n:.4f} train_eval_loss={evaluacion['bce']:.4f} train_roc_auc={evaluacion['auc']:.4f} train_pr_auc={evaluacion['pr_auc']:.4f} train_AP={evaluacion['average_precision']:.4f} train_f1_macro_t05={evaluacion['f1_macro']:.4f} lr={r['lr']:g} time={historial[-1]['segundos']:.1f}s checkpoint={checkpoint.name}")
    guardar_json(args.salida / "artefacto.json", dict(checkpoint=str(checkpoint.relative_to(args.salida)).replace("\\", "/"),
                 sha256_checkpoint=sha(checkpoint), prueba=args.prueba, epocas=epocas, umbral=r["umbral"],
                 segundos=time.perf_counter()-total,
                 memoria_gpu_max_bytes=torch.cuda.max_memory_allocated(dispositivo) if dispositivo.type == "cuda" else None))
    fig, ejes = plt.subplots(1, 2, figsize=(11, 4))
    for ax in ejes:
        ax.plot(range(1, epocas+1), [h["bce"] for h in historial], "o-", label="BCE train, sin aumentos")
        ax.set(xlabel="Época", ylabel="BCE", xticks=range(1,epocas+1))
        ax.legend()
    ejes[0].set(title="Escala con contexto", ylim=(0,max(0.8,max(h["bce"] for h in historial)*1.1)))
    ejes[1].set_title("Detalle de la misma curva")
    fig.suptitle("Reentrenamiento final · solo train, sin selección por test")
    guardar_figura(fig, args.salida / "curva_train_final.png")
    informar(f"Finalizado. Última época fijada por CV: {checkpoint}. Llevar la carpeta completa al portátil.")


def validar_checkpoint_final(cp):
    if cp.get("tipo") != "cnn_final" or cp.get("prueba", True):
        raise ValueError("No es un modelo final completo; no evaluar pruebas técnicas ni modelos de fold")
    r = cp["receta"]
    validar_receta(r)
    if cp.get("epoca") != r["epocas"] or cp.get("pacientes_train") != r["pacientes_train"]:
        raise ValueError("No corresponde a la última época fijada ni a todo train")
    return r


def cargar_modelo_final(carpeta, dispositivo="cpu"):
    carpeta = Path(carpeta)
    a = leer_json(carpeta / "artefacto.json")
    checkpoint = (carpeta / a["checkpoint"]).resolve()
    if not checkpoint.is_relative_to(carpeta.resolve()) or sha(checkpoint) != a["sha256_checkpoint"]:
        raise ValueError("Ruta o checksum del modelo incorrectos")
    cp = torch.load(checkpoint, map_location=dispositivo, weights_only=True)
    r = validar_checkpoint_final(cp)
    modelo = CNNBreastDCE(r["dropout"]).to(dispositivo)
    modelo.load_state_dict(cp["state_dict"])
    modelo.eval()
    return modelo, r, a


def evaluar_test(args):
    if args.dispositivo == "cpu":
        torch.set_num_threads(getattr(args,"threads_cpu",2))
    modelo, r, a = cargar_modelo_final(args.modelo, args.dispositivo)
    if sha(args.datos / "metadata/samples.csv") != r["sha256_samples"]:
        raise ValueError("Índice de datos diferente del cierre")
    registro = REPO / "runs/test_una_vez.json"
    if registro.exists() or args.salida.exists():
        raise ValueError("Ya hay un registro de test o salida existente: revisar los resultados guardados, sin repetir ni reajustar")
    muestras = uc.cargar_samples(args.datos)
    test = uc.conjunto_test(muestras)
    if test.empty or set(test.patient_id) & set(muestras[muestras.split == "train"].patient_id):
        raise ValueError("Test vacío o con fuga por paciente")
    args.salida.mkdir(parents=True, exist_ok=False)
    registro.parent.mkdir(parents=True, exist_ok=True)
    with registro.open("x", encoding="utf-8") as archivo:
        json.dump(dict(estado="iniciada", sha256_checkpoint=a["sha256_checkpoint"], umbral=r["umbral"], salida=str(args.salida.resolve())), archivo, indent=2)
    dispositivo = torch.device(args.dispositivo)
    dl_args = SimpleNamespace(batch_size=r["batch_size"], workers=0, semilla=r["semilla"], dispositivo=args.dispositivo)
    sincronizar(dispositivo)
    inicio_inferencia = time.perf_counter()
    probs = predecir(modelo, loader(test, args.datos, dl_args, False), dispositivo)
    sincronizar(dispositivo)
    segundos_inferencia = time.perf_counter()-inicio_inferencia
    cortes = test[["sample_id", "patient_id", "pCR"]].copy()
    cortes["prob"] = probs
    cortes.to_csv(args.salida / "predicciones_cortes.csv", index=False)
    tabla = uc.agregar_por_paciente(test.patient_id, probs).join(test.groupby("patient_id").pCR.first()).reset_index()
    pacientes = uc.cargar_patients(args.datos)
    tabla = tabla.merge(pacientes[["pid", "dataset"]], left_on="patient_id", right_on="pid", validate="one_to_one").drop(columns="pid")
    tabla["pred"] = (tabla.prob >= r["umbral"]).astype(int)
    tabla.to_csv(args.salida / "predicciones_pacientes.csv", index=False)
    m = metricas(tabla.pCR, tabla.prob, r["umbral"])
    m["auc_ic95_bootstrap"] = intervalo_auc(tabla.pCR, tabla.prob, 2000)
    m["baseline_siempre_no_pcr"] = metricas(tabla.pCR, np.zeros(len(tabla)), 0.5)
    m["average_precision"] = float(average_precision_score(tabla.pCR, tabla.prob))
    m["sha256_checkpoint"] = a["sha256_checkpoint"]
    m["segundos_inferencia_con_carga"] = segundos_inferencia
    m["entorno_inferencia"] = dict(torch=str(torch.__version__), dispositivo=args.dispositivo,
                                   threads_cpu=torch.get_num_threads(), python=platform.python_version())
    guardar_json(args.salida / "metricas_test.json", m)
    por_cohorte = {}
    for cohorte, g in tabla.groupby("dataset"):
        por_cohorte[cohorte] = metricas(g.pCR, g.prob, r["umbral"]) if g.pCR.nunique() == 2 else dict(pacientes=len(g), auc=None)
    guardar_json(args.salida / "metricas_por_cohorte.json", por_cohorte)
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    fpr, tpr, _ = roc_curve(tabla.pCR, tabla.prob)
    precision, recall, _ = precision_recall_curve(tabla.pCR, tabla.prob)
    ax[0].plot(fpr, tpr, label=f"AUC={m['auc']:.3f}")
    ax[0].plot([0, 1], [0, 1], ":", color="gray")
    ax[0].set(xlabel="1 − especificidad", ylabel="Sensibilidad", title="ROC · test final")
    ax[0].legend()
    ax[1].plot(recall, precision, label=f"AP={m['average_precision']:.3f}")
    ax[1].axhline(tabla.pCR.mean(), linestyle=":", color="gray")
    ax[1].set(xlabel="Recall pCR", ylabel="Precision pCR", title="Precision–recall · test final")
    ax[1].legend()
    observado, predicho = calibration_curve(tabla.pCR, tabla.prob, n_bins=8)
    ax[2].plot(predicho, observado, "o-")
    ax[2].plot([0, 1], [0, 1], ":", color="gray")
    ax[2].set(xlabel="Probabilidad media", ylabel="pCR observada", title=f"Calibración · Brier={m['brier']:.3f}", xlim=(0, 1), ylim=(0, 1))
    guardar_figura(fig, args.salida / "roc_pr_calibracion_test.png")
    fig, ax = plt.subplots(figsize=(5, 4.5))
    matriz = np.asarray(m["confusion"])
    ax.imshow(matriz, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(matriz[i,j]), ha="center", va="center", fontsize=17)
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["no pCR", "pCR"], yticklabels=["no pCR", "pCR"],
           xlabel="Predicción", ylabel="Real", title=f"Test final · umbral fijo {r['umbral']:.2f}")
    guardar_figura(fig, args.salida / "matriz_confusion_test.png")
    guardar_json(registro, dict(estado="completada", sha256_checkpoint=a["sha256_checkpoint"], umbral=r["umbral"], salida=str(args.salida.resolve())))
    print(f"Test completado una vez. AUC={m['auc']:.3f}, F1 macro={m['f1_macro']:.3f}. Resultados en {args.salida}. No reajustar el modelo.")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="accion", required=True)
    prep = sub.add_parser("preparar", help="Verifica cinco folds y cierra la receta sin consultar test")
    prep.add_argument("--cv", type=Path, required=True)
    prep.add_argument("--datos", type=Path, default=Path("breastdcedl"))
    prep.add_argument("--salida", type=Path, required=True)
    prep.set_defaults(func=preparar)
    train = sub.add_parser("entrenar", help="GPU de clase: CNN única sobre todo train, épocas fijas")
    train.add_argument("--receta", type=Path, required=True)
    train.add_argument("--datos", type=Path, default=Path("breastdcedl"))
    train.add_argument("--salida", type=Path, required=True)
    train.add_argument("--dispositivo", choices=["cpu", "cuda"], default="cuda")
    train.add_argument("--workers", type=int, default=2)
    train.add_argument("--prueba", action="store_true", help="Dos épocas con cinco pacientes; el resultado no se puede evaluar en test")
    train.set_defaults(func=entrenar_final)
    test = sub.add_parser("evaluar", help="Solo después del reentrenamiento final; una evaluación de test")
    test.add_argument("--modelo", type=Path, required=True)
    test.add_argument("--datos", type=Path, default=Path("breastdcedl"))
    test.add_argument("--salida", type=Path, default=Path("runs/test_final"))
    test.add_argument("--dispositivo", choices=["cpu", "cuda"], default="cpu")
    test.add_argument("--threads-cpu", type=int, default=2)
    test.set_defaults(func=evaluar_test)
    args = p.parse_args()
    if getattr(args, "workers", 0) < 0:
        p.error("workers debe ser no negativo")
    if getattr(args,"threads_cpu",2) < 1:
        p.error("threads-cpu debe ser positivo")
    try:
        args.func(args)
    except (OSError, ValueError) as exc:
        p.exit(1, f"No se ha completado {args.accion}: {exc}\n")


if __name__ == "__main__":
    main()
