"""Audita cinco pesos y recalcula TRAIN final; --todas-las-epocas recalcula todas."""
import argparse
import json
import math
import time
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, auc, log_loss

from analizar_resultados import guardar_figura, metricas
from entrenar import uc, loader, guardar_json
from finalizar import cargar_modelo_final, sha, leer_json, validar_receta
from modelo import CNNBreastDCE


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--modelo", type=Path, default=Path("runs/modelo_final"))
    p.add_argument("--datos", type=Path, default=Path("breastdcedl"))
    p.add_argument("--salida", type=Path, default=Path("runs/auditoria_modelo_final"))
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--todas-las-epocas", action="store_true")
    args = p.parse_args()
    if args.salida.exists() or args.threads < 1:
        p.error("Usa una carpeta de salida nueva y threads positivo")
    torch.set_num_threads(args.threads)
    _, receta, artefacto = cargar_modelo_final(args.modelo)
    validar_receta(receta)
    if sha(args.datos / "metadata/samples.csv") != receta["sha256_samples"]:
        raise ValueError("Índice de datos distinto del cierre")
    samples = uc.cargar_samples(args.datos)
    tr = samples[samples.split == "train"].reset_index(drop=True)
    particion = pd.read_csv(args.modelo / "particion_train.csv")
    if particion.sample_id.duplicated().any() or set(particion.sample_id) != set(tr.sample_id):
        raise ValueError("No se entrenó con todos los cortes de train exactamente una vez por época")
    unido = particion.merge(tr[["sample_id", "patient_id", "pCR", "fold"]], on="sample_id", validate="one_to_one", suffixes=("", "_ref"))
    if any(not unido[c].equals(unido[c+"_ref"]) for c in ("patient_id", "pCR", "fold")):
        raise ValueError("Etiquetas o particiones incoherentes")
    if tr.patient_id.nunique() != receta["pacientes_train"]:
        raise ValueError("Número de pacientes incorrecto")
    if (tr.groupby("patient_id").pCR.nunique() != 1).any():
        raise ValueError("Etiqueta no constante por paciente")
    h = leer_json(args.modelo / "historial.json")
    if [f["epoca"] for f in h] != list(range(1, receta["epocas"]+1)):
        raise ValueError("Historial de épocas incompleto")
    for fila in h:
        c = fila["matriz_confusion"]
        tp, tn, fp, fn = c["VP"], c["VN"], c["FP"], c["FN"]
        if tp+tn+fp+fn != receta["pacientes_train"]:
            raise ValueError("La matriz de confusión no cubre todos los pacientes")
        esperadas = dict(accuracy=(tp+tn)/(tp+tn+fp+fn), sensibilidad=tp/(tp+fn),
                         especificidad=tn/(tn+fp),
                         f1_macro=(2*tp/(2*tp+fp+fn)+2*tn/(2*tn+fp+fn))/2)
        if any(abs(fila[k]-v)>1e-10 for k,v in esperadas.items()):
            raise ValueError("Error aritmético en métricas del historial")
    modelos, comprobaciones, hashes = [], [], set()
    pasos_epoca = math.ceil(len(tr) / receta["batch_size"])
    for epoca in range(1, receta["epocas"]+1):
        ruta = args.modelo / "checkpoints" / f"epoca_{epoca:03d}.pt"
        cp = torch.load(ruta, map_location="cpu", weights_only=True)
        if cp.get("tipo") != "cnn_final" or cp.get("prueba") is not False or cp["epoca"] != epoca or cp["receta"] != receta or cp["pacientes_train"] != receta["pacientes_train"]:
            raise ValueError(f"Metadatos de checkpoint incorrectos: época {epoca}")
        if not all(torch.isfinite(t).all() for t in cp["state_dict"].values()):
            raise ValueError("Pesos no finitos")
        pasos = {int(s["step"].item()) for s in cp["optimizer_state_dict"]["state"].values()}
        if pasos != {epoca * pasos_epoca}:
            raise ValueError(f"Pasos de Adam incorrectos: {pasos}")
        for grupo in cp["optimizer_state_dict"]["param_groups"]:
            if grupo["lr"] != receta["lr"] or grupo["weight_decay"] != receta["weight_decay"]:
                raise ValueError("Optimizador diferente a la receta")
        modelo = CNNBreastDCE(receta["dropout"])
        modelo.load_state_dict(cp["state_dict"])
        modelo.eval()
        if sum(p.numel() for p in modelo.parameters()) != receta["parametros"]:
            raise ValueError("Arquitectura incorrecta")
        modelos.append(modelo)
        huella = sha(ruta)
        hashes.add(huella)
        comprobaciones.append(dict(epoca=epoca, sha256=huella, pasos_adam=next(iter(pasos))))
    if len(hashes) != receta["epocas"]:
        raise ValueError("Hay checkpoints repetidos")
    epocas_recalculadas = list(range(1,receta["epocas"]+1)) if args.todas_las_epocas else [receta["epocas"]]
    modelos = [modelos[e-1] for e in epocas_recalculadas]
    args.salida.mkdir(parents=True)
    guardar_json(args.salida / "integridad_checkpoints.json", comprobaciones)
    print(f"Integridad correcta: {len(tr)} cortes, {tr.patient_id.nunique()} pacientes, {pasos_epoca} pasos/época, {len(hashes)} checkpoints distintos. Recalculando épocas: {epocas_recalculadas}", flush=True)
    dl_args = SimpleNamespace(batch_size=receta["batch_size"], workers=0, semilla=receta["semilla"], dispositivo="cpu")
    dl = loader(tr, args.datos, dl_args, False)
    logits_por_modelo = [[] for _ in modelos]
    inicio, ultimo_aviso, n = time.perf_counter(), time.perf_counter(), 0
    with torch.inference_mode():
        for x, _ in dl:
            if x.shape[1:] != (3,256,256) or not torch.isfinite(x).all() or x.min() < 0 or x.max() > 1:
                raise ValueError("Entrada inválida")
            for i, modelo in enumerate(modelos):
                logits_por_modelo[i].append(modelo(x).numpy())
            n += len(x)
            if time.perf_counter() - ultimo_aviso > 20:
                print(f"Recalculando train en CPU: {n}/{len(tr)} cortes ({100*n/len(tr):.1f}%), {time.perf_counter()-inicio:.0f}s", flush=True)
                ultimo_aviso = time.perf_counter()
    predicciones = tr[["sample_id", "patient_id", "pCR"]].copy()
    filas, comparaciones, tablas = [], [], []
    for i, partes in enumerate(logits_por_modelo):
        epoca = epocas_recalculadas[i]
        historial_epoca = h[epoca-1]
        logits = np.concatenate(partes)
        if len(logits) != len(tr) or not np.isfinite(logits).all():
            raise ValueError("Predicciones incompletas o no finitas")
        probs = torch.sigmoid(torch.from_numpy(logits)).numpy()
        predicciones[f"prob_epoca_{epoca:03d}"] = probs
        bce = float(log_loss(tr.pCR, probs, labels=[0,1]))
        tabla = uc.agregar_por_paciente(tr.patient_id, probs).join(tr.groupby("patient_id").pCR.first()).reset_index()
        tabla["pred"] = (tabla.prob >= receta["umbral"]).astype(int)
        tablas.append(tabla)
        precision, recall, _ = precision_recall_curve(tabla.pCR, tabla.prob)
        ap = float(average_precision_score(tabla.pCR, tabla.prob))
        pr_auc = float(auc(recall, precision))
        for umbral in (0.5, receta["umbral"]):
            m = metricas(tabla.pCR, tabla.prob, umbral)
            filas.append(dict(epoca=epoca, bce=bce, pr_auc=pr_auc, average_precision=ap,
                               **{k: m[k] for k in ("umbral", "auc", "accuracy", "sensibilidad", "especificidad", "precision_pcr", "f1_macro", "f1_no_pcr", "f1_pcr", "brier")}))
        m = metricas(tabla.pCR, tabla.prob, historial_epoca["umbral"])
        diferencias = {"bce": abs(bce-historial_epoca["bce"]), "auc": abs(m["auc"]-historial_epoca["auc"]),
                       "average_precision": abs(ap-historial_epoca["average_precision"]), "pr_auc": abs(pr_auc-historial_epoca["pr_auc"]),
                       "f1_macro": abs(m["f1_macro"]-historial_epoca["f1_macro"])}
        c = historial_epoca["matriz_confusion"]
        matriz_historial = [[c["VN"],c["FP"]],[c["FN"],c["VP"]]]
        coincide = max(diferencias.values()) < 1e-4 and m["confusion"] == matriz_historial
        comparaciones.append(dict(epoca=epoca, coincide=coincide, diferencias_absolutas=diferencias))
    predicciones.to_csv(args.salida / "predicciones_train_auditadas.csv", index=False)
    pd.DataFrame(filas).to_csv(args.salida / "metricas_train_recalculadas.csv", index=False)
    tablas[-1].to_csv(args.salida / "predicciones_pacientes_train_final.csv", index=False)
    correcto = all(c["coincide"] for c in comparaciones)
    guardar_json(args.salida / "auditoria.json", dict(correcto=correcto, pacientes=tr.patient_id.nunique(), cortes=len(tr),
                 pasos_por_epoca=pasos_epoca, sha256_checkpoint_final=artefacto["sha256_checkpoint"],
                 epocas_con_inferencia_completa=epocas_recalculadas,
                 comparacion_cpu_gpu=comparaciones, tolerancia=0.0001, segundos_auditoria=time.perf_counter()-inicio,
                 aviso="Solo entrenamiento; no estima generalización ni selecciona época o umbral. Test no se ha usado."))
    fig, ax = plt.subplots(2,2,figsize=(12,8))
    epocas = list(range(1,len(h)+1))
    bces = [f["bce"] for f in h]
    online = [f["loss_online"] for f in h]
    p_clase = float(tr.pCR.mean())
    entropia = -p_clase*np.log(p_clase)-(1-p_clase)*np.log(1-p_clase)
    for eje in (ax[0,0],ax[0,1]):
        eje.plot(epocas, bces, "o-", label="BCE train, eval sin aumentos")
        eje.plot(epocas, online, "o-", label="Pérdida online, con aumentos")
        eje.axhline(entropia, color="gray", linestyle=":", label="Constante: prevalencia por corte")
        eje.set(xlabel="Época", ylabel="BCE por corte", xticks=epocas)
        eje.legend(fontsize=8)
    ax[0,0].set(ylim=(0,max(0.8,max(online)*1.1)), title="Escala con contexto")
    ax[0,1].set_title("Detalle de las mismas pérdidas (eje ampliado)")
    referencia = pd.DataFrame([f for f in filas if f["umbral"] == 0.5])
    ax[1,0].plot(epocas, [f["auc"] for f in h], "o-", label="ROC-AUC train, historial GPU")
    ax[1,0].plot(epocas, [f["average_precision"] for f in h], "o-", label="AP train, historial GPU")
    ax[1,0].scatter(referencia.epoca, referencia.auc, marker="x", s=70, label="ROC-AUC recalculada CPU")
    ax[1,0].axhline(0.5, color="gray", linestyle=":", label="Referencia ROC-AUC")
    ax[1,0].set(xlabel="Época", ylabel="Métrica por paciente", ylim=(0,1), xticks=epocas, title="Métricas de entrenamiento, no validación")
    ax[1,0].legend(fontsize=8)
    for t in (0.5,receta["umbral"]):
        f = pd.DataFrame([fila for fila in filas if fila["umbral"] == t])
        ax[1,1].plot(f.epoca,f.f1_macro,"o-",label=f"F1 macro train recalculado, t={t:.2f}")
    ax[1,1].set(xlabel="Época", ylabel="F1 macro", ylim=(0,1), xticks=epocas, title="Umbral de historial y umbral final congelado")
    ax[1,1].legend(fontsize=8)
    fig.suptitle("Historial GPU y recálculo CPU marcado explícitamente · solo train, sin utilizar test")
    guardar_figura(fig,args.salida/"curvas_train_auditadas.png")
    if not correcto:
        raise ValueError("Hay diferencias superiores a tolerancia; revisar auditoria.json antes de test")
    print(f"Las épocas recalculadas {epocas_recalculadas} coinciden con GPU. AUC train final={referencia.iloc[-1].auc:.6f}. Auditoría en {args.salida}",flush=True)


if __name__ == "__main__":
    main()
