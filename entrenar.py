"""Entrena en train; selecciona por AUC de validación por paciente; nunca evalúa test."""
import argparse
import hashlib
from importlib.metadata import version
import json
import math
import platform
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import auc, average_precision_score, precision_recall_curve
from torch import nn
from torch.utils.data import DataLoader

from modelo import CNNBreastDCE, aumento_geometrico

# Utilidades oficiales originales, descargadas por descargar_datos.py.
sys.path.insert(0, str(Path(__file__).resolve().parent / "breastdcedl"))
import utils_caso as uc


def json_seguro(valor):
    if isinstance(valor, dict):
        return {k: json_seguro(v) for k, v in valor.items()}
    if isinstance(valor, (list, tuple)):
        return [json_seguro(v) for v in valor]
    if isinstance(valor, float) and not math.isfinite(valor):
        return None
    return valor


def guardar_json(ruta, contenido):
    ruta.write_text(json.dumps(json_seguro(contenido), indent=2, ensure_ascii=False,
                               allow_nan=False) + "\n", encoding="utf-8")


def semilla_worker(_):
    semilla = torch.initial_seed() % (2 ** 32)
    np.random.seed(semilla)
    random.seed(semilla)


def fijar_semilla(semilla):
    random.seed(semilla)
    np.random.seed(semilla)
    torch.manual_seed(semilla)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(semilla)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # No promete igualdad exacta entre plataformas o versiones.
    torch.use_deterministic_algorithms(True, warn_only=True)


def sincronizar(dispositivo):
    if dispositivo.type == "cuda":
        torch.cuda.synchronize(dispositivo)


def loader(filas, raiz, args, entrenamiento):
    dataset = uc.BreastDCEDataset(filas, raiz,
                                transform=aumento_geometrico if entrenamiento else None)
    generador = torch.Generator().manual_seed(args.semilla)
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=entrenamiento,
                      num_workers=args.workers, pin_memory=args.dispositivo == "cuda",
                      worker_init_fn=semilla_worker, generator=generador)


@torch.inference_mode()
def predecir(modelo, dl, dispositivo, devolver_logits=False):
    modelo.eval()
    probabilidades = []
    for x, _ in dl:
        logits = modelo(x.to(dispositivo))
        prob = (logits if devolver_logits else torch.sigmoid(logits)).cpu().numpy()
        if not np.isfinite(prob).all():
            raise ValueError("Predicciones no finitas")
        probabilidades.append(prob)
    return np.concatenate(probabilidades)


def evaluar_epoca(modelo, dl, filas, dispositivo, peso):
    """Evaluación sin aumentos/dropout: BCE por corte y AUC por paciente."""
    logits = torch.from_numpy(predecir(modelo, dl, dispositivo, devolver_logits=True))
    etiquetas = torch.tensor(filas.pCR.to_numpy(), dtype=torch.float32)
    bce = nn.functional.binary_cross_entropy_with_logits(logits, etiquetas).item()
    bce_ponderada = nn.functional.binary_cross_entropy_with_logits(
        logits, etiquetas, pos_weight=torch.tensor(peso)).item()
    probs = torch.sigmoid(logits).numpy()
    metricas = uc.evaluar_por_paciente(probs, filas, metodo="mean", umbral=0.5)
    pacientes = uc.agregar_por_paciente(filas.patient_id, probs, metodo="mean")
    pacientes = pacientes.join(filas.groupby("patient_id").pCR.first())
    precision, recall, _ = precision_recall_curve(pacientes.pCR, pacientes.prob)
    metricas["pr_auc"] = float(auc(recall, precision))
    metricas["average_precision"] = float(average_precision_score(pacientes.pCR, pacientes.prob))
    c = metricas["matriz_confusion"]
    f1_pos = 2 * c["VP"] / (2 * c["VP"] + c["FP"] + c["FN"]) if 2 * c["VP"] + c["FP"] + c["FN"] else 0.0
    f1_neg = 2 * c["VN"] / (2 * c["VN"] + c["FP"] + c["FN"]) if 2 * c["VN"] + c["FP"] + c["FN"] else 0.0
    metricas["f1_macro"] = (f1_pos + f1_neg) / 2
    return dict(bce=bce, bce_objetivo=bce_ponderada, **metricas)


def limitar_para_prueba(filas, negativos=2):
    """Prueba técnica por paciente; entrenamiento mantiene desbalance para probar pos_weight."""
    pacientes = filas.groupby("patient_id", sort=True).pCR.first()
    elegidos = []
    for clase in (0, 1):
        cantidad = negativos if clase == 0 else 2
        ids = pacientes[pacientes == clase].index[:cantidad]
        if len(ids) < cantidad:
            raise ValueError("La prueba necesita al menos dos pacientes por clase")
        elegidos.extend(ids)
    return filas[filas.patient_id.isin(elegidos)].reset_index(drop=True)


def validar_particion(tr, va):
    if tr.empty or va.empty:
        raise ValueError("Partición vacía")
    if set(tr.patient_id) & set(va.patient_id):
        raise ValueError("Fuga por paciente")
    for filas in (tr, va):
        if set(filas.split) != {"train"}:
            raise ValueError("Solo se permite train para el desarrollo")
        if set(filas.pCR) != {0, 1}:
            raise ValueError("La partición debe contener ambas clases")
        if (filas.groupby("patient_id").pCR.nunique() != 1).any():
            raise ValueError("Etiquetas inconsistentes dentro de paciente")


def ejecutar(args, perdida, tr, va, directorio):
    fijar_semilla(args.semilla)
    dispositivo = torch.device(args.dispositivo)
    directorio.mkdir(parents=True, exist_ok=False)
    modelo = CNNBreastDCE(args.dropout).to(dispositivo)
    optimizador = torch.optim.Adam(modelo.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    peso = uc.pos_weight(tr) if perdida == "ponderada" else 1.0
    criterio = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(peso, device=dispositivo))
    dl_tr, dl_va = loader(tr, args.datos, args, True), loader(va, args.datos, args, False)
    dl_tr_eval = loader(tr, args.datos, args, False)
    checkpoints = directorio / "checkpoints"
    checkpoints.mkdir()
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(perdida=perdida, pos_weight=peso, canales=["PRE", "EARLY", "LATE"],
                  normalizacion="float32 / 255", parametros=sum(p.numel() for p in modelo.parameters()),
                  seleccion="AUC por paciente, agregación mean", umbral_referencia=0.5,
                  pacientes_train=tr.patient_id.nunique(), pacientes_val=va.patient_id.nunique(),
                  uso="prueba_tecnica" if args.prueba else "validacion_interna")
    config["sha256_samples"] = hashlib.sha256((args.datos / "metadata/samples.csv").read_bytes()).hexdigest()
    config["sha256_utils"] = hashlib.sha256(Path(uc.__file__).read_bytes()).hexdigest()
    config["sha256_modelo"] = hashlib.sha256((Path(__file__).resolve().parent / "modelo.py").read_bytes()).hexdigest()
    config["sha256_entrenar"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    guardar_json(directorio / "config.json", config)
    entorno = dict(python=sys.version, plataforma=platform.platform(), torch=torch.__version__,
                   numpy=np.__version__, pandas=pd.__version__, hip=torch.version.hip,
                   cuda=torch.version.cuda, dispositivo=str(dispositivo))
    entorno["paquetes"] = {nombre: version(nombre) for nombre in ("torch", "numpy", "pandas", "pillow", "scikit-learn")}
    if dispositivo.type == "cuda":
        entorno["gpu"] = torch.cuda.get_device_name(dispositivo)
        torch.cuda.reset_peak_memory_stats(dispositivo)
    guardar_json(directorio / "entorno.json", entorno)
    lineas_log = []

    def informar(mensaje):
        print(mensaje, flush=True)
        lineas_log.append(mensaje)
        (directorio / "entrenamiento.log").write_text("\n".join(lineas_log) + "\n", encoding="utf-8")

    informar(f"{perdida} | fold={args.fold} | dispositivo={dispositivo} "
             f"| gpu={entorno.get('gpu', 'no utilizada')} | parametros={config['parametros']} "
             f"| pacientes train/val={config['pacientes_train']}/{config['pacientes_val']}")
    informar(f"entrada=solo imagenes PRE/EARLY/LATE | perdida={perdida} | pos_weight={peso:.4f} "
             f"| batch={args.batch_size} | lr={args.lr:g} | semilla={args.semilla} "
             f"| agregacion=mean | umbral_referencia=0.5")
    if args.prueba:
        informar("PRUEBA TECNICA: estas metricas no estiman el rendimiento del proyecto.")
    tr[["sample_id", "patient_id", "pCR", "fold"]].to_csv(directorio / "particion_train.csv", index=False)
    va[["sample_id", "patient_id", "pCR", "fold"]].to_csv(directorio / "particion_val.csv", index=False)
    mejor, sin_mejora, historial = -math.inf, 0, []
    checkpoint_elegido = None
    sincronizar(dispositivo)
    inicio_total = time.perf_counter()
    for epoca in range(1, args.epocas + 1):
        sincronizar(dispositivo)
        inicio = time.perf_counter()
        modelo.train()
        acumulada, ejemplos = 0.0, 0
        for x, y in dl_tr:
            x, y = x.to(dispositivo), y.to(dispositivo)
            optimizador.zero_grad(set_to_none=True)
            loss = criterio(modelo(x), y)
            if not torch.isfinite(loss):
                raise ValueError("Pérdida no finita")
            loss.backward()
            optimizador.step()
            acumulada += loss.item() * len(y)
            ejemplos += len(y)
        evaluacion_train = evaluar_epoca(modelo, dl_tr_eval, tr, dispositivo, peso)
        evaluacion_val = evaluar_epoca(modelo, dl_va, va, dispositivo, peso)
        metricas = {k: v for k, v in evaluacion_val.items() if k not in ("bce", "bce_objetivo")}
        if not math.isfinite(metricas["auc"]):
            raise ValueError("AUC de validación no válida")
        sincronizar(dispositivo)
        registro = dict(epoca=epoca, loss_train=acumulada / ejemplos,
                        loss_train_eval=evaluacion_train["bce"], loss_val=evaluacion_val["bce"],
                        loss_train_eval_objetivo=evaluacion_train["bce_objetivo"],
                        loss_val_objetivo=evaluacion_val["bce_objetivo"],
                        auc_train=evaluacion_train["auc"], auc_val=metricas["auc"],
                        pr_auc_train=evaluacion_train["pr_auc"], pr_auc_val=metricas["pr_auc"],
                        average_precision_train=evaluacion_train["average_precision"],
                        average_precision_val=metricas["average_precision"],
                        lr=float(optimizador.param_groups[0]["lr"]),
                        segundos=time.perf_counter() - inicio, **metricas)
        historial.append(registro)
        guardar_json(directorio / "historial.json", historial)
        informar(f"epoch={epoca:03d} train_loss={registro['loss_train']:.4f} "
                 f"train_eval_loss={registro['loss_train_eval_objetivo']:.4f} "
                 f"val_loss={registro['loss_val_objetivo']:.4f} "
                 f"train_roc_auc={registro['auc_train']:.4f} "
                 f"patient_roc_auc={metricas['auc']:.4f} patient_pr_auc={metricas['pr_auc']:.4f} "
                 f"AP={metricas['average_precision']:.4f} "
                 f"recall_pcr={metricas['sensibilidad']:.4f} especificidad={metricas['especificidad']:.4f} "
                 f"f1_macro={metricas['f1_macro']:.4f} lr={registro['lr']:g} time={registro['segundos']:.1f}s")
        ruta_checkpoint = checkpoints / f"epoca_{epoca:03d}.pt"
        temporal = ruta_checkpoint.with_suffix(".pt.tmp")
        torch.save({"state_dict": modelo.state_dict(), "optimizer_state_dict": optimizador.state_dict(),
                    "config": config, "epoca": epoca, "auc_val": metricas["auc"],
                    "metricas_train": evaluacion_train, "metricas_val": evaluacion_val}, temporal)
        temporal.replace(ruta_checkpoint)
        if metricas["auc"] > mejor:
            mejor, sin_mejora = metricas["auc"], 0
            checkpoint_elegido = ruta_checkpoint
            guardar_json(directorio / "seleccion.json", {
                "checkpoint": str(ruta_checkpoint.relative_to(directorio)).replace("\\", "/"),
                "epoca": epoca, "auc_val": mejor,
                "criterio": "Máxima AUC por paciente con media de cortes; primer máximo en empates",
            })
            informar(f"Seleccion actual: epoca={epoca:03d} val_auc={mejor:.4f} checkpoint={ruta_checkpoint.name}")
        else:
            sin_mejora += 1
        if sin_mejora >= args.paciencia:
            informar(f"Early stopping: {sin_mejora} epocas sin mejora de ROC-AUC por paciente.")
            break
    duracion = time.perf_counter() - inicio_total
    checkpoint = torch.load(checkpoint_elegido, map_location=dispositivo, weights_only=True)
    modelo.load_state_dict(checkpoint["state_dict"])
    probs = predecir(modelo, dl_va, dispositivo)
    predicciones = va[["sample_id", "patient_id", "pCR"]].copy()
    predicciones["prob"] = probs
    predicciones.to_csv(directorio / "predicciones_cortes_val.csv", index=False)
    resultados = {}
    for metodo in ("mean", "median", "max"):
        resultados[metodo] = uc.evaluar_por_paciente(probs, va, metodo=metodo, umbral=0.5)
        tabla = uc.agregar_por_paciente(va.patient_id, probs, metodo)
        tabla = tabla.join(va.groupby("patient_id").pCR.first())
        tabla["pred"] = (tabla.prob >= 0.5).astype(int)
        tabla.to_csv(directorio / f"predicciones_pacientes_{metodo}.csv")
    resultados["baseline_siempre_cero"] = uc.evaluar_por_paciente(np.zeros(len(va)), va)
    guardar_json(directorio / "metricas_val.json", resultados)
    guardar_json(directorio / "ejecucion.json", {
        "epoca_elegida": checkpoint["epoca"], "segundos_entrenamiento_y_validacion": duracion,
        "memoria_gpu_max_bytes": torch.cuda.max_memory_allocated(dispositivo) if dispositivo.type == "cuda" else None,
        "checkpoint_elegido": str(checkpoint_elegido.relative_to(directorio)).replace("\\", "/"),
        "sha256_checkpoint": hashlib.sha256(checkpoint_elegido.read_bytes()).hexdigest(),
        "advertencia": "Prueba técnica sin valor predictivo" if args.prueba else "Resultados de validación interna; test no evaluado",
    })
    informar(f"Finalizado {perdida}: epoca elegida={checkpoint['epoca']} AUC={checkpoint['auc_val']:.4f} "
             f"tiempo_total={duracion:.1f}s | resultados={directorio}")
    return dict(perdida=perdida, epoca=checkpoint["epoca"], **resultados["mean"])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--datos", type=Path, default=Path("breastdcedl"))
    p.add_argument("--salida", type=Path, required=True, help="Carpeta nueva: nunca sobrescribe experimentos")
    p.add_argument("--fold", type=int, choices=range(5), default=0)
    p.add_argument("--perdida", choices=("normal", "ponderada", "ambas"), default="ambas")
    p.add_argument("--dispositivo", choices=("cpu", "cuda"), default="cuda")
    p.add_argument("--epocas", type=int, default=40)
    p.add_argument("--paciencia", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--semilla", type=int, default=42)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--weight-decay", type=float, default=0.0001)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--prueba", action="store_true", help="Cinco pacientes train y cuatro validación; solo prueba técnica")
    args = p.parse_args()
    if min(args.epocas, args.paciencia, args.batch_size) < 1 or args.workers < 0 or args.lr <= 0 or args.weight_decay < 0 or not 0 <= args.dropout < 1 or not all(math.isfinite(v) for v in (args.lr, args.weight_decay, args.dropout)):
        p.error("Parámetros numéricos fuera de rango")
    if args.dispositivo == "cuda" and not torch.cuda.is_available():
        p.error("PyTorch no detecta GPU. Ejecuta scripts/diagnosticar_entorno.py o usa --dispositivo cpu para pruebas.")
    if args.salida.exists():
        p.error("La carpeta de salida ya existe. Elige otra para conservar los experimentos.")
    samples = uc.cargar_samples(args.datos)
    tr, va = uc.particion(samples, args.fold)
    if args.prueba:
        tr, va = limitar_para_prueba(tr, negativos=3), limitar_para_prueba(va)
    validar_particion(tr, va)
    args.salida.mkdir(parents=True, exist_ok=False)
    perdidas = ("normal", "ponderada") if args.perdida == "ambas" else (args.perdida,)
    comparacion = [ejecutar(args, perdida, tr, va, args.salida / perdida) for perdida in perdidas]
    guardar_json(args.salida / "comparacion.json", comparacion)


if __name__ == "__main__":
    main()
