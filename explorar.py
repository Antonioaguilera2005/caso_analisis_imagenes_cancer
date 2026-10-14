"""Explora seis pacientes de train (una por cohorte/clase), sin consultar imágenes de test."""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent / "breastdcedl"))
import utils_caso as uc


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--datos", type=Path, default=Path("breastdcedl"))
    p.add_argument("--salida", type=Path, default=Path("reports/exploracion"))
    args = p.parse_args()
    args.salida.mkdir(parents=True, exist_ok=True)
    pacientes = uc.cargar_patients(args.datos)
    train = pacientes[pacientes.split == "train"].sort_values("pid")
    seleccion = train.groupby(["dataset", "pCR"], sort=True).head(1)
    samples = uc.cargar_samples(args.datos)
    samples = samples[samples.split == "train"]
    fig, ejes = plt.subplots(len(seleccion), 4, figsize=(13, 3 * len(seleccion)), squeeze=False)
    resumen = []
    for ejes_fila, paciente in zip(ejes, seleccion.itertuples()):
        filas = samples[samples.patient_id == paciente.pid].sort_values("slice_index")
        fases = []
        for fila in filas.itertuples():
            for columna in ("path_pre", "path_early", "path_late"):
                with Image.open(args.datos / getattr(fila, columna)) as imagen:
                    if imagen.format != "PNG" or imagen.size != (256, 256) or imagen.mode != "L":
                        raise ValueError(f"Formato inesperado: {getattr(fila, columna)}")
            x = uc.cargar_imagen(fila, args.datos)
            if x.shape != (3, 256, 256) or not np.isfinite(x).all() or x.min() < 0 or x.max() > 1:
                raise ValueError(f"Tensor inválido: {fila.sample_id}")
            tejido = x[0] > 0.1
            if not tejido.any():
                raise ValueError(f"Máscara de tejido vacía: {fila.sample_id}")
            fases.append([float(c[tejido].mean()) for c in x])
        medias = np.mean(fases, axis=0)
        centro = filas.iloc[len(filas) // 2]
        x = uc.cargar_imagen(centro, args.datos)
        for eje, fase, nombre in zip(ejes_fila[:3], x, uc.FASES):
            eje.imshow(fase, cmap="gray", vmin=0, vmax=1)
            eje.set_title(nombre)
            eje.axis("off")
        realce = x[1] - x[0]
        limite = max(float(np.abs(realce).max()), 1 / 255)
        ejes_fila[3].imshow(realce, cmap="coolwarm", vmin=-limite, vmax=limite)
        ejes_fila[3].set_title("EARLY − PRE (escala por corte)")
        ejes_fila[3].axis("off")
        ejes_fila[0].set_title(f"{paciente.pid} | {paciente.dataset} | pCR={paciente.pCR}\nPRE")
        resumen.append(dict(pid=paciente.pid, cohorte=paciente.dataset, pCR=int(paciente.pCR),
                            cortes=len(filas), muestra_visualizada=centro.sample_id,
                            medias_tejido_PRE_EARLY_LATE=medias.tolist(),
                            realce_medio=float(medias[1] - medias[0]),
                            late_menor_early=bool(medias[2] < medias[1])))
    fig.suptitle("Ejemplos de entrenamiento; no prueban diferencias predictivas entre clases")
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(args.salida / "fases_y_realce.png", dpi=140)
    plt.close(fig)
    (args.salida / "resumen.json").write_text(json.dumps(resumen, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    train.groupby(["dataset", "pCR"]).size().unstack(fill_value=0).to_csv(args.salida / "clases_por_cohorte_train.csv")
    print(f"Explorados {len(resumen)} pacientes de train. Figuras y resumen: {args.salida}")


if __name__ == "__main__":
    main()
