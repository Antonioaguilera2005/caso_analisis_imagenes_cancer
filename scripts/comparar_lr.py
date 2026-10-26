"""Comparación descriptiva por fold; leer análisis ya validados, sin inferencia/test."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from analizar_resultados import plt, guardar_figura


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--referencia", type=Path, default=Path("runs/cv5_v3_60ep/analisis"))
    p.add_argument("--nuevo", type=Path, default=Path("runs/cv5_simple_lr3e4/analisis"))
    p.add_argument("--salida", type=Path, default=Path("runs/comparacion_lr_baja"))
    a = p.parse_args()
    tablas = []
    for nombre, carpeta in [("lr=0,001", a.referencia), ("lr=0,0003", a.nuevo)]:
        t = pd.read_csv(carpeta / "metricas_por_fold.csv")
        if len(t) != 10 or t.duplicated(["perdida", "fold"]).any() or any(set(t[t.perdida == l].fold) != set(range(5)) for l in ["normal", "ponderada"]):
            p.error("Se requieren cinco folds completos y únicos para ambas pérdidas")
        tablas.append(t.assign(receta=nombre))
    a.salida.mkdir(parents=True, exist_ok=False)
    df = pd.concat(tablas, ignore_index=True)
    df.to_csv(a.salida / "comparacion_folds.csv", index=False)
    resumen = df.groupby(["receta", "perdida"])[["auc", "brecha_auc_train_val", "f1_macro"]].agg(["mean", "std"])
    resumen.to_csv(a.salida / "resumen.csv")
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    for ax, perdida in zip(axs, ["normal", "ponderada"]):
        for nombre in df.receta.unique():
            t = df[(df.receta == nombre) & (df.perdida == perdida)].sort_values("fold")
            ax.plot(t.fold, t.auc, "o-", label=nombre)
        ax.set(title=perdida, xlabel="Fold", ylabel="ROC-AUC validación por paciente", ylim=(0, 1), xticks=range(5))
        ax.axhline(.5, color="gray", linestyle=":")
        ax.legend()
    guardar_figura(fig, a.salida / "comparacion_auc.png")
    print(resumen.to_string())
    print("Comparación descriptiva: no demuestra significación ni evalúa test. Revisar curvas y cada fold.")


if __name__ == "__main__":
    main()
