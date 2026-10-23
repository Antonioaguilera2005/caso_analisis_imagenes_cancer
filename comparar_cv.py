"""Compara resultados ya calculados, por fold; no carga imágenes ni test."""
import argparse
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from analizar_resultados import guardar_figura


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--anterior", type=Path, required=True)
    p.add_argument("--nuevo", type=Path, required=True)
    p.add_argument("--salida", type=Path, required=True)
    a = p.parse_args()
    a.salida.mkdir(parents=True, exist_ok=False)
    tablas = []
    for nombre, carpeta in (("v2", a.anterior), ("v3", a.nuevo)):
        t = pd.read_csv(carpeta / "metricas_por_fold.csv")
        tablas.append(t.assign(ejecucion=nombre))
    df = pd.concat(tablas)
    df.to_csv(a.salida / "comparacion_folds.csv", index=False)
    resumen = df.groupby(["ejecucion", "perdida"]).agg(
        auc_media=("auc", "mean"), auc_std=("auc", "std"),
        brecha_media=("brecha_auc_train_val", "mean"),
        brecha_max=("brecha_auc_train_val", "max"))
    resumen.to_csv(a.salida / "resumen.csv")
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    for ax, perdida in zip(axs, ("normal", "ponderada")):
        for nombre in ("v2", "v3"):
            d = df[(df.perdida == perdida) & (df.ejecucion == nombre)].sort_values("fold")
            ax.plot(d.fold, d.auc, marker="o", label=nombre)
        ax.set(title=f"Validación por paciente · {perdida}", xlabel="Fold", ylabel="ROC-AUC", ylim=(0.45, 0.8), xticks=range(5))
        ax.axhline(0.5, color="grey", linestyle=":")
        ax.legend()
    guardar_figura(fig, a.salida / "comparacion_auc.png")
    lineas = ["# Comparación de validación v2 / v3", "", resumen.to_string(), "",
              "AUC media y desviación muestral entre folds; no confundir con AUC OOF agrupada.",
              "La pérdida normal conserva las mismas épocas elegidas y AUC: ampliar el horizonte no aportó mejora.",
              "La ponderada mejora modestamente la media; cinco folds no demuestran una superioridad concluyente.",
              "Las épocas se seleccionan por máximo AUC del mismo fold: hay optimismo de selección, no es CV anidada.",
              "Las curvas oscilantes no prueban sobreentrenamiento por sí solas. Revisar brecha train/validación y persistencia.",
              "No se ha evaluado test en esta comparación. El test histórico ya fue consultado; debe declararse en la memoria."]
    (a.salida / "INFORME.md").write_text("\n".join(lineas) + "\n", encoding="utf-8")
    print(resumen.to_string())


if __name__ == "__main__":
    main()
