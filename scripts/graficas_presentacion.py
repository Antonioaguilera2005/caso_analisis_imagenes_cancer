"""PNG de historiales reales: escala compartida, detalle y selección; no hace inferencia."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from analizar_resultados import plt, guardar_figura

COLORES = ("#2463a3", "#cc652b")


def leer(ruta):
    return json.loads(ruta.read_text(encoding="utf-8-sig"))


def limites(valores, cero=False):
    valores = np.asarray(valores, dtype=float)
    if not np.isfinite(valores).all():
        raise ValueError("Historial con valores no finitos")
    inferior, superior = float(valores.min()), float(valores.max())
    margen = max((superior - inferior) * .08, .01)
    return (0 if cero else max(0, inferior - margen), superior + margen)


def marcar(ax, h, epoca, xmax):
    ax.axvline(epoca, color="#374151", linestyle="--", linewidth=1)
    if epoca < h.epoca.max():
        ax.axvspan(epoca + .5, h.epoca.max(), color="#64748b", alpha=.09)
    ax.set_xlim(.5, xmax + .5)
    ax.set_xlabel("Época")
    ax.grid(alpha=.18)
    ax.spines[["top", "right"]].set_visible(False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", type=Path, nargs="+", required=True)
    p.add_argument("--salida", type=Path, required=True)
    a = p.parse_args()
    registros = []
    for raiz in a.runs:
        for d in sorted(raiz.glob("fold*_semilla*/normal")) + sorted(raiz.glob("fold*_semilla*/ponderada")):
            c, seleccion = leer(d/"config.json"), leer(d/"seleccion.json")
            if c.get("prueba"):
                raise ValueError("No mezclar pruebas técnicas con resultados")
            h = pd.DataFrame(leer(d/"historial.json"))
            requeridas = {"epoca", "loss_train_eval", "loss_val", "auc_train", "auc_val"}
            if not requeridas.issubset(h.columns) or h.epoca.duplicated().any():
                raise ValueError(f"Historial incompleto: {d}")
            e = int(seleccion["epoca"])
            if e not in set(h.epoca):
                raise ValueError("Selección ausente del historial")
            registros.append((raiz.name, c, h, e))
    if not registros or len({r.name for r in a.runs}) != len(a.runs):
        p.error("Sin ejecuciones o nombres de raíz duplicados")
    # La escala BCE es idéntica entre todos los experimentos y pérdidas.
    bce = np.concatenate([h[["loss_train_eval", "loss_val"]].to_numpy().ravel() for _, _, h, _ in registros])
    amplio = limites(bce, cero=True)
    a.salida.mkdir(parents=True, exist_ok=False)
    resumen = []
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "figure.facecolor": "white"})
    for raiz in a.runs:
        destino = a.salida/raiz.name
        destino.mkdir()
        for perdida in ("normal", "ponderada"):
            rs = sorted([r for r in registros if r[0] == raiz.name and r[1]["perdida"] == perdida], key=lambda r: r[1]["fold"])
            if len(rs) != 5 or {r[1]["fold"] for r in rs} != set(range(5)):
                raise ValueError("Se requieren cinco folds por pérdida")
            xmax = max(h.epoca.max() for _, _, h, _ in rs)
            detalle = limites(np.concatenate([h[["loss_train_eval", "loss_val"]].to_numpy().ravel() for _, _, h, _ in rs]))
            fig, axs = plt.subplots(5, 2, figsize=(13, 17))
            for fila, (_, c, h, e) in zip(axs, rs):
                elegida = h.loc[h.epoca == e].iloc[0]
                resumen.append(dict(experimento=raiz.name, perdida=perdida, fold=c["fold"], epoca=e,
                                    auc_train=elegida.auc_train, auc_val=elegida.auc_val,
                                    brecha=elegida.auc_train-elegida.auc_val))
                for ax, rango, titulo in zip(fila, (amplio, detalle), ("Contexto · eje desde cero", "Detalle · eje ampliado")):
                    for clave, color, etiqueta in zip(("loss_train_eval", "loss_val"), COLORES, ("Train sin aumentos", "Validación")):
                        ax.plot(h.epoca, h[clave], color=color, linewidth=1.6, label=etiqueta)
                        ax.scatter([e], [elegida[clave]], color=color, s=26, zorder=4)
                    marcar(ax, h, e, xmax)
                    ax.set(ylim=rango, ylabel="BCE sin ponderar / corte", title=f"Fold {c['fold']} · {titulo} · selección: {e}")
                    if c["fold"] == 0:
                        ax.legend(loc="best", fontsize=9)
            fig.suptitle(f"{raiz.name} · pérdida {perdida}\nContexto: escala global · detalle: escala común entre estos cinco folds · sin suavizado", fontsize=14)
            fig.text(.5, .009, "Línea discontinua: selección por AUC de validación por paciente. Zona gris: épocas posteriores descartadas; siguen visibles.", ha="center", fontsize=9)
            fig.tight_layout(rect=(0,.03,1,.95))
            fig.savefig(destino/f"perdidas_{perdida}.png", dpi=150)
            plt.close(fig)
            fig, axs = plt.subplots(5, 1, figsize=(11, 15))
            for ax, (_, c, h, e) in zip(axs, rs):
                for clave, color, etiqueta in zip(("auc_train", "auc_val"), COLORES, ("Train sin aumentos", "Validación")):
                    ax.plot(h.epoca, h[clave], color=color, label=etiqueta, linewidth=1.5)
                marcar(ax, h, e, xmax)
                ax.axhline(.5, color="gray", linestyle=":", linewidth=1)
                elegida=h.loc[h.epoca==e].iloc[0]
                ax.set(ylim=(0,1), ylabel="ROC-AUC / paciente", title=f"Fold {c['fold']} · época {e} · train {elegida.auc_train:.3f} / validación {elegida.auc_val:.3f}")
                if c["fold"] == 0:
                    ax.legend(fontsize=9)
            fig.suptitle(f"{raiz.name} · {perdida} · AUC real por paciente", fontsize=15)
            fig.tight_layout(rect=(0,0,1,.96))
            fig.savefig(destino/f"auc_{perdida}.png", dpi=150)
            plt.close(fig)
            if perdida == "ponderada":
                claves = ["loss_train_eval_objetivo", "loss_val_objetivo"]
                rango = limites(np.concatenate([h[claves].to_numpy().ravel() for _, _, h, _ in rs]), cero=True)
                fig, axs = plt.subplots(5, 1, figsize=(11, 15))
                for ax, (_, c, h, e) in zip(axs, rs):
                    for clave, color, etiqueta in zip(claves, COLORES, ("Train sin aumentos", "Validación")):
                        ax.plot(h.epoca, h[clave], color=color, label=etiqueta)
                    marcar(ax, h, e, xmax)
                    ax.set(ylim=rango, ylabel="BCE ponderada / corte", title=f"Fold {c['fold']} · selección: {e}")
                    if c["fold"] == 0:
                        ax.legend(fontsize=9)
                fig.suptitle(f"{raiz.name} · objetivo ponderado evaluado sin aumentos\nSeparado de la BCE sin ponderar; pesos calculados en train de cada fold", fontsize=14)
                fig.tight_layout(rect=(0,0,1,.95))
                fig.savefig(destino/"objetivo_ponderado.png", dpi=150)
                plt.close(fig)
    pd.DataFrame(resumen).to_csv(a.salida/"epocas_y_brechas.csv", index=False)
    (a.salida/"LEEME.md").write_text(
        "# Gráficas de presentación\n\nDatos originales sin modificar ni suavizar. "
        "Los dos paneles de pérdida representan exactamente los mismos datos: contexto con escala global compartida "
        "entre experimentos y detalle con escala compartida entre los cinco folds de cada figura (no entre experimentos). "
        "No se recortan valores ni épocas. La zona gris identifica épocas posteriores al checkpoint seleccionado, "
        "no una predicción de test. La selección maximiza AUC de validación por paciente, no minimiza BCE por corte. "
        "Los archivos perdidas muestran BCE sin ponderar para poder comparar. El objetivo ponderado evaluado "
        "sin aumentos se muestra aparte en objetivo_ponderado.png; no es la pérdida online con aumentos. "
        "Las gráficas originales se conservan. No se han recalculado predicciones ni consultado test. "
        "El reentrenamiento final no tiene validación y no debe presentarse como si la tuviera.\n", encoding="utf-8")
    print(f"PNG y tabla guardados en {a.salida}")


if __name__ == "__main__":
    main()
