"""Ejecuta cinco folds y ambas pérdidas secuencialmente, sin tocar test."""
import argparse
import subprocess
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--datos", type=Path, default=Path("breastdcedl"))
    p.add_argument("--clinica", action="store_true", help="Añadir edad y menopausia; ajuste solo en train de cada fold")
    p.add_argument("--salida", type=Path, required=True)
    p.add_argument("--dispositivo", choices=("cpu", "cuda"), default="cuda")
    p.add_argument("--folds", type=int, nargs="+", choices=range(5), default=list(range(5)))
    p.add_argument("--epocas", type=int, default=40)
    p.add_argument("--paciencia", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--semilla", type=int, default=42)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--weight-decay", type=float, default=0.0001)
    p.add_argument("--prueba", action="store_true", help="Prueba técnica pequeña; no analiza como CV científica")
    args = p.parse_args()
    if len(set(args.folds)) != len(args.folds):
        p.error("No repetir folds")
    if args.salida.exists():
        p.error("El destino ya existe; usa otro para conservar los resultados")
    repo = Path(__file__).resolve().parent
    args.salida.mkdir(parents=True)
    carpetas = []
    for fold in args.folds:
        salida = args.salida / f"fold{fold}_semilla{args.semilla}"
        carpetas.append(str(salida))
        comando = [sys.executable, str(repo / "entrenar.py"), "--datos", str(args.datos),
                   "--salida", str(salida), "--fold", str(fold), "--perdida", "ambas",
                   "--dispositivo", args.dispositivo, "--epocas", str(args.epocas),
                   "--paciencia", str(args.paciencia), "--batch-size", str(args.batch_size),
                   "--workers", str(args.workers), "--semilla", str(args.semilla),
                   "--lr", str(args.lr), "--dropout", str(args.dropout), "--weight-decay", str(args.weight_decay)]
        if args.prueba:
            comando.append("--prueba")
        if args.clinica:
            comando.append("--clinica")
        print(f"Iniciando fold {fold}; las dos pérdidas se entrenan secuencialmente", flush=True)
        subprocess.run(comando, check=True)
    if not args.prueba:
        subprocess.run([sys.executable, str(repo / "analizar_resultados.py"), "--datos", str(args.datos),
                        "--runs", *carpetas, "--salida", str(args.salida / "analisis")], check=True)
    else:
        print("Prueba técnica completada; no constituye validación cruzada científica.")


if __name__ == "__main__":
    main()
