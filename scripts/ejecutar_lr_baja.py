"""Última comparación controlada en GPU: CNN simple, solo imagen, lr=0,0003."""
import argparse
from pathlib import Path
import subprocess
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mostrar", action="store_true", help="Mostrar receta sin ejecutar")
    a = p.parse_args()
    repo = Path(__file__).resolve().parents[1]
    comando = [sys.executable, "-u", "validacion_cruzada.py",
               "--arquitectura", "simple", "--datos", "breastdcedl",
               "--salida", "runs/cv5_simple_lr3e4", "--dispositivo", "cuda",
               "--epocas", "60", "--paciencia", "12", "--batch-size", "16",
               "--workers", "2", "--semilla", "42", "--lr", "0.0003",
               "--dropout", "0.3", "--weight-decay", "0.0001"]
    print("CNN simple | solo imágenes | 5 folds | ambas pérdidas | sin test", flush=True)
    print(" ".join(comando), flush=True)
    if a.mostrar:
        return
    import torch
    if not torch.cuda.is_available():
        p.error("GPU no disponible: activar el entorno PyTorch/ROCm de la universidad")
    if (repo / "runs/cv5_simple_lr3e4").exists():
        p.error("La salida ya existe; conservarla y revisar antes de reiniciar")
    if not (repo / "breastdcedl/metadata/samples.csv").is_file():
        p.error("Falta el dataset en breastdcedl")
    print(f"GPU: {torch.cuda.get_device_name(0)} | ROCm: {torch.version.hip}", flush=True)
    subprocess.run(comando, cwd=repo, check=True)


if __name__ == "__main__":
    main()
