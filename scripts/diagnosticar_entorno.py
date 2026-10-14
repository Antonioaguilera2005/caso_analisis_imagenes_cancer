"""Diagnóstico de PyTorch; no instala ni modifica paquetes."""
import json
import platform
import sys
from pathlib import Path


def main():
    info = {"python": sys.version, "executable": sys.executable,
            "platform": platform.platform()}
    try:
        import torch
        info.update(torch=torch.__version__, hip=torch.version.hip,
                    cuda=torch.version.cuda, gpu_available=torch.cuda.is_available())
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            x = torch.randn(128, 128, device="cuda")
            result = x @ x.T
            torch.cuda.synchronize()
            info["gpu_compute_ok"] = bool(torch.isfinite(result).all().item())
        else:
            info["gpu_compute_ok"] = False
    except ImportError:
        info["torch"] = "no instalado"
    except Exception as exc:
        info["error"] = str(exc)
    output = json.dumps(info, ensure_ascii=False, indent=2)
    print(output)
    Path("diagnostico_entorno.json").write_text(output + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
