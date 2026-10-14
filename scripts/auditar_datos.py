"""Audita metadatos y separación por paciente sin cargar imágenes ni usar test para ajustar.

Uso: python scripts/auditar_datos.py --datos breastdcedl
Solo requiere la biblioteca estándar de Python.
"""
import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


def leer(path, columnas):
    with path.open(encoding="utf-8-sig", newline="") as archivo:
        reader = csv.DictReader(archivo)
        faltan = set(columnas) - set(reader.fieldnames or [])
        if faltan:
            raise ValueError(f"{path}: faltan columnas {sorted(faltan)}")
        filas = list(reader)
    if not filas:
        raise ValueError(f"{path}: no contiene filas")
    return filas


def auditar(raiz, comprobar_imagenes=False):
    pacientes = leer(raiz / "metadata/patients.csv", ["pid", "pCR", "split"])
    cortes = leer(raiz / "metadata/samples.csv", ["sample_id", "patient_id", "pCR", "split", "fold", "path_pre", "path_early", "path_late"])
    indice = {}
    for fila in pacientes:
        pid = fila["pid"]
        if pid in indice:
            raise ValueError(f"Paciente duplicado: {pid}")
        if fila["pCR"] not in ("0", "1") or fila["split"] not in ("train", "test"):
            raise ValueError(f"Etiqueta o split inválidos: {pid}")
        indice[pid] = fila
    grupos = defaultdict(list)
    ids, rutas = set(), set()
    faltantes = []
    for fila in cortes:
        pid = fila["patient_id"]
        if pid not in indice:
            raise ValueError(f"Corte sin paciente: {pid}")
        if fila["sample_id"] in ids:
            raise ValueError(f"Corte duplicado: {fila['sample_id']}")
        ids.add(fila["sample_id"])
        paciente = indice[pid]
        if any(fila[c] != paciente[c] for c in ("pCR", "split")):
            raise ValueError(f"Etiqueta o split incoherente: {pid}")
        permitidos = {"0", "1", "2", "3", "4"} if fila["split"] == "train" else {"-1"}
        if fila["fold"] not in permitidos:
            raise ValueError(f"Fold inválido: {pid}")
        grupos[pid].append(fila)
        for columna in ("path_pre", "path_early", "path_late"):
            ruta = (raiz / fila[columna]).resolve()
            if not ruta.is_relative_to(raiz.resolve()):
                raise ValueError(f"Ruta fuera del dataset: {fila[columna]}")
            if ruta in rutas:
                raise ValueError(f"Imagen reutilizada: {fila[columna]}")
            rutas.add(ruta)
            if comprobar_imagenes and (not ruta.is_file() or ruta.stat().st_size == 0):
                faltantes.append(fila[columna])
    if set(indice) != set(grupos):
        raise ValueError("Hay pacientes sin cortes")
    for pid, filas in grupos.items():
        if len({f["fold"] for f in filas}) != 1:
            raise ValueError(f"Fuga entre folds: {pid}")
    resumen = {}
    for split in ("train", "test"):
        seleccion = [p for p in pacientes if p["split"] == split]
        resumen[split] = {
            "pacientes": len(seleccion),
            "clases_por_paciente": dict(Counter(p["pCR"] for p in seleccion)),
            "cohortes": dict(Counter(p.get("dataset", "desconocida") for p in seleccion)),
            "cortes": sum(f["split"] == split for f in cortes),
        }
    resumen["folds_train"] = {
        str(fold): dict(Counter(indice[pid]["pCR"] for pid, filas in grupos.items()
                                if filas[0]["split"] == "train" and filas[0]["fold"] == str(fold)))
        for fold in range(5)
    }
    resumen["imagenes_indexadas"] = len(rutas)
    resumen["imagenes_comprobadas"] = comprobar_imagenes
    resumen["imagenes_ausentes_o_vacias"] = len(faltantes) if comprobar_imagenes else None
    resumen["ejemplos_ausentes"] = faltantes[:10]
    return resumen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datos", type=Path, default=Path("breastdcedl"))
    parser.add_argument("--comprobar-imagenes", action="store_true")
    parser.add_argument("--salida", type=Path, default=Path("reports/auditoria.json"))
    args = parser.parse_args()
    try:
        resumen = auditar(args.datos, args.comprobar_imagenes)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Auditoría fallida: {exc}\n")
    texto = json.dumps(resumen, indent=2, ensure_ascii=False)
    args.salida.parent.mkdir(parents=True, exist_ok=True)
    args.salida.write_text(texto + "\n", encoding="utf-8")
    print(texto)
    if resumen["imagenes_ausentes_o_vacias"]:
        parser.exit(1, "Descarga incompleta; reanuda el descargador.\n")


if __name__ == "__main__":
    main()
