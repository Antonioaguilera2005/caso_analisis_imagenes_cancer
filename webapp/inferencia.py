"""Inferencia CPU del artefacto cerrado; sin entrenamiento ni lectura de etiquetas."""
import hashlib
import io
import json
import re
import threading
import time
from pathlib import Path
import numpy as np
from PIL import Image, UnidentifiedImageError
import torch
from modelo import CNNBreastDCE

PATRON = re.compile(r"^(?P<paciente>[A-Za-z0-9_-]+)_z(?P<corte>\d+)_(?P<fase>PRE|EARLY|LATE)\.png$", re.I)
FASES = ("PRE", "EARLY", "LATE")
MAX_ARCHIVO = 1024 * 1024
MAX_CORTES = 32


def cargar_png(contenido):
    if not contenido or len(contenido) > MAX_ARCHIVO:
        raise ValueError("Cada PNG debe ocupar entre 1 byte y 1 MB.")
    try:
        with Image.open(io.BytesIO(contenido)) as im:
            if im.format != "PNG" or im.size != (256, 256) or im.mode != "L" or getattr(im, "n_frames", 1) != 1:
                raise ValueError("Se requiere PNG de 256 × 256, escala de grises de 8 bits (modo L), sin animación.")
            im.load()
            x = np.asarray(im, dtype=np.float32) / np.float32(255.)
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ValueError("Archivo PNG corrupto o ilegible.") from exc
    if not np.isfinite(x).all() or x.min() < 0 or x.max() > 1:
        raise ValueError("Intensidades no válidas.")
    return x


def preparar_archivos(archivos):
    """(nombre, bytes, fase explícita o None); acepta un corte manual o lote nombrado."""
    if not archivos or len(archivos) > MAX_CORTES * 3:
        raise ValueError(f"Carga entre 3 y {MAX_CORTES * 3} imágenes de una sola paciente.")
    explicitas = any(fase is not None for _, _, fase in archivos)
    if explicitas and (len(archivos) != 3 or any(fase not in FASES for _, _, fase in archivos)):
        raise ValueError("Carga exactamente una PRE, una EARLY y una LATE del mismo corte.")
    grupos, pacientes = {}, set()
    for nombre, datos, explicita in archivos:
        # El nombre solo identifica la muestra; nunca se abre como ruta.
        if not nombre or '/' in nombre or '\\' in nombre or len(nombre) > 180:
            raise ValueError("Nombre de archivo no válido; no se aceptan rutas.")
        m = PATRON.fullmatch(nombre)
        if m:
            paciente, corte, fase = m.group('paciente'), str(int(m.group('corte'))), m.group('fase').upper()
            if explicita and explicita != fase:
                raise ValueError("El nombre de la fase no coincide con la casilla seleccionada.")
            pacientes.add(paciente)
            clave = (paciente, corte)
        elif explicitas:
            clave, fase = ('muestra_manual', '0'), explicita
        else:
            raise ValueError("Para un lote usa nombres PACIENTE_z012_PRE.png, _EARLY.png y _LATE.png.")
        grupo = grupos.setdefault(clave, {})
        if fase in grupo:
            raise ValueError("Hay una fase duplicada para el mismo corte.")
        grupo[fase] = cargar_png(datos)
    if len(pacientes) > 1 or (explicitas and len(grupos) != 1):
        raise ValueError("Las imágenes deben pertenecer a una sola paciente y las tres fases al mismo corte.")
    if any(set(g) != set(FASES) for g in grupos.values()):
        raise ValueError("Faltan fases: cada corte necesita PRE, EARLY y LATE.")
    orden = sorted(grupos, key=lambda k: (k[0], int(k[1])))
    return np.stack([np.stack([grupos[k][f] for f in FASES]) for k in orden]), [f"z{int(k[1]):03d}" for k in orden]


class Motor:
    def __init__(self, carpeta):
        carpeta = Path(carpeta)
        self.resumen = json.loads((carpeta/'resumen.json').read_text(encoding='utf-8'))
        peso = carpeta/'checkpoint.pt'
        checksum = hashlib.sha256(peso.read_bytes()).hexdigest()
        if checksum != self.resumen['sha256_checkpoint']:
            raise ValueError('Checksum del modelo incorrecto')
        cp = torch.load(peso, map_location='cpu', weights_only=True)
        r = cp['receta']
        if cp.get('tipo') != 'cnn_final' or cp.get('prueba', True) or cp.get('epoca') != r['epocas'] or cp.get('pacientes_train') != r['pacientes_train']:
            raise ValueError('No es el modelo final completo')
        if r['canales'] != list(FASES) or r['normalizacion'] != 'float32 / 255' or r['agregacion'] != 'mean':
            raise ValueError('Pipeline no compatible')
        sha_fuente = hashlib.sha256(Path(__file__).resolve().parents[1].joinpath('modelo.py').read_text(encoding='utf-8').replace('\r\n','\n').encode()).hexdigest()
        if sha_fuente != r['sha256_modelo_texto'] or self.resumen['receta'] != r:
            raise ValueError('La arquitectura o receta no corresponde al artefacto')
        self.modelo = CNNBreastDCE(r['dropout'])
        self.modelo.load_state_dict(cp['state_dict'])
        self.modelo.eval()
        torch.set_num_threads(2)
        self.umbral = float(r['umbral'])
        self.sha = checksum
        self.lock = threading.Lock()

    def predecir(self, x, ids):
        inicio = time.perf_counter()
        with self.lock, torch.inference_mode():
            self.modelo.eval()
            probs = []
            for i in range(0, len(x), 4):
                probs.extend(torch.sigmoid(self.modelo(torch.from_numpy(x[i:i+4]))).numpy().tolist())
        if not np.isfinite(probs).all():
            raise ValueError('Predicción no finita')
        prob = float(np.mean(probs))
        return dict(probabilidad=prob, clase=int(prob >= self.umbral), umbral=self.umbral,
                    cortes=len(probs), agregacion='Media aritmética de probabilidades',
                    alcance='Un corte: estimación parcial' if len(probs)==1 else 'Paciente: cortes aportados',
                    por_corte=[dict(corte=k, probabilidad=float(p)) for k,p in zip(ids,probs)],
                    sha256=self.sha, dispositivo='CPU', segundos=time.perf_counter()-inicio)
