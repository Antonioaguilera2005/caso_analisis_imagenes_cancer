"""Descarga opcional de un paquete autorizado por el administrador, nunca por el visitante."""
import hashlib
import io
import os
from pathlib import Path
import re
import urllib.request
import zipfile


def preparar_paquete_remoto(destino):
    url = os.environ.get('MODEL_BUNDLE_URL')
    if not url:
        return
    esperado = os.environ.get('MODEL_BUNDLE_SHA256','').lower()
    if not url.startswith('https://') or not re.fullmatch('[0-9a-f]{64}', esperado):
        raise ValueError('Configurar URL HTTPS y SHA-256 del paquete')
    with urllib.request.urlopen(url, timeout=60) as r:
        if not r.geturl().startswith('https://'):
            raise ValueError('Redirección no segura')
        contenido = r.read(32*1024*1024+1)
    if len(contenido) > 32*1024*1024 or hashlib.sha256(contenido).hexdigest() != esperado:
        raise ValueError('Paquete demasiado grande o checksum incorrecto')
    destino = Path(destino)
    with zipfile.ZipFile(io.BytesIO(contenido)) as z:
        if len(z.infolist()) > 300 or sum(i.file_size for i in z.infolist()) > 48*1024*1024:
            raise ValueError('Contenido del paquete excede límites')
        for info in z.infolist():
            n = info.filename
            permitido = n in ('checkpoint.pt','resumen.json','ATRIBUCION.txt') or re.fullmatch(r'(ejemplos/[a-z0-9_-]+/[A-Za-z0-9_-]+\.png|figuras/[A-Za-z0-9_-]+\.png)', n)
            if not permitido or info.is_dir():
                raise ValueError('Contenido de paquete no permitido')
        destino.mkdir(parents=True, exist_ok=True)
        for info in z.infolist():
            ruta = destino/info.filename
            ruta.parent.mkdir(parents=True, exist_ok=True)
            ruta.write_bytes(z.read(info))
