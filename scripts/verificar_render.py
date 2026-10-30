"""Ensayo técnico de cinco pacientes públicas de train; no evalúa rendimiento ni test."""
import argparse
import json
from pathlib import Path
import sys
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import requests
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'breastdcedl'))
import utils_caso as uc
from webapp.inferencia import Motor, preparar_archivos


def main():
    p = argparse.ArgumentParser(description=__doc__)
    modo = p.add_mutually_exclusive_group(required=True)
    modo.add_argument('--url', help='URL HTTPS pública de la aplicación propia; envía imágenes de train al servidor')
    modo.add_argument('--local', action='store_true', help='Prueba en memoria; no envía imágenes fuera del equipo')
    p.add_argument('--datos', type=Path, default=REPO / 'breastdcedl')
    p.add_argument('--paquete', type=Path, default=REPO / 'artifacts/dashboard')
    p.add_argument('--salida', type=Path, default=REPO / 'artifacts/verificacion_render/defensa_cinco_muestras.json')
    a = p.parse_args()
    url = a.url.rstrip('/') if a.url else ''
    if a.url and (urlparse(url).scheme != 'https' or urlparse(url).username or urlparse(url).password):
        raise ValueError('Usa una URL HTTPS sin credenciales.')
    motor = Motor(a.paquete)
    datos = pd.read_csv(a.datos / 'metadata/samples.csv')
    train = datos.loc[datos.split.eq('train')].sort_values(['patient_id', 'slice_index'])
    cohortes = pd.read_csv(a.datos / 'metadata/patients.csv', usecols=['pid', 'dataset'])
    candidatas = train.drop_duplicates('patient_id').merge(cohortes, left_on='patient_id', right_on='pid', validate='one_to_one')
    # Representar las tres cohortes sin consultar ni seleccionar por etiquetas.
    partes = [candidatas[candidatas.dataset.eq(cohorte)].sample(n=1, random_state=42)
              for cohorte in ('spy1', 'spy2', 'duke')]
    inicial = pd.concat(partes)
    restantes = candidatas[~candidatas.patient_id.isin(inicial.patient_id)].sample(n=2, random_state=42)
    muestras = pd.concat([inicial, restantes])
    assert len(muestras) == muestras.patient_id.nunique() == 5
    informe = dict(url=url or 'TestClient local, sin red', sha256=motor.sha, tipo='Ensayo técnico: cinco pacientes públicas de train; sin métricas de rendimiento', muestras=[])
    if a.local:
        from fastapi.testclient import TestClient
        from webapp.main import app
        cliente = TestClient(app)
    else:
        cliente = requests.Session()
    with cliente as session:
        salud = session.get(url + '/api/health', timeout=90)
        salud.raise_for_status()
        if salud.json()['modelo'] != motor.sha:
            raise ValueError('Render tiene un modelo diferente al paquete local.')
        meta = session.get(url + '/api/modelo', timeout=30)
        meta.raise_for_status()
        assert meta.json()['receta'] == motor.resumen['receta']
        primera_carga = None
        for _, fila in muestras.iterrows():
            carga = [(fase, (Path(fila[f'path_{fase.lower()}']).name,
                            (a.datos / fila[f'path_{fase.lower()}']).read_bytes(), 'image/png'))
                     for fase in ('PRE', 'EARLY', 'LATE')]
            x, ids = preparar_archivos([(nombre, contenido, fase) for fase, (nombre, contenido, _) in carga])
            oficial = uc.cargar_imagen(fila, a.datos)
            np.testing.assert_array_equal(x[0], oficial)
            with torch.inference_mode():
                prob_local = torch.sigmoid(motor.modelo(torch.from_numpy(oficial[None]))).item()
            respuesta = session.post(url + '/api/predecir', files=carga, timeout=60)
            respuesta.raise_for_status()
            r = respuesta.json()
            diferencia = abs(r['probabilidad'] - prob_local)
            assert diferencia <= 1e-6, (fila.sample_id, diferencia)
            assert r['sha256'] == motor.sha and r['umbral'] == motor.umbral
            assert r['clase'] == int(prob_local >= motor.umbral) and r['cortes'] == 1
            informe['muestras'].append(dict(patient_id=fila.patient_id, sample_id=fila.sample_id,
                                             probabilidad_pipeline=prob_local, diferencia_absoluta=diferencia,
                                             tensor_igual_oficial=True, resultado_web=r))
            print(f"{len(informe['muestras'])}/5: {fila.sample_id}; diferencia={diferencia:.2g}", flush=True)
            if primera_carga is None:
                primera_carga = carga
        repeticion = session.post(url + '/api/predecir', files=primera_carga, timeout=60)
        repeticion.raise_for_status()
        assert repeticion.json()['probabilidad'] == informe['muestras'][0]['resultado_web']['probabilidad']
        informe['repetibilidad_exacta'] = True
        informe['errores'] = []
        for nombre, carga in [
            ('corrupto', [('PRE', ('corrupto.png', b'no es una imagen', 'image/png'))] + primera_carga[1:]),
            ('fase_ausente', primera_carga[:2]),
            ('ruta_prohibida', [('PRE', ('../PRE.png', primera_carga[0][1][1], 'image/png'))] + primera_carga[1:]),
        ]:
            r = session.post(url + '/api/predecir', files=carga, timeout=30)
            # Render puede rechazar un nombre con ../ en su WAF antes de FastAPI.
            bloqueo_waf = (not a.local and nombre == 'ruta_prohibida' and r.status_code == 403
                           and 'web application firewall (WAF)' in r.text)
            assert r.status_code == 422 or bloqueo_waf, (nombre, r.status_code, r.text[:200])
            mensaje = 'Rechazada por el WAF de Render antes de llegar a FastAPI.' if bloqueo_waf else r.json()['detail']
            informe['errores'].append(dict(caso=nombre, status=r.status_code, mensaje=mensaje))
    a.salida.parent.mkdir(parents=True, exist_ok=True)
    a.salida.write_text(json.dumps(informe, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(a.salida, flush=True)


if __name__ == '__main__':
    main()
