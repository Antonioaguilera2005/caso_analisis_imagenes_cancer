"""Prepara paquete local de despliegue; datos y pesos permanecen fuera de Git."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import zipfile
import pandas as pd

REPO = Path(__file__).resolve().parents[1]


def leer(p):
    return json.loads(p.read_text(encoding='utf-8-sig'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--salida', type=Path, default=REPO/'artifacts/dashboard')
    a = p.parse_args()
    a.salida.mkdir(parents=True, exist_ok=False)
    final = REPO/'runs/modelo_final'
    artefacto, receta = leer(final/'artefacto.json'), leer(final/'receta.json')
    peso = (final/artefacto['checkpoint']).resolve()
    if not peso.is_relative_to(final.resolve()) or hashlib.sha256(peso.read_bytes()).hexdigest() != artefacto['sha256_checkpoint']:
        raise ValueError('Checkpoint no coincide')
    test = leer(REPO/'runs/test_final/metricas_test.json')
    if test['sha256_checkpoint'] != artefacto['sha256_checkpoint'] or test['umbral'] != receta['umbral']:
        raise ValueError('Test y modelo no coinciden')
    shutil.copyfile(peso, a.salida/'checkpoint.pt')
    experimentos = []
    for nombre, ruta in [('CNN sencilla · referencia', 'cv5_v3_60ep'), ('Imagen + edad y menopausia', 'cv5_clinica_v1'), ('Doble convolución', 'cv5_doble_v1'), ('Tasa de aprendizaje 0,0003', 'cv5_simple_lr3e4')]:
        tabla = pd.read_csv(REPO/'runs'/ruta/'analisis/metricas_por_fold.csv')
        for perdida, t in tabla.groupby('perdida'):
            experimentos.append(dict(nombre=nombre, perdida=perdida, auc_media=float(t.auc.mean()), auc_std=float(t.auc.std()), brecha=float(t.brecha_auc_train_val.mean())))
    curvas=[]
    for f in range(5):
        d=REPO/'runs/cv5_v3_60ep'/f'fold{f}_semilla42'/'normal'
        curvas.append(dict(fold=f, seleccion=leer(d/'seleccion.json')['epoca'], historial=[{k:v for k,v in h.items() if k in ('epoca','loss_train_eval','loss_val','auc_train','auc_val')} for h in leer(d/'historial.json')]))
    ejemplos=[]
    samples=pd.read_csv(REPO/'breastdcedl/metadata/samples.csv')
    train=samples[samples.split=='train']
    for i, clase in enumerate([0,1],1):
        pid=train[train.pCR==clase].patient_id.iloc[0]
        filas=train[train.patient_id==pid].sort_values('slice_index')
        eid=f'ejemplo-{i:02d}'
        carpeta=a.salida/'ejemplos'/eid
        carpeta.mkdir(parents=True)
        nombres=[]
        for fila in filas.itertuples():
            for fase in ['pre','early','late']:
                ruta=REPO/'breastdcedl'/getattr(fila, f'path_{fase}')
                shutil.copyfile(ruta,carpeta/ruta.name)
                nombres.append(ruta.name)
        ejemplos.append(dict(id=eid,nombre=f'Ejemplo docente {i:02d}',origen='Paciente de entrenamiento · no es una validación independiente',cortes=len(filas),archivos=nombres))
    figuras=[]
    rutas=[REPO/'runs/test_final/matriz_confusion_test.png',REPO/'runs/test_final/roc_pr_calibracion_test.png',REPO/'runs/graficas_entrega/cv5_v3_60ep/perdidas_normal.png']
    for ruta in rutas:
        if ruta.is_file():
            (a.salida/'figuras').mkdir(exist_ok=True)
            shutil.copyfile(ruta,a.salida/'figuras'/ruta.name)
            figuras.append(ruta.name)
    resumen=dict(version='BreastDCE · CNN-01',autor='Antonio Aguilera',sha256_checkpoint=artefacto['sha256_checkpoint'],
                 receta=receta,test=test,cohortes=leer(REPO/'runs/test_final/metricas_por_cohorte.json'),
                 experimentos=experimentos,curvas=curvas,ejemplos=ejemplos,figuras=figuras,
                 entrenamiento=leer(final/'entorno.json'),tiempo_entrenamiento=artefacto['segundos'],
                 memoria_gpu=artefacto['memoria_gpu_max_bytes'],
                 limitacion_test='El test se evaluó una vez para la versión original. Después se exploraron variantes con validación y se conservó el modelo original. Su resultado es histórico; no es una evaluación independiente de toda la búsqueda posterior.',
                 uso='Educativo y de investigación. Sin validez clínica. No diagnostica ni recomienda tratamientos o evitar cirugía.')
    (a.salida/'resumen.json').write_text(json.dumps(resumen,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    (a.salida/'ATRIBUCION.txt').write_text('BreastDCEDL MinCrop. Fridman et al., Scientific Data 2026, DOI 10.1038/s41597-026-06589-6. Fuentes: TCIA I-SPY1, I-SPY2 y Duke. Uso docente no comercial según enunciado CC BY-NC 4.0. LICENSE descargada dice CC BY 4.0; se conserva la discrepancia y se aplica el uso no comercial exigido por el caso. Ejemplos públicos de train; no se incluye validación privada. Resultados del estudiante Antonio Aguilera.\n',encoding='utf-8')
    paquete=a.salida.parent/'dashboard_bundle.zip'
    if paquete.exists():
        raise ValueError('El ZIP de salida ya existe; conservar el paquete anterior')
    with zipfile.ZipFile(paquete,'w',zipfile.ZIP_DEFLATED) as z:
        for ruta in sorted(a.salida.rglob('*')):
            if ruta.is_file():
                z.write(ruta,ruta.relative_to(a.salida).as_posix())
    checksum=hashlib.sha256(paquete.read_bytes()).hexdigest()
    paquete.with_suffix('.sha256').write_text(checksum+'\n',encoding='utf-8')
    print(f'Paquete local: {paquete}\nSHA-256: {checksum}\nNo subir a Git.')


if __name__=='__main__':
    main()
