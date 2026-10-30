"""Prepara informe PDF, pesos y presentación usando exclusivamente artefactos guardados."""
import argparse
import base64
import hashlib
import html
import json
from pathlib import Path
import shutil
import zipfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from playwright.sync_api import sync_playwright
from pypdf import PdfReader

REPO = Path(__file__).resolve().parents[1]
URL = 'https://breastdce-antonio.onrender.com'
GITHUB = 'https://github.com/Antonioaguilera2005/caso_analisis_imagenes_cancer'


def leer(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(datos):
    return hashlib.sha256(datos).hexdigest()


def num(valor, decimales=3):
    return f'{valor:.{decimales}f}'.replace('.', ',')


def tabla(cabeceras, filas):
    return '<table><thead><tr>'+''.join(f'<th>{html.escape(str(c))}</th>' for c in cabeceras)+'</tr></thead><tbody>'+''.join('<tr>'+''.join(f'<td>{html.escape(str(c))}</td>' for c in fila)+'</tr>' for fila in filas)+'</tbody></table>'


def imagen(path, alt='', clase=''):
    p = Path(path)
    tipo = 'image/svg+xml' if p.suffix == '.svg' else 'image/png'
    return f'<img class="{clase}" alt="{html.escape(alt)}" src="data:{tipo};base64,{base64.b64encode(p.read_bytes()).decode()}">'


def empaquetar_pesos(salida, resumen):
    peso = REPO/'runs/modelo_final/checkpoints/epoca_005.pt'
    if sha(peso.read_bytes()) != resumen['sha256_checkpoint']:
        raise ValueError('El checkpoint no coincide con el modelo publicado.')
    contenido = {
        'modelo_final/checkpoints/epoca_005.pt': peso.read_bytes(),
        'modelo_final/receta.json': (REPO/'runs/modelo_final/receta.json').read_bytes(),
        'modelo_final/artefacto.json': (REPO/'runs/modelo_final/artefacto.json').read_bytes(),
        'modelo_final/entorno.json': (REPO/'runs/modelo_final/entorno.json').read_bytes(),
        'modelo.py': (REPO/'modelo.py').read_bytes(),
        'requirements-inferencia.txt': b'--extra-index-url https://download.pytorch.org/whl/cpu\ntorch==2.14.1+cpu\n',
    }
    contenido['modelo_final/receta.sha256'] = (sha(contenido['modelo_final/receta.json'])+'\n').encode()
    contenido['comprobar_pesos.py'] = '''"""Verifica y carga el modelo; no entrena ni evalúa ningún dataset."""
import hashlib
import json
from pathlib import Path
import torch
from modelo import CNNBreastDCE

raiz = Path(__file__).resolve().parent
carpeta = raiz / 'modelo_final'
artefacto = json.loads((carpeta / 'artefacto.json').read_text(encoding='utf-8'))
peso = (carpeta / artefacto['checkpoint']).resolve()
assert peso.is_relative_to(carpeta.resolve())
checksum = hashlib.sha256(peso.read_bytes()).hexdigest()
assert checksum == artefacto['sha256_checkpoint'], 'Checksum incorrecto'
checkpoint = torch.load(peso, map_location='cpu', weights_only=True)
receta = json.loads((carpeta / 'receta.json').read_text(encoding='utf-8'))
assert checkpoint['receta'] == receta
assert checkpoint['tipo'] == 'cnn_final' and not checkpoint['prueba']
assert checkpoint['epoca'] == receta['epocas'] == 5
assert checkpoint['pacientes_train'] == receta['pacientes_train'] == 1097
fuente = (raiz / 'modelo.py').read_text(encoding='utf-8').replace('\\r\\n', '\\n').encode()
assert hashlib.sha256(fuente).hexdigest() == receta['sha256_modelo_texto']
modelo = CNNBreastDCE(receta['dropout'])
modelo.load_state_dict(checkpoint['state_dict'])
modelo.eval()
assert sum(p.numel() for p in modelo.parameters()) == receta['parametros'] == 97809
print('Modelo final cargado correctamente en CPU; 97.809 parámetros.')
print('SHA-256:', checksum)
print('Fases: PRE, EARLY, LATE; normalización: float32 /255.')
print('Media de probabilidades por paciente; umbral fijo:', receta['umbral'])
'''.encode('utf-8')
    contenido['LEEME_PESOS.txt'] = f'''PESOS DEFINITIVOS — ANTONIO AGUILERA — BREASTDCEDL

Este ZIP contiene el checkpoint completo de la época 5, copiado sin modificar.
Incluye state_dict (pesos y estadísticas BatchNorm), receta y metadatos.
No contiene el dataset, imágenes de pacientes, validación privada ni otros folds.
Los checkpoints de todas las épocas se conservan como respaldo local; se entrega
solo el checkpoint elegido para la CNN final y utilizado por la aplicación.

SHA-256 del archivo epoca_005.pt:
{resumen['sha256_checkpoint']}

Arquitectura: CNNBreastDCE, modelo.py; 97.809 parámetros, desde cero.
Entrada: tensor float32 [N,3,256,256], gris 8 bits /255, PRE/EARLY/LATE.
Inferencia: eval(), torch.inference_mode(), sigmoid por corte.
Paciente: media aritmética de probabilidades; clase pCR si media >=0.33.
No aplicar normalización de ImageNet ni normalizar cada fase por separado.

Comprobación tras descomprimir:
  python comprobar_pesos.py
Si falta PyTorch en un entorno CPU de Python 3.11:
  python -m pip install -r requirements-inferencia.txt
No hace falta reinstalar un entorno PyTorch/ROCm que ya funcione.

El pipeline completo está en:
{GITHUB}
En ese repositorio también se puede usar:
  from finalizar import cargar_modelo_final
  modelo, receta, artefacto = cargar_modelo_final('RUTA/modelo_final')
Esto solo carga el modelo; no es necesario volver a ejecutar evaluar_test.

Aplicación: {URL}
Uso exclusivamente educativo. Sin validez clínica.
'''.encode('utf-8')
    contenido['MANIFIESTO_SHA256.txt'] = ''.join(f'{sha(datos)}  {nombre}\n' for nombre, datos in sorted(contenido.items())).encode()
    destino = salida/'PESOS_MODELO_FINAL.zip'
    with zipfile.ZipFile(destino, 'w', zipfile.ZIP_DEFLATED) as z:
        for nombre, datos in contenido.items():
            z.writestr(nombre, datos)
    with zipfile.ZipFile(destino) as z:
        assert z.testzip() is None
        assert sha(z.read('modelo_final/checkpoints/epoca_005.pt')) == resumen['sha256_checkpoint']
    destino.with_suffix('.zip.sha256').write_text(sha(destino.read_bytes())+'\n', encoding='utf-8')


def crear_informe(salida, m):
    t, r = m['test'], m['receta']
    auditoria = leer(REPO/'reports/auditoria.json')
    cohortes = leer(REPO/'runs/test_final/metricas_por_cohorte.json')
    cv_original = leer(REPO/'cv5_v2/analisis/resumen.json')
    referencias = [('Original · 40 épocas máx.', REPO/'cv5_v2'), ('Horizonte 60 épocas', REPO/'runs/cv5_v3_60ep'), ('Imagen + edad/menopausia', REPO/'runs/cv5_clinica_v1'), ('Doble convolución', REPO/'runs/cv5_doble_v1'), ('Learning rate 0,0003', REPO/'runs/cv5_simple_lr3e4')]
    filas = []
    for nombre, carpeta in referencias:
        d = pd.read_csv(carpeta/'analisis/metricas_por_fold.csv')
        fila = [nombre]
        for perdida in ('normal', 'ponderada'):
            v = d.loc[d.perdida.eq(perdida), 'auc']
            assert len(v) == 5
            fila.append(f'{num(v.mean())} ± {num(v.std())}')
        filas.append(fila)
    comparacion = tabla(['Experimento', 'BCE normal · AUC', 'BCE ponderada · AUC'], filas)
    fold_metricas = pd.read_csv(REPO/'cv5_v2/analisis/metricas_por_fold.csv').query("perdida == 'normal'")
    fold_tabla = tabla(['Fold', 'Época', 'AUC train', 'AUC validación', 'Brecha'], [[int(f.fold), int(f.epoca), num(f.auc_train_elegida), num(f.auc), num(f.brecha_auc_train_val)] for _, f in fold_metricas.iterrows()])
    # Mostrar todos los historiales reales, sin alisado y sin ejecutar inferencia.
    fig, axs = plt.subplots(5, 2, figsize=(7.1, 8.2))
    historiales = [pd.DataFrame(leer(REPO/f'runs/cv5_v3_60ep/fold{i}_semilla42/normal/historial.json')) for i in range(5)]
    bcemax = max(float(h[['loss_train_eval','loss_val']].max().max()) for h in historiales)
    for i, (h, e) in enumerate(zip(historiales, r['resumen_cv']['normal']['epocas'])):
        for j, (claves, ylabel) in enumerate([( ('loss_train_eval','loss_val'), 'BCE por corte'), (('auc_train','auc_val'), 'AUC por paciente')]):
            ax = axs[i,j]
            for clave, color, etiqueta in zip(claves, ['#0760AB','#BD6C39'], ['Train sin aumentos','Validación']):
                ax.plot(h.epoca,h[clave],color=color,lw=1.1,label=etiqueta)
            ax.axvline(e,color='#61798B',ls='--',lw=.8)
            ax.set(title=f'Fold {i} · época elegida {e}',xlabel='Época',ylabel=ylabel,ylim=(0,bcemax*1.05) if j==0 else (0,1))
            ax.spines[['top','right']].set_visible(False)
            ax.grid(alpha=.17);ax.tick_params(labelsize=6)
            ax.xaxis.label.set_size(7);ax.yaxis.label.set_size(7);ax.title.set_size(8)
            if i==0:ax.legend(fontsize=6,frameon=False,loc='best')
    fig.tight_layout(h_pad=.8,w_pad=1.2)
    curvas = salida/'assets/curvas_cinco_folds.png'
    fig.savefig(curvas,dpi=190);plt.close(fig)
    ejemplo=m['ejemplos'][0]; grupo={}
    for nombre in ejemplo['archivos']:
        muestra,fase=nombre.rsplit('_',1)
        grupo.setdefault(muestra,{})[fase.removesuffix('.png')]=nombre
    corte=list(grupo.values())[len(grupo)//2]
    fases=''.join(f'<figure>{imagen(REPO/"artifacts/dashboard/ejemplos"/ejemplo["id"]/corte[f], f)}<figcaption>{f}</figcaption></figure>' for f in ['PRE','EARLY','LATE'])
    clases = tabla(['Conjunto', 'Pacientes', 'no pCR / pCR', 'Cortes'], [[etiqueta, auditoria[k]['pacientes'], f"{auditoria[k]['clases_por_paciente']['0']} / {auditoria[k]['clases_por_paciente']['1']}", auditoria[k]['cortes']] for k,etiqueta in [('train','Train público'),('test','Test histórico')]])
    ctabla = tabla(['Cohorte','Train','Test'], [[nombre, auditoria['train']['cohortes'][k], auditoria['test']['cohortes'][k]] for k,nombre in [('spy1','I-SPY1'),('spy2','I-SPY2'),('duke','Duke')]])
    arquitectura = tabla(['Etapa','Salida C × H × W','Parámetros'], [['Entrada','3 × 256 × 256','0'],['Conv/BN/ReLU/Pool 1','16 × 128 × 128','464'],['Conv/BN/ReLU/Pool 2','32 × 64 × 64','4.672'],['Conv/BN/ReLU/Pool 3','64 × 32 × 32','18.560'],['Conv/BN/ReLU/Pool 4','128 × 16 × 16','73.984'],['GAP / flatten / dropout','128','0'],['Linear','1 logit','129'],['TOTAL','','97.809']])
    config = tabla(['Parámetro','Valor'], [['Optimizador','Adam'],['Learning rate',r['lr']],['Weight decay',r['weight_decay']],['Batch',r['batch_size']],['Dropout',r['dropout']],['Semilla',r['semilla']],['Inicialización','Kaiming en Conv / Xavier en Linear'],['Modelo final','5 épocas fijas; BCE normal']])
    metrica = tabla(['Métrica','CNN final','Siempre no pCR'], [[nombre,num(t[k]),num(t['baseline_siempre_no_pcr'][k])] for k,nombre in [('auc','ROC-AUC'),('accuracy','Accuracy'),('sensibilidad','Sensibilidad pCR'),('especificidad','Especificidad'),('f1_macro','F1 macro'),('balanced_accuracy','Balanced accuracy')]])
    por_clase = tabla(['Clase','Precisión','Recall','F1','Soporte'], [[etiqueta,num(t['por_clase'][k]['precision']),num(t['por_clase'][k]['recall']),num(t['por_clase'][k]['f1']),t['por_clase'][k]['soporte']] for k,etiqueta in [('0','no pCR'),('1','pCR')]])
    ct = tabla(['Cohorte','N','pCR reales','AUC','Sensibilidad','Especificidad'], [[nombre,cohortes[k]['pacientes'],cohortes[k]['por_clase']['1']['soporte'],num(cohortes[k]['auc']),num(cohortes[k]['sensibilidad']),num(cohortes[k]['especificidad'])] for k,nombre in [('duke','Duke'),('spy1','I-SPY1'),('spy2','I-SPY2')]])
    normal,ponderada = [cv_original[k]['referencia'] for k in ('normal','ponderada')]
    paginas=[]
    paginas.append(('Problema y resumen', f'''
    <p class="eyebrow">CASO 1 · APRENDIZAJE AUTOMÁTICO · 30 OCTUBRE 2026</p>
    <h1>Predicción de respuesta<br>patológica completa<br>con DCE-MRI</h1>
    <p class="lead">CNN propia, entrenada desde cero, con una decisión por paciente.</p>
    <p><strong>Antonio Aguilera</strong> · Informe del proyecto individual</p>
    <div class="notice">Demostrador educativo. <strong>Sin validez clínica.</strong> No detecta cáncer ni recomienda tratamientos.</div>
    <h2>Objetivo</h2><p>Estimar si una paciente alcanzará pCR a partir de una resonancia previa al tratamiento neoadyuvante. pCR=1 significa ausencia de enfermedad invasiva residual en el tejido evaluado tras el tratamiento, según la definición del estudio. La etiqueta se conoce posteriormente mediante anatomía patológica; no equivale a curación ni supervivencia.</p>
    <h2>Resultado principal</h2><div class="kpis"><div><b>{num(t['auc'])}</b><span>AUC · test histórico</span></div><div><b>{num(t['f1_macro'])}</b><span>F1 macro</span></div><div><b>0,33</b><span>umbral de validación</span></div></div>
    <p>La discriminación es modesta: IC95% de AUC [{num(t['auc_ic95_bootstrap'][0])}, {num(t['auc_ic95_bootstrap'][1])}]. La sensibilidad es {num(t['sensibilidad'])} y existen diferencias importantes por cohorte. No se demuestra utilidad clínica.</p>
    <h2>Decisión y alcance</h2><p>Se conserva una CNN de cuatro bloques y 97.809 parámetros, con entrada de imagen exclusivamente. Añadir edad y menopausia, duplicar convoluciones o reducir el learning rate no produjo una mejora consistente que justificase sustituirla.</p>
    <div class="warning"><strong>Salvedad metodológica:</strong> se consultó una vez el test original antes de explorar variantes. Se conservó el modelo original y no se repitió su test. Este resultado es histórico; no representa una evaluación independiente de toda la búsqueda posterior. La cronología se declara en la discusión.</div>
    <p class="small">Aplicación: <a href="{URL}">{URL}</a><br>Repositorio: <a href="{GITHUB}">{GITHUB}</a></p>'''))
    paginas.append(('Datos y preparación', f'''
    <h2>01 / Entender los datos y evitar fugas</h2>
    <p>La edición docente incluye 1.273 pacientes, 12.703 cortes y 38.109 PNG. Cada corte aporta tres fases temporales: PRE, EARLY y LATE. Un índice z identifica altura anatómica; no es un instante de tratamiento.</p>{clases}
    <div class="two"><div><h3>Desbalance</h3><p>En train, 775 pacientes son no pCR y 322 pCR: 70,6% frente a 29,4%. Reportar únicamente accuracy favorecería un clasificador mayoritario sin detección de pCR.</p></div><div>{ctabla}</div></div>
    <div class="phases">{fases}</div><p class="caption">Ejemplo desidentificado del train público. Misma paciente, corte y ventana de intensidades. No es una muestra independiente de evaluación.</p>
    <h3>Preprocesado idéntico en entrenamiento e inferencia</h3><p>PNG gris de 8 bits, 256×256 → arrays float32 divididos por 255 → apilado PRE/EARLY/LATE, tensor 3×256×256. No se aplica normalización de ImageNet, RGB ni una nueva normalización independiente por fase. El realce EARLY−PRE del visor es descriptivo; no es segmentación ni un mapa de atención.</p>
    <h3>Separación y aumentos</h3><p>Cinco folds oficiales por paciente, estratificados por pCR. Se comprueba que los IDs de train y validación sean disjuntos; cada paciente valida una vez. Los flips horizontales y verticales se aplican al tensor completo, solo durante entrenamiento. Al evaluar se utiliza <code>shuffle=False</code>, modo eval y sin gradientes.</p>
    <p>Los cortes de una paciente están correlacionados. La pérdida se calcula por corte: pacientes con más cortes contribuyen más al entrenamiento. I-SPY selecciona cortes por máscara tumoral; Duke utiliza límites tumorales. Esta heterogeneidad puede introducir sesgo de cohorte.</p>'''))
    paginas.append(('Arquitectura y entrenamiento', f'''
    <h2>02 / Una CNN propia de 97.809 parámetros</h2><p>Cuatro bloques Conv3×3, BatchNorm, ReLU y MaxPool2. La convolución tiene padding1, stride1 y sin bias; conserva el tamaño espacial. Cada pooling divide la resolución entre dos.</p>{arquitectura}
    <div class="equation">Xᵦ = MaxPool₂ [ReLU (BN (Wᵦ ∗ Xᵦ₋₁))]<br>f꜀ = promedio de las 16×16 posiciones del mapa c<br>zᵢ = wᵀ Dropout₀,₃(fᵢ) + b &nbsp; ; &nbsp; pᵢ = sigmoid(zᵢ)</div>
    <p>Los parámetros por bloque incluyen los pesos y sesgos aprendidos de BatchNorm. GAP reduce cada mapa a una media y evita una gran capa densa dependiente de la resolución. Dropout y weight decay regularizan; no garantizan ausencia de sobreajuste. La salida es un único logit, sin pesos preentrenados ni transfer learning.</p>
    <h3>Entrenamiento reproducible</h3><div class="two"><div>{config}</div><div><p class="small">Entrenamiento final: Linux, AMD Radeon RX 6700 XT, PyTorch 2.14.0+rocm7.14 y Python 3.12.14. Duración {num(m['tiempo_entrenamiento'],1)} s; pico GPU {num(m['memoria_gpu']/1024**2,1)} MiB. Inferencia del dashboard en CPU.</p><p class="small">Las cinco épocas son la mediana de [3,5,29,5,6]. El modelo final se entrena con todo train, durante ese número fijo; no se detiene con test.</p><p class="small">Semillas y configuración están documentadas. Entre plataformas pueden existir pequeñas diferencias numéricas. La CV en mejores épocas (0,604 ± 0,045) no equivale a fijar cinco épocas en todos los folds (≈0,573 ± 0,033).</p></div></div>'''))
    paginas.append(('Validación y decisiones', f'''
    <h2>03 / Comparar hipótesis con los mismos folds</h2><p>ROC-AUC por paciente: media ± desviación muestral entre cinco folds. No es el intervalo de confianza de una evaluación externa. Se compararon BCE normal y BCE ponderada en cada experimento.</p>{comparacion}
    <h3>Normal frente a ponderada</h3><p>BCEWithLogitsLoss usa α=1 en la pérdida normal y α=N₀/N₁ de los <em>cortes de train del fold</em> en la ponderada. Penalizar más los falsos negativos pCR desplaza las puntuaciones y puede aumentar sensibilidad a costa de especificidad; no garantiza una AUC mayor.</p>
    <p>En la CV original, con umbral de referencia 0,5, sensibilidad OOF normal {num(normal['sensibilidad'])} y especificidad {num(normal['especificidad'])}; ponderada {num(ponderada['sensibilidad'])} y {num(ponderada['especificidad'])}. Estas métricas mezclan predicciones de cinco modelos y no deben confundirse con la media de sus AUC.</p>
    <h3>Por qué se conserva el modelo original</h3><p>La clínica fue una comparación adicional: no se incorpora al modelo principal. Los faltantes pueden codificar la cohorte. Doble convolución elevó la capacidad a 294.129 parámetros sin una ganancia consistente. Con lr menor, la diferencia de AUC media fue mínima y mejoró un fold mientras empeoraban cuatro. No se elige por un fold aislado ni se afirma superioridad estadística.</p>
    <h3>Selección original y sobreajuste</h3>{fold_tabla}
    <p class="small">La época se selecciona por AUC de validación por paciente. La búsqueda original usó máximo 40 épocas/paciencia8; la extensión a 60/paciencia12 conservó las mismas selecciones normales. Se guarda un checkpoint por época y se señala el elegido. Una brecha negativa aislada no demuestra mejor generalización: hay variabilidad muestral y de entrenamiento.</p>'''))
    paginas.append(('Curvas y cierre del modelo', f'''
    <h2>04 / Mostrar el sobreajuste, no ocultarlo</h2><p class="small">Historiales completos de <code>cv5_v3_60ep</code>, BCE normal. Train se mide sin aumentos; BCE sin ponderar por corte, AUC por paciente. Ejes comunes entre folds, sin suavizado. Discontinua: época elegida; se conservan visibles las posteriores.</p>
    {imagen(curvas, 'Curvas completas de pérdida y AUC, train y validación, cinco folds', 'curvas')}
    <p class="small">Los picos de pérdida no bastan para diagnosticar sobreajuste. Importa la separación persistente train/validación. El fold 3 muestra una brecha importante desde temprano; entrenar más no la resuelve. Una CNN pequeña también puede sobreajustar con pocos pacientes independientes.</p>'''))
    paginas.append(('Agregación, umbral y test histórico', f'''
    <h2>05 / Una decisión final por paciente</h2><div class="equation">p̂paciente = (p₁ + ⋯ + pₙ) / n &nbsp; ; &nbsp; ŷ = 1 si p̂paciente ≥ 0,33</div>
    <p>La agregación es la media de probabilidades, no de logits ni un voto por imagen. El umbral 0,33 se eligió por máximo F1 macro OOF en una rejilla 0,01–0,99, con desempate hacia 0,5. Es un criterio estadístico; no un balance clínico de costes validado. Elegir época y umbral con la misma validación introduce optimismo: no es CV anidada.</p>
    <h3>Evaluación guardada: 176 pacientes</h3>{metrica}
    <div class="two"><div>{imagen(REPO/'runs/test_final/matriz_confusion_test.png', 'Matriz del test histórico: 93 VN, 30 FP, 37 FN y 16 VP', 'matriz')}</div><div><h3>Interpretación de la matriz</h3><p>Filas reales, columnas predichas; orden no pCR/pCR. VN=93, FP=30, FN=37, VP=16.</p><p>Se identifican 16 de las 53 pCR; se omiten 37. De 46 predicciones de pCR, 30 son falsas. Accuracy inferior al baseline mayoritario, aunque se obtienen AUC y F1 macro mayores.</p></div></div>
    {por_clase}<p class="small">IC95% AUC [{num(t['auc_ic95_bootstrap'][0])}, {num(t['auc_ic95_bootstrap'][1])}], bootstrap estratificado por paciente, condicional al modelo elegido. No recoge toda la incertidumbre de selección. Estas métricas se leen de archivos ya calculados; no se vuelve a consultar test.</p>'''))
    paginas.append(('Discriminación, calibración y cohortes', f'''
    <h2>06 / El rendimiento global no cuenta toda la historia</h2>
    {imagen(REPO/'runs/test_final/roc_pr_calibracion_test.png', 'Curvas ROC, precisión-recall y calibración del test histórico', 'roc')}
    <p>ROC-AUC resume el ordenamiento de puntuaciones; no es el porcentaje de aciertos. Average precision = {num(t['average_precision'])}; Brier = {num(t['brier'])}. La curva de calibración compara frecuencias observadas y probabilidades: con pocos pacientes por intervalo es inestable y no certifica que la salida sea una confianza clínica fiable.</p>
    <h3>Resultados por procedencia</h3>{ct}
    <p>Duke tiene la mayor AUC observada, pero solo nueve pCR. I-SPY1 tiene AUC inferior al azar y no detecta ninguna de sus doce pCR con el umbral fijado. No se invierten probabilidades ni se reajusta el umbral con ese test. El hallazgo se documenta como un problema de generalización.</p>
    <p>Las diferencias pueden relacionarse con adquisiciones, selección de cortes y composición clínica. Estos resultados no demuestran qué factor causa el fallo; tampoco permiten afirmar que la red haya aprendido únicamente biología tumoral. Los subgrupos son pequeños y los intervalos globales amplios.</p>
    <h3>Qué significa un error</h3><p>Un falso positivo predice respuesta completa cuando queda enfermedad invasiva residual. Podría ser especialmente grave si alguien lo usase para reducir tratamiento u omitir cirugía, decisiones que esta app no permite recomendar. Un falso negativo omite una respuesta completa y podría llevar a una estimación demasiado pesimista. La gravedad relativa depende del uso; en este proyecto el umbral no se presenta como decisión clínica.</p>
    <div class="warning">No se puede justificar el modelo únicamente por acercarse al 65% de accuracy ni por que sus curvas parezcan suaves. Importan la clase minoritaria, la calibración, la validación y los sesgos.</div>'''))
    paginas.append(('Aplicación y reproducibilidad', f'''
    <h2>07 / Una demostración con el pipeline real</h2><p><a href="{URL}">{URL}</a> · HTML/CSS/JavaScript y FastAPI. Pesos reales del modelo final, inferencia CPU, sin reentrenamiento y sin consultar etiquetas.</p>
    <h3>Entrada y resultado</h3><p>Un corte con PRE/EARLY/LATE o un lote de una sola paciente, además de dos ejemplos públicos de train. Se muestran las fases por separado y un mapa descriptivo EARLY−PRE. La salida incluye probabilidad media, clase, umbral, SHA-256, número de cortes, dispositivo y duración. Un corte se identifica como estimación parcial; la media usa solo los cortes aportados.</p>
    <h3>Validación y ejecución</h3><p>PNG gris 8 bits, 256², sin animación; máximo 1 MB por imagen, 32 cortes/96 PNG y 16 MB por solicitud. Se rechazan fases duplicadas/ausentes, mezcla de pacientes, nombres con rutas y contenido corrupto. No se ejecutan subidas ni se guardan en disco. El motor comprueba finitud, carga pesos propios con <code>weights_only=True</code>, llama a <code>eval()</code> y usa <code>torch.inference_mode()</code>. Las salidas no finitas no emiten una predicción y generan un mensaje legible.</p>
    <h3>Evidencia técnica verificada el 30/10/2026</h3>{tabla(['Comprobación','Resultado'], [['Pruebas automatizadas','26 pruebas pasan; incluye modelo real y normalización'],['Normalización','Tensor exactamente igual a utils_caso.cargar_imagen'],['Cinco pacientes de train · local','Cinco probabilidades idénticas al pipeline original'],['Cinco pacientes de train · Render','Diferencia máxima 5,96×10⁻⁸; mismas clases, umbral y SHA'],['Repetición en Render','Misma probabilidad exacta'],['Errores de entrada','Corrupto/fase ausente 422; ruta rechazada por app/WAF'],['Navegador Chrome','Ejemplo, corte, lote, navegación y móvil sin errores JS']])}
    <p>Los ensayos usan datos públicos de train y verifican coherencia de software; no estiman rendimiento independiente. La validación privada del profesor no se ha consultado y no se publica. El registro JSON de defensa se exporta en el navegador. Un arranque tras inactividad en el plan gratuito puede tardar: comprobar disponibilidad antes de exponer.</p>
    <h3>Identidad del modelo entregado</h3><p class="hash">{m['sha256_checkpoint']}</p><p class="small">Checkpoint original: <code>runs/modelo_final/checkpoints/epoca_005.pt</code>. El ZIP de pesos conserva esos bytes, la arquitectura, receta, entorno y un verificador de carga. Repositorio con configuración/semillas y dependencias; datos, pesos y resultados se distribuyen fuera de Git.</p>'''))
    paginas.append(('Discusión, límites y fuentes', f'''
    <h2>08 / Conclusiones y límites de la evidencia</h2>
    <p>Se ha construido un pipeline reproducible de DCE-MRI con tres fases, una CNN propia desde cero, validación por paciente, comparación de pérdidas e inferencia web coherente con el entrenamiento. Los experimentos más complejos no justificaron sustituir el modelo original. El resultado predictivo es modesto y heterogéneo por cohorte.</p>
    <h3>Cronología del test: salvedad explícita</h3><div class="warning">La guía C5 exige cerrar el modelo antes de consultar test y prohíbe volver a ajustarlo después. Se evaluó una vez el modelo original; posteriormente se exploraron variantes con validación. Se conservó el modelo original y no se repitió su test. Esto no permite declarar cumplimiento pleno de la regla ni independencia del test respecto de toda la búsqueda. La secuencia debe aclararse con el profesor; no se corrige retrospectivamente cambiando el texto o borrando registros.</div>
    <h3>Limitaciones metodológicas</h3><ul><li>Pocos pacientes independientes respecto al número de cortes correlacionados; ponderación implícita por número de cortes.</li><li>Selección de época y umbral sobre validación; OOF ajustada optimista y ausencia de CV anidada.</li><li>Transferencia del umbral OOF al reentrenamiento; cinco épocas finales no equivalen al mejor checkpoint de cada fold.</li><li>Baja sensibilidad y fallo notable en I-SPY1; sin demostración de calibración clínica ni de utilidad prospectiva.</li><li>Desbalance, cohortes heterogéneas y subgrupos pequeños. No se ha realizado un estudio suficiente de equidad.</li></ul>
    <h3>Qué faltaría para estudiar utilidad clínica</h3><p>Validación externa y prospectiva, calibración independiente, un uso previsto y costes de errores definidos con especialistas, análisis de sesgos y comparación con alternativas clínicas pertinentes. Este trabajo no es un dispositivo médico.</p>
    <h3>Fuentes y licencia</h3><ol class="small"><li>GUIA.md y enunciado oficial <code>caso_breastdcedl.pdf</code> de la edición docente; aclaración de cortes y utilidades oficiales conservadas.</li><li>Fridman et al. BreastDCEDL, Scientific Data (2026). DOI: <a href="https://doi.org/10.1038/s41597-026-06589-6">10.1038/s41597-026-06589-6</a>. Descriptor del dataset original; no garantiza las métricas de este subconjunto docente.</li><li>Datos derivados de cohortes públicas I-SPY1, I-SPY2 y Duke. La guía exige CC BY-NC 4.0; la LICENSE descargada indica CC BY 4.0. Se conserva la discrepancia y el uso docente no comercial. No se reutilizan figuras adaptadas del artículo con su licencia distinta.</li></ol>
    <p class="small">Código y configuración: <a href="{GITHUB}">{GITHUB}</a>. Todas las métricas y curvas proceden de artefactos ya guardados. El informe, los pesos y los datos de defensa permanecen fuera de Git. Revisión personal y defensa del contenido a cargo del autor.</p>'''))
    css='''*{box-sizing:border-box}body{margin:0;background:#e8edf1;font-family:"Segoe UI",Arial,sans-serif;color:#203e51;font-size:10.2pt;line-height:1.48}.page{width:210mm;height:297mm;padding:14mm 16mm 17mm;background:#fff;position:relative;break-after:page}.page:last-child{break-after:auto}.page header{height:12mm;border-bottom:1px solid #d8e5ef;color:#0760ab;font-size:8pt;display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:5mm}.content{height:246mm}.page footer{position:absolute;bottom:8mm;left:16mm;right:16mm;border-top:1px solid #d8e5ef;padding-top:3mm;display:flex;justify-content:space-between;font-size:7.3pt;color:#637c8e}h1{font-size:29pt;line-height:1.18;letter-spacing:-.8pt;color:#0760ab;margin:5mm 0}h2{font-size:16pt;line-height:1.25;color:#0760ab;margin:0 0 4mm}h3{font-size:11pt;color:#0760ab;margin:5mm 0 2mm}p{margin:2.7mm 0}strong{font-weight:650}.lead{font-size:14pt;line-height:1.4}.eyebrow{font-size:8pt;letter-spacing:1pt;color:#657e90}.small,.caption{font-size:8.5pt;line-height:1.43}.caption{color:#637c8e;margin-top:1mm}.notice,.warning{padding:3.2mm 4mm;background:#edf4fa;border-left:3px solid #0760ab;margin:4mm 0;font-size:9.2pt}.warning{background:#fbf1e8;border-color:#bf733d}.kpis{display:flex;gap:3mm;margin:5mm 0}.kpis>div{flex:1;background:#edf4fa;padding:4mm}.kpis b{display:block;font-size:24pt;color:#0760ab}.kpis span{font-size:8.5pt}table{width:100%;border-collapse:collapse;margin:3mm 0;font-size:9pt;line-height:1.3}th{background:#0760ab;color:#fff;text-align:left;font-weight:600}th,td{padding:2mm 2.6mm;vertical-align:top}tbody tr:nth-child(odd){background:#edf4fa}tbody tr:nth-child(even){background:#fafcfe}td{border-bottom:1px solid #e2ebf3}.two{display:grid;grid-template-columns:1fr 1fr;gap:6mm;align-items:start}.two p{font-size:9.4pt}.phases{display:flex;justify-content:center;gap:7mm;margin:4mm 0 0}.phases figure{margin:0;width:35mm;text-align:center}.phases img{width:35mm;height:35mm;display:block}.phases figcaption{font-size:9pt;font-weight:600;color:#0760ab;margin-top:1mm}.equation{background:#edf4fa;border-radius:4px;padding:3mm 4mm;font-size:11pt;line-height:1.7;text-align:center;margin:4mm 0}.curvas{width:100%;height:187mm;object-fit:contain;display:block}.matriz{width:100%;height:58mm;object-fit:contain}.roc{width:100%;max-height:76mm;object-fit:contain;margin:3mm 0 4mm}.hash{font:8pt Consolas,monospace;overflow-wrap:anywhere;background:#edf4fa;padding:3mm}code{font-size:.9em;background:#edf4fa;padding:0 1mm}a{color:#0760ab;text-decoration:none;overflow-wrap:anywhere}ul,ol{padding-left:5mm}li{margin:2mm 0}@page{size:A4;margin:0}@media print{body{background:#fff}.page{print-color-adjust:exact;-webkit-print-color-adjust:exact}}'''
    cuerpo=''.join(f'<section class="page"><header><b>BREASTDCEDL · INFORME</b><span>ANTONIO AGUILERA / {html.escape(titulo)}</span></header><main class="content">{contenido}</main><footer><span>Proyecto educativo · Sin validez clínica · 30/10/2026</span><span>{i} / {len(paginas)}</span></footer></section>' for i,(titulo,contenido) in enumerate(paginas,1))
    documento='<!doctype html><html lang="es"><head><meta charset="utf-8"><title>Informe BreastDCEDL — Antonio Aguilera</title><style>'+css+'</style></head><body>'+cuerpo+'</body></html>'
    ruta=salida/'INFORME_BREASTDCEDL.html';ruta.write_text(documento,encoding='utf-8')
    with sync_playwright() as pw:
        browser=pw.chromium.launch(executable_path='C:/Program Files/Google/Chrome/Application/chrome.exe',headless=True)
        page=browser.new_page(viewport={'width':1000,'height':1200})
        page.goto(ruta.as_uri(),wait_until='load');page.evaluate('document.fonts.ready')
        problemas=page.evaluate('''() => [...document.querySelectorAll('.content')].flatMap((e,i)=>[...e.children].filter(c=>c.getBoundingClientRect().bottom>e.getBoundingClientRect().bottom+1).map(c=>({pagina:i+1,texto:c.textContent.slice(0,90)})))''')
        if problemas:
            browser.close()
            raise ValueError(f'Contenido desbordado: {problemas}')
        page.pdf(path=str(salida/'INFORME_BREASTDCEDL.pdf'),format='A4',print_background=True,prefer_css_page_size=True,margin={'top':'0','right':'0','bottom':'0','left':'0'})
        for i in (1,5,6,9):
            page.locator('.page').nth(i-1).screenshot(path=str(salida/f'assets/pagina_{i}.png'))
        browser.close()
    assert len(PdfReader(salida/'INFORME_BREASTDCEDL.pdf').pages)==9


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--salida',type=Path,default=REPO/'runs/entrega_final')
    p.add_argument('--regenerar',action='store_true')
    a=p.parse_args();salida=a.salida.resolve()
    if salida.exists() and (not a.regenerar or not (salida/'LEEME_ENTREGA.txt').exists()):
        p.error('La salida existe. Elige otra o usa --regenerar solo para una entrega propia ya generada.')
    salida.mkdir(parents=True,exist_ok=True);(salida/'assets').mkdir(exist_ok=True)
    # Marcador para permitir corregir únicamente esta salida generada, sin borrar directorios.
    (salida/'LEEME_ENTREGA.txt').write_text('Entrega local generada desde artefactos existentes. Ver ENLACES_ENTREGA.txt.\n',encoding='utf-8')
    m=leer(REPO/'artifacts/dashboard/resumen.json')
    empaquetar_pesos(salida,m)
    crear_informe(salida,m)
    presentacion=REPO/'runs/presentacion_uax/PRESENTACION_UAX_5_DIAPOSITIVAS.pdf'
    assert len(PdfReader(presentacion).pages)==5
    shutil.copyfile(presentacion,salida/'PRESENTACION_5_DIAPOSITIVAS.pdf')
    (salida/'ENLACES_ENTREGA.txt').write_text(f'Repositorio:\n{GITHUB}\n\nAplicación:\n{URL}\n\nAutor: Antonio Aguilera\nUso educativo, sin validez clínica.\n',encoding='utf-8')
    (salida/'LEEME_ENTREGA.txt').write_text('''ARCHIVOS PARA LA ENTREGA — ANTONIO AGUILERA

1. Repositorio y URL de aplicación: copiar de ENLACES_ENTREGA.txt.
2. Presentación: PRESENTACION_5_DIAPOSITIVAS.pdf (exactamente 5 páginas).
3. Pesos: PESOS_MODELO_FINAL.zip (modelo final época 5 y configuración).
4. Informe: INFORME_BREASTDCEDL.pdf (9 páginas, métricas, figuras y límites).

Leer el informe y comprobar que puedes explicar sus decisiones antes de enviarlo.
Se declara la consulta del test original previa a explorar variantes; no se afirma
cumplimiento pleno de esa regla. La validación privada sigue sin consultarse.

Si el aula permite un solo archivo, ENTREGA_BREASTDCEDL_ANTONIO.zip reúne los
documentos y el ZIP de pesos. Si tiene casillas separadas, usar los archivos
individuales y los dos enlaces. HTML y assets son fuentes locales del informe:
no hace falta adjuntarlos ni enviar todos los runs o el dataset.

No subir estos pesos/resultados a Git. Revisar que la app esté disponible antes
de la defensa. Mantener también una copia local o en pendrive como respaldo.
''',encoding='utf-8')
    nombres=['INFORME_BREASTDCEDL.pdf','PRESENTACION_5_DIAPOSITIVAS.pdf','PESOS_MODELO_FINAL.zip','ENLACES_ENTREGA.txt','LEEME_ENTREGA.txt']
    (salida/'SHA256_ENTREGA.txt').write_text(''.join(f'{sha((salida/n).read_bytes())}  {n}\n' for n in nombres),encoding='utf-8')
    with zipfile.ZipFile(salida/'ENTREGA_BREASTDCEDL_ANTONIO.zip','w',zipfile.ZIP_DEFLATED) as z:
        for nombre in nombres+['SHA256_ENTREGA.txt']:
            z.write(salida/nombre,nombre)
    print(f'Entrega preparada en {salida}\nInforme: 9 páginas. Presentación: 5. Modelo: época 5; hash verificado.')


if __name__=='__main__':
    main()
