# Experimento de doble convolución, solo imagen

Comparación con `cv5_v3_60ep`: se cambia exclusivamente la arquitectura. No se añade clínica ni se modifica el modelo histórico (`modelo.py`). La variante se define en `modelo_doble.py` y se identifica en cada config y log.

## Arquitectura

Cuatro bloques con canales 16, 32, 64 y 128. Cada bloque:

```text
Conv 3×3 → BatchNorm → ReLU → Conv 3×3 → BatchNorm → ReLU → MaxPool 2×2
```

Entrada: PRE/EARLY/LATE de 256×256, float32 /255. Salidas espaciales por bloque: 128×128, 64×64, 32×32 y 16×16. Cabeza: global average pooling → dropout 0,3 → capa lineal 128→1. Ocho convoluciones y 294.129 parámetros (simple: 97.809). Inicialización aleatoria Kaiming/Xavier; sin preentrenamiento. Se conserva la media de probabilidades de cortes por paciente.

No se esperan curvas suaves ni mejoras por definición. La mayor capacidad puede aumentar el sobreajuste. Evaluar si aumenta la AUC de validación consistentemente, además de observar la pérdida y la brecha train/validación. La época de máxima AUC no es necesariamente la de mínima BCE.

## Entrenamiento GPU AMD en la universidad

Desde la carpeta del repositorio:

```bash
git pull
conda activate cancer
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'Sin GPU')"
python -u validacion_cruzada.py --arquitectura doble --datos breastdcedl --salida runs/cv5_doble_v1 --dispositivo cuda --epocas 60 --paciencia 12 --batch-size 16 --workers 2 --semilla 42 --lr 0.001 --dropout 0.3 --weight-decay 0.0001
```

La cabecera debe indicar `arquitectura=doble`, `solo imagenes` y 294.129 parámetros. Son diez entrenamientos: cinco folds, pérdidas normal y ponderada. Máximo 60 épocas, paciencia 12 por máxima AUC de validación por paciente. Checkpoint en cada época, log, entorno, índices y checksums guardados. Generación de PNG y métricas automática al terminar, en `runs/cv5_doble_v1/analisis`.

Se mantiene batch 16 para conservar la comparación; la memoria y duración pueden aumentar. Si aparece falta de memoria, guardar el error antes de modificar el batch: cambiarlo constituye una receta distinta y debe registrarse para todos los folds.

No pasar `--clinica`: el programa rechaza mezclar ambos cambios. No ejecutar `finalizar.py` con esta variante: el cierre histórico se restringe a la CNN simple. No volver a consultar test para seleccionar arquitectura, épocas o umbral; el test histórico ya fue consultado y hay que declararlo.

Conservar los resultados y pesos fuera de Git. Traer la carpeta completa. Si el volumen impide copiarla, para análisis inicial copiar `analisis` y todos los JSON/CSV/log de cada fold/pérdida; conservar los checkpoints originales y traer al menos el seleccionado para verificar checksum.

## Comparación posterior

Usar los mismos folds oficiales por paciente y comparar AUC media ± desviación, diferencias por fold, PR-AUC/AP, brecha train/validación en la época seleccionada, calibración y métricas por clase. Los umbrales OOF exploratorios son optimistas; no se interpretan como evaluación independiente. No afirmar significación a partir de una diferencia pequeña ni suavizar los datos para aparentar estabilidad.
