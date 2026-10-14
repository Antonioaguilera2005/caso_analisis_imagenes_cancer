# Plan de desarrollo y experimentos

Estado: pipeline implementado; sin entrenamiento completo ni resultados finales. Este plan se basa en
GUIA.md y en el PDF oficial, leído durante la descarga del dataset.

## Reparto entre equipos

Portátil: código, auditoría, exploración, pruebas pequeñas, informe y aplicación.
Universidad: Linux, PyTorch, AMD Radeon RX 6700 XT y entorno Conda ya configurado.
Conservar ese entorno. Ejecutar el diagnóstico antes del primer entrenamiento;
la presencia de la tarjeta no demuestra que PyTorch pueda utilizarla.
Guardar versión de Python, PyTorch, HIP y resultado de una operación real.

## Datos y objetivo

Predecir pCR binaria tras tratamiento usando imágenes anteriores al tratamiento.
Una observación de entrada es un corte con canales PRE, EARLY, LATE, en ese orden,
en escala de grises y dividido por 255. La salida final es por paciente.
No normalizar como ImageNet. Aplicar aumentos geométricos conjuntamente a las fases.
Mantener utils_caso.py original. Usar sus folds y funciones de referencia.

## Primera comparación

CNN propia de cuatro bloques convolucionales, pooling global, dropout y un logit.
Inicialización aleatoria; sin arquitecturas ni pesos preentrenados.
Propuesta inicial, pendiente de probar memoria y velocidad: Adam, lr 0.001,
batch 16, máximo 40 épocas, semilla 42, parada tras 8 épocas sin mejora en
AUC de validación por paciente. Guardar el mejor checkpoint según ese criterio.

Comparar BCE normal y BCE ponderada con la misma arquitectura, fold, semilla,
presupuesto y aumentos. Calcular pos_weight exclusivamente con entrenamiento
de cada fold, siguiendo utils_caso.py. Registrar la fórmula y los recuentos.
Comenzar con fold 0; repetir la comparación en los cinco folds si el tiempo permite.
La prueba con cinco pacientes solo verifica funcionamiento, no calidad del modelo.

## Validación y cierre

Generar probabilidades de corte sin shuffle y conservar sample_id y patient_id.
Agregar por paciente con media como referencia. Comparar mediana y máximo con
validación interna. Usar umbral 0.5 como referencia y justificar cualquier ajuste
usando exclusivamente validación. Registrar accuracy, sensibilidad, especificidad,
ROC-AUC y matriz de confusión; comparar con predecir siempre pCR=0.
Las métricas usadas para seleccionar modelos son internas y pueden ser optimistas.

Cerrar arquitectura, pérdida, agregación, umbral y receta del modelo final antes
de acceder a test para evaluación. Documentar si se reentrena con todo train y
cómo se fija el número de épocas. Evaluar test una vez y no reajustar después.
Analizar resultados por cohorte y limitaciones sin convertir test en validación.

## Entregables posteriores al entrenamiento

Pesos, configuración, historial, versiones, predicciones y métricas reproducibles.
Informe y cinco diapositivas. Aplicación accesible por URL con el mismo
preprocesado: entrada de tres fases por corte y varios cortes para paciente,
probabilidad, umbral, checksum, tiempo y dispositivo. Validar formato y tamaño.
Indicar cuándo solo se aporta un corte y no hay agregación de múltiples cortes.
Mostrar aviso de uso educativo sin validez clínica. Conservar licencia y atribución.

## Pendientes para empezar a entrenar

Revisar el artículo científico complementario. El enunciado oficial ya está leído.
El portátil usa Python 3.11 con .venv y dependencias de preparación instaladas.
La descarga está completa y ha pasado la auditoría de metadatos y presencia de
imágenes. Se han explorado seis pacientes de train y comprobado el pipeline en CPU.
Realizar diagnóstico y prueba corta en GPU de clase antes del entrenamiento completo.
