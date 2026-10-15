# Auditoría del entrenamiento y de los resultados

## Lo que se comprueba antes de test

`auditar_modelo_final.py` verifica los cinco checkpoints: pesos finitos,
arquitectura, receta, pacientes, lr, weight decay y pasos de Adam por época.
Contrasta la partición con todo train y la aritmética de las matrices y métricas
del historial. Recalcula por defecto la última época sobre todos los cortes de
train en CPU y la compara con el historial GPU: BCE, ROC-AUC, PR-AUC, AP, F1
macro y matriz de confusión. La tolerancia declarada contempla diferencias
numéricas pequeñas entre CPU y GPU.

```bash
python auditar_modelo_final.py --modelo runs/modelo_final --salida runs/auditoria_final
```

La opción `--todas-las-epocas` recalcula también las épocas intermedias, con un
coste considerablemente mayor. Sin ella, los resultados identifican la última
época como la única con inferencia completa recalculada. No afirmar que se han
recalculado los AUC intermedios si solo se comprobó su historial e integridad.

Todos los cálculos de esta auditoría usan train. No sirven para escoger otra
época o umbral ni para demostrar generalización. El historial original y los
pesos originales permanecen intactos. Los CSV recalculados y la auditoría
se guardan junto con una figura de pérdidas con escala amplia y detalle.

La pérdida online incluye aumentos y dropout; la BCE de evaluación no. Una
subida puntual de BCE no implica por sí sola sobreajuste. El número de épocas
de otros trabajos tampoco es comparable sin conocer su arquitectura, datos,
criterio de selección, tamaño de lote y dispositivo.

## Umbral del historial y umbral final

El historial de train utiliza 0.5 como referencia. La receta final conserva el
umbral elegido en validación. La auditoría muestra ambos umbrales explícitamente:
no mezclar sus sensibilidades, matrices o valores de F1. El AUC no depende del
umbral. AP y PR-AUC trapezoidal se etiquetan separadamente.

## Evaluación final e informe

Después de superar la auditoría y con la receta cerrada:

```bash
python finalizar.py evaluar --modelo runs/modelo_final --salida runs/test_final
python resumir_evaluacion.py --cv cv5_v2 --salida runs/informe_final
```

El primer comando genera predicciones de test una única vez, con el modelo y
umbral ya fijados. El segundo lee esas predicciones guardadas y verifica:
cobertura de pacientes, etiquetas, media de cortes y concordancia de las métricas
con scikit-learn. Genera gráficos de train/CV/test, comparación con el baseline,
distribuciones, ejemplos fallados e informe. No vuelve a inferir sobre test.

En la figura conjunta, CV se refiere a cinco modelos de fold, mientras que train
y test pertenecen al único modelo final. La desviación entre folds y el IC95%
de test tienen significados distintos y se etiquetan como tales.
Se muestran tanto los mejores checkpoints por fold como la misma época fija
en todos los folds. La AUC de los primeros no se debe atribuir automáticamente
al reentrenamiento con la mediana de épocas. Ambas estimaciones también están
condicionadas por las decisiones ya tomadas con validación interna.

Un test completado se consulta a partir de sus archivos, sin probar nuevas
épocas, umbrales o pérdidas en ese conjunto. La capacidad predictiva puede ser
modesta aunque el entrenamiento y las métricas sean técnicamente correctos.
Todas las cifras deben presentarse con sus limitaciones y sin validez clínica.

Código e instrucciones pueden ir a GitHub; pesos, métricas y figuras permanecen
en runs/. No sobrescribir la evidencia original para cambiar su presentación.
