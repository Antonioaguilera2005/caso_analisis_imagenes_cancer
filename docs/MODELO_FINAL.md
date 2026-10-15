# De la validación cruzada al test final

## Orden de trabajo

1. Verificar los diez entrenamientos: cinco folds por cada pérdida.
2. Cerrar la receta usando únicamente validación interna.
3. Entrenar una única CNN desde cero con todo train, en la GPU de clase.
4. Traer la carpeta del modelo final al portátil.
5. Evaluar test una sola vez, sin cambiar después modelo ni umbral.
6. Construir la aplicación con el mismo cargador de pesos y preprocesado.

Las métricas de validación cruzada son media ± desviación muestral entre folds.
No confundirlas con la AUC de todas las predicciones OOF combinadas.
La selección de pérdida sigue la AUC media; diferencias pequeñas no demuestran
superioridad estadística. El umbral maximiza F1 macro en OOF como criterio
educativo de igual peso entre clases, sin pretender ser un coste clínico validado.

## Generar la receta en clase

Activar el entorno GPU existente y entrar en el repositorio:

```bash
conda activate cancer
git pull --ff-only
python finalizar.py preparar --cv runs/cv5_v2 --salida runs/cierre_final
```

Si cv5_v2 está en la raíz, usar `--cv cv5_v2`. El script verifica particiones,
etiquetas, predicciones, hashes de pesos y cobertura completa de las pacientes
de train. Genera `receta.json`, `receta.sha256`, tabla de comparación y PNG.
Elegir otra salida si esa carpeta ya existe; no borrar resultados originales.

La receta fija arquitectura, pérdida, optimizador, lr, regularización, batch,
semilla, agregación, normalización y umbral. Fija las épocas como mediana de
las épocas elegidas en los cinco folds de la pérdida seleccionada.
No contiene ninguna decisión basada en test.

También se puede transportar `runs/cierre_final` desde el portátil, conservando
`receta.json` y su checksum juntos. La receta contiene resultados de CV: no va a Git.

## Reentrenar una sola CNN con todo train

```bash
python finalizar.py entrenar --receta runs/cierre_final/receta.json --datos breastdcedl --dispositivo cuda --workers 2 --salida runs/modelo_final
```

Se usan todas las pacientes de train. No hay validación ni early stopping en
este reentrenamiento: se ejecuta el número de épocas ya fijado por CV. Se guarda
cada época y `artefacto.json` apunta a la última, sin elegir por test.
Las curvas de validación proceden de los folds; las curvas de esta etapa son
solo train y no deben presentarse como rendimiento de generalización.

Traer al portátil **toda** `runs/modelo_final`, con pesos, receta, artefacto,
historial, diagnóstico de entorno y gráfico. No copiar solo un .pt.
Conservar también `runs/cierre_final` y la CV original. El modelo reentrenado
puede producir una distribución de probabilidades distinta de OOF: es una
limitación de transferir el umbral. No resolverla ajustando con test.

Para comprobar el código sin generar un modelo evaluable:

```bash
python finalizar.py entrenar --receta runs/cierre_final/receta.json --dispositivo cpu --workers 0 --prueba --salida runs/prueba_final
```

La prueba usa cinco pacientes y hasta dos épocas, se marca como técnica y no
puede evaluarse en test. En el portátil no lanzar el entrenamiento completo
en lugar de la ejecución GPU acordada.

## Evaluación única de test, después de traer el modelo

```bash
python finalizar.py evaluar --modelo runs/modelo_final --datos breastdcedl --salida runs/test_final
```

Se comprueba que se trata de la última época fijada y de un modelo entrenado
con todo train, con el código de arquitectura y el índice esperados. Se aplica
`eval()`, sin gradientes, media de cortes y umbral congelado.

Se generan predicciones por corte y paciente, matriz de confusión, métricas por
clase, F1 macro, AUC con IC95% bootstrap de pacientes, calibración, Brier,
precision–recall y comparación con el baseline mayoritario. También se guardan
métricas por cohorte. No se busca un nuevo umbral en test.

El archivo `runs/test_una_vez.json` registra el inicio y la finalización para
prevenir repeticiones accidentales. Si una ejecución falla, revisar el error y
los archivos conservados antes de cualquier recuperación; no borrar ese registro
para probar configuraciones. Los resultados guardados se consultan sin repetir
inferencia. Este mecanismo no sustituye la disciplina metodológica.

## Inferencia compartida con la futura aplicación

`finalizar.cargar_modelo_final` verifica el checksum, valida la receta, carga
los pesos con `weights_only=True` y devuelve el modelo en modo evaluación.
El entrenamiento y la evaluación usan la carga oficial de PNG en el orden
PRE/EARLY/LATE y división por 255. La aplicación reutilizará ese comportamiento.
No tiene validez clínica ni permite decisiones de tratamiento.

## Git y transporte

El código y esta guía están en GitHub. `runs/`, `cv5_v2/`, resultados, recetas
con métricas y pesos permanecen locales. Transportarlos por USB, correo o
almacenamiento personal, comprobando que se pueden abrir antes de borrar la copia.
