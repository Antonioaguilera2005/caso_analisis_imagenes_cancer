# Entrenamiento preparado para la universidad

## Qué hace el código

`modelo.py` define una CNN propia inicializada al azar. Entrada PRE/EARLY/LATE,
un logit por corte y sigmoid solo al obtener probabilidades. Cuatro bloques:

| Etapa | Salida por corte |
|---|---|
| Entrada | 3 × 256 × 256 |
| Conv 3×3 + BatchNorm + ReLU + MaxPool | 16 × 128 × 128 |
| Conv 3×3 + BatchNorm + ReLU + MaxPool | 32 × 64 × 64 |
| Conv 3×3 + BatchNorm + ReLU + MaxPool | 64 × 32 × 32 |
| Conv 3×3 + BatchNorm + ReLU + MaxPool | 128 × 16 × 16 |
| Promedio espacial global + dropout 0.3 + lineal | 1 logit |

La red tiene 97.809 parámetros entrenables. El pooling global reduce parámetros
respecto a aplanar todo el mapa espacial.
BatchNorm, dropout y weight decay son controles iniciales contra sobreajuste;
su eficacia debe comprobarse con validación. Los volteos horizontales y verticales
se aplican conjuntamente a las tres fases. No hay aumentado de color ni pesos externos.
No se presupone que esta arquitectura sea óptima: es el baseline propio inicial.

`entrenar.py` usa las utilidades originales, el fold dado y los CSV como índice.
Excluye test. Selecciona checkpoint por AUC de validación a nivel paciente,
agregando la media de probabilidades. Después registra también mediana y máximo
con umbral 0.5. El umbral todavía no está optimizado y estas comparaciones no
cierran la receta del modelo final. No hay evaluación de test en este script.

La comparación normal/ponderada reinicia modelo, semilla, optimizador y generador
de datos para cada pérdida. pos_weight es N0/N1 de los cortes de entrenamiento
del fold, siguiendo la utilidad oficial. No se extrae información de validación
para calcular el peso. La pérdida aún trata cada corte como ejemplo: cada paciente
puede contribuir ligeramente distinto por su número de cortes; discutirlo en el informe.

## Pruebas en el portátil

Desde la carpeta del repositorio, sin necesidad de activar PowerShell:

```powershell
.\.venv\Scripts\python.exe explorar.py
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe entrenar.py --dispositivo cpu --prueba --epocas 1 --batch-size 4 --salida runs/prueba_cpu
```

La prueba usa tres pacientes negativos y dos positivos en entrenamiento, y dos
por clase en validación; no mezcla pacientes. Así también se comprueba una pérdida
ponderada con peso distinto de uno. Sus métricas no estiman rendimiento científico.
La exploración genera `reports/exploracion/fases_y_realce.png`, con una paciente
por clase y cohorte de train, y un resumen de intensidades de todos sus cortes.
La selección es ilustrativa, no una muestra representativa para inferir diferencias.

Comprobación realizada en el portátil: cuatro pruebas unitarias superadas,
exploración de seis pacientes generada y una época completa para ambas pérdidas
con cinco pacientes train/cuatro validación. Se comprobó ponderación mayor que
uno, alineación de muestras y reproducción de una probabilidad al recargar los
pesos. Los resultados técnicos están en `runs/prueba_cpu_desbalance` y no son
resultados del trabajo. `requirements-portatil.lock.txt` conserva las versiones
instaladas; es específico de este entorno CPU y no debe instalarse en clase.

## Linux con la AMD de clase

Llevar el código y descargar o copiar `breastdcedl` en disco local. No copiar
`.venv`: es específico del portátil Windows. Conservar el entorno Conda con GPU.
Activarlo por su nombre habitual y, desde la carpeta del repositorio, ejecutar:

```bash
python scripts/diagnosticar_entorno.py
python -m pip check
python -m unittest discover -s tests -v
python entrenar.py --dispositivo cuda --prueba --epocas 1 --batch-size 4 --salida runs/prueba_gpu
```

Primero comprobar que `gpu_compute_ok` es true y que el dispositivo mostrado
corresponde a la GPU esperada. PyTorch ROCm utiliza la interfaz `cuda`, incluso
en AMD; ver [documentación oficial](https://docs.pytorch.org/docs/2.14/notes/hip.html).
La prueba corta GPU sigue pendiente: aquí solo puede comprobarse CPU.
El código usa `torch.load(..., weights_only=True)` y requiere PyTorch 2.x.
Si faltan paquetes de preparación, instalar `requirements.txt` en ese entorno;
este fichero no incluye torch. No instalar la versión CPU del portátil en clase.

Después de superar la prueba, empezar la comparación real:

```bash
python entrenar.py --datos /ruta/local/breastdcedl --dispositivo cuda --fold 0 --perdida ambas --workers 2 --salida runs/fold0_semilla42
```

Batch inicial 16, lr 0.001, Adam, weight decay 0.0001, máximo 40 épocas,
paciencia 8, semilla 42. Si falta memoria, reducir batch y documentar el cambio.
Para otros folds, cambiar `--fold` y `--salida`; la carpeta debe ser nueva.
Guardar el diagnóstico junto con los resultados que se traigan al portátil.

## Archivos de cada experimento

- `config.json`: decisiones, partición, parámetros y hashes de índice/utilidades.
- `entorno.json`: Python, paquetes, backend y dispositivo.
- `particion_train.csv`, `particion_val.csv`: muestras exactas de cada partición.
- `historial.json`: pérdida de entrenamiento, métricas por paciente y tiempo por época.
- `mejor.pt`: pesos y época seleccionada; no es un modelo final cerrado.
- `predicciones_cortes_val.csv`: identificadores y probabilidades alineadas.
- `predicciones_pacientes_*.csv`, `metricas_val.json`: agregaciones y baseline de clase cero.
- `ejecucion.json`: duración, pico de memoria GPU asignada y checksum.
- `comparacion.json`: comparación de pérdidas usando agregación mean y umbral 0.5.

Los JSON guardan null cuando una métrica, como precisión sin positivos predichos,
es indefinida. Los experimentos no sobrescriben una carpeta anterior.
Se fijan semillas y se solicitan operaciones deterministas con avisos cuando no
estén disponibles. No se garantiza igualdad exacta entre CPU/GPU o versiones:
ver [reproducibilidad en PyTorch](https://docs.pytorch.org/docs/2.14/notes/randomness.html).

## Pendiente después del entrenamiento

Examinar sobreajuste, sensibilidad/especificidad, estabilidad entre folds,
agregación, umbral, calibración y cohortes. Cerrar el modelo antes de evaluar
test. Preparar después aplicación web, informe y cinco diapositivas.

## Atribución del dataset

BreastDCEDL, Fridman y colaboradores, DOI 10.1038/s41597-026-06589-6.
El enunciado indica CC BY-NC 4.0, mientras que el LICENSE descargado indica
CC BY 4.0. Se conservan ambos documentos originales. Para este proyecto se
mantiene el uso docente no comercial; aclarar esa discrepancia con la fuente
antes de cualquier reutilización con otra finalidad.
