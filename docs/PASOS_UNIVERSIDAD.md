# Pasos en el ordenador de la universidad (Linux)

## 1. Abrir una terminal y activar el entorno GPU existente

```bash
conda env list
```

Busca el nombre del entorno donde ya configuraste PyTorch y la GPU. Actívalo
sustituyendo `NOMBRE_DEL_ENTORNO` por ese nombre:

```bash
conda activate NOMBRE_DEL_ENTORNO
python --version
```

No crear otro entorno ni instalar el PyTorch CPU del portátil.

## 2. Descargar el código en disco local

Elige una carpeta local con espacio para las imágenes. Por ejemplo:

```bash
mkdir -p ~/proyectos
cd ~/proyectos
git clone https://github.com/Antonioaguilera2005/caso_analisis_imagenes_cancer.git
cd caso_analisis_imagenes_cancer
```

Si ya tienes el repositorio clonado, entra en su carpeta y ejecuta
`git pull --ff-only`. Si Git avisa de cambios locales, conservarlos y resolver
el aviso antes de continuar. No usar force ni reset para descartarlos.

## 3. Seleccionar el intérprete de clase en VS Code

Abre la carpeta del repositorio. Con Ctrl+Shift+P, elige
**Python: Select Interpreter** y selecciona el Python del entorno Conda activado.
Puedes consultar su ruta con `which python`.
La configuración `.venv/Scripts/python.exe` del repositorio es del portátil
Windows; no utilizarla en Linux. Tampoco copiar la carpeta `.venv` a clase.

## 4. Comprobar dependencias y GPU

```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('HIP:', torch.version.hip); print('GPU disponible:', torch.cuda.is_available())"
python -m pip install -r requirements.txt
python -m pip check
python scripts/diagnosticar_entorno.py
```

`requirements.txt` instala las herramientas de datos sin incluir torch.
No ejecutar `requirements-cpu.txt`, `requirements-portatil.lock.txt` ni
`conda env create -f environment.yml` sobre la configuración GPU de clase.

El diagnóstico guarda `diagnostico_entorno.json`. Comprobar que muestra la
GPU AMD esperada y `gpu_compute_ok: true`. El código de entrenamiento requiere
PyTorch 2.x. Si falta torch, la GPU no se detecta, la operación falla o aparece
alguna incompatibilidad, guardar el mensaje y resolverla antes del entrenamiento.
No reinstalar PyTorch automáticamente.

## 5. Descargar las imágenes y verificar los datos

Los metadatos están en GitHub; las 38.109 imágenes no. Desde el repositorio:

```bash
python descargar_datos.py --destino breastdcedl
python scripts/auditar_datos.py --datos breastdcedl --comprobar-imagenes
```

También puedes copiar desde el portátil la carpeta `breastdcedl/dataset` a
`breastdcedl/dataset` en clase y ejecutar la auditoría. Usa disco local para
el entrenamiento. Si la descarga se interrumpe, repetir el descargador para
reanudarla. El resumen esperado es 1.097 pacientes train, 176 test y cero
imágenes ausentes o vacías.

## 6. Comprobar el código y hacer una prueba corta en GPU

```bash
python -m unittest discover -s tests -v
python entrenar.py --dispositivo cuda --prueba --epocas 1 --batch-size 4 --salida runs/prueba_gpu
```

En AMD/ROCm también se usa `cuda` como nombre del dispositivo en PyTorch.
Esta prueba utiliza cinco pacientes train y cuatro de validación. Las métricas
son técnicas, no resultados válidos del proyecto. Debe terminar para ambas
pérdidas y crear `runs/prueba_gpu/comparacion.json`.

Cada salida tiene que ser una carpeta nueva: si repites la prueba, usa
`runs/prueba_gpu_2`. Si falta memoria, reduce `--batch-size`; si fallan los
trabajadores, conserva `--workers 0`.

## 7. Ejecutar la validación cruzada completa

```bash
python validacion_cruzada.py --datos breastdcedl --dispositivo cuda --batch-size 16 --workers 2 --salida runs/cv5_v2
```

Mantener el equipo encendido y la terminal abierta. Ejecuta los cinco folds y
las dos pérdidas secuencialmente: diez entrenamientos. Cada uno usa máximo
40 épocas y parada después de ocho épocas sin mejora de AUC por paciente.
Guarda un checkpoint por época y señala el elegido en `seleccion.json`.
Registra train/validación juntos y genera gráficas al terminar. No evalúa test.
Aunque ya se haya realizado el fold 0 antiguo, repetirlo con esta versión para
registrar las curvas faltantes y conservar todos sus checkpoints.
Si cambias batch o workers, registra los cambios y usa una carpeta nueva.

## 8. Traer los resultados al portátil

Copiar en un USB o almacenamiento personal:

- `diagnostico_entorno.json`.
- La carpeta completa `runs/cv5_v2`, incluidos todos los `checkpoints` y el análisis.

No borrar los resultados del equipo de clase hasta comprobar la copia.
`runs/` está ignorado por Git: un `git push` no transporta los pesos ni métricas.
En el portátil, colocar la carpeta en `runs/` del repositorio.
Revisaremos historial, sensibilidad, especificidad, AUC y matriz de confusión
antes de decidir ajustes o cerrar el modelo. No usar test para ajustar.

Más detalles: [ENTRENAMIENTO.md](ENTRENAMIENTO.md).

## 9. Después de revisar la CV en el portátil

Si se decide reentrenar una única CNN con todo train, seguir
[MODELO_FINAL.md](MODELO_FINAL.md). Primero se cierra la receta con CV,
después se reentrena en la GPU con épocas fijas y se trae la carpeta completa
`runs/modelo_final`. Test se evalúa una sola vez después de esa etapa, sin
modificar el modelo ni el umbral a partir de su resultado.
