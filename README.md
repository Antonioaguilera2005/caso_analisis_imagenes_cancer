# Predicción de pCR con BreastDCEDL

Proyecto docente de predicción de respuesta patológica completa a partir de
resonancias previas al tratamiento. Cada corte tiene fases PRE, EARLY y LATE;
la predicción final se obtiene por paciente.

## Estado y documentación

Exploración y pipeline de entrenamiento implementados. No hay entrenamiento
completo ni métricas finales todavía. Las pruebas técnicas usan pocos pacientes.
Consultar [la guía](docs/GUIA.md) y el
[enunciado oficial](docs/caso_breastdcedl.pdf), revisado durante la preparación.

## Portátil

Desde la carpeta del repositorio, en PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe descargar_datos.py --pacientes 5
```

En VS Code, abrir esta carpeta y seleccionar `.venv\Scripts\python.exe`
con **Python: Select Interpreter**. Se puede activar con
`.\.venv\Scripts\Activate.ps1`; si PowerShell bloquea la activación, usar
directamente el ejecutable como arriba. No hace falta cambiar políticas del sistema.
Para probar el modelo en CPU, instalar PyTorch aparte:

```powershell
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```
`environment.yml` queda como alternativa Conda; el portátil usa `.venv`.

El descargador crea `breastdcedl/` con metadatos, utilidades e imágenes.
Cinco pacientes sirven para explorar; para la descarga completa omitir
`--pacientes 5`. No modificar `utils_caso.py`. Los datos y pesos no se suben a Git.

Auditar metadatos y comprobar la descarga:

```sh
python scripts/auditar_datos.py --datos breastdcedl
python scripts/auditar_datos.py --datos breastdcedl --comprobar-imagenes
```

La primera orden comprueba etiquetas, identificadores y separación por paciente.
La segunda exige todas las imágenes indexadas; fallará con una descarga de solo
cinco pacientes. El resumen se guarda en `reports/auditoria.json`. No valida
el contenido de los PNG ni mide rendimiento predictivo.

## Equipo de clase

Equipo confirmado: Linux con AMD Radeon RX 6700 XT y PyTorch.

Seguir [los pasos para la universidad](docs/PASOS_UNIVERSIDAD.md), desde la
activación del entorno GPU hasta la copia de resultados al portátil.

Obtener el proyecto en el ordenador de clase:

```sh
git clone https://github.com/Antonioaguilera2005/caso_analisis_imagenes_cancer.git
cd caso_analisis_imagenes_cancer
```

Si ya está clonado, usar `git pull --ff-only` desde esa carpeta.
Activar el entorno Conda GPU habitual. Las imágenes no están en GitHub:
copiar el dataset a disco local o descargarlo con `python descargar_datos.py`.
El repositorio incluye los metadatos y utilidades oficiales para ejecutar el código.
La configuración de VS Code de `.venv` corresponde al portátil: en Linux
seleccionar el intérprete del entorno Conda existente.

Activar el entorno Conda que ya tiene la GPU configurada y ejecutar:

```sh
python scripts/diagnosticar_entorno.py
```

Este diagnóstico comprueba PyTorch y una operación en GPU. Si la configuración
usa otro framework, habrá que comprobarlo por separado. No recrear el entorno
de clase ni reinstalar PyTorch hasta conocer su sistema y backend. Guardar el
resultado para adaptar el entrenamiento. Los datos de entrenamiento deben
estar en disco local del equipo de clase.

## Plan de trabajo

Consultar [las instrucciones de entrenamiento](docs/ENTRENAMIENTO.md) y
[el plan de experimentos](docs/PLAN_EXPERIMENTOS.md) para las órdenes de pruebas,
entrenamiento en clase, comparación de pérdidas y criterios de selección.

La comparación completa se ejecuta con `validacion_cruzada.py`; el análisis
genera PNG con `analizar_resultados.py`. Se guarda un checkpoint por época y
se señala el elegido en `seleccion.json`. Los resultados y sus gráficas se
conservan localmente en `runs/` y se transportan fuera de GitHub.

Tras completar CV, seguir [la guía del modelo final](docs/MODELO_FINAL.md):
`finalizar.py preparar` cierra la receta, `entrenar` reentrena una única CNN con
todo train y `evaluar` realiza la evaluación única de test con umbral fijo.
La [guía de auditoría](docs/AUDITORIA_FINAL.md) explica cómo contrastar pesos,
historial y métricas, y generar informes desde las predicciones guardadas.

1. Revisar el enunciado y explorar imágenes y metadatos.
2. Validar etiquetas y separación por paciente con los folds dados.
3. Implementar una CNN propia desde cero, con un logit de salida.
4. Comparar BCE normal y ponderada con igual partición y semilla.
5. Elegir agregación de cortes y umbral usando validación interna.
6. Registrar AUC, sensibilidad, especificidad, accuracy y matriz de confusión por paciente.
7. Cerrar el modelo antes de evaluar test y analizar sesgos entre cohortes.
8. Preparar aplicación web, informe y cinco diapositivas.

Mantener alineadas las tres fases al aplicar transformaciones. No usar test
para elegir hiperparámetros ni modelos preentrenados.

## Git y licencia

El remoto determina el repositorio de destino. El nombre y correo locales
determinan la autoría; la autenticación para subir cambios es independiente.
Configurar la identidad de la universidad solo en este repositorio cuando se
confirme su correo.

Datos BreastDCEDL: CC BY-NC 4.0. Conservar atribución y licencia suministradas
por el descargador. Uso educativo, sin validez clínica.
