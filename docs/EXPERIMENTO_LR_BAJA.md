# Última comparación: tasa de aprendizaje menor

Hipótesis: reducir lr de 0,001 a 0,0003 puede estabilizar el entrenamiento. No se presupone mejora ni se buscan curvas suaves como objetivo. Conservar CNN simple solo imagen; mismos cinco folds, semilla 42, batch 16, dropout 0,3, weight decay 0,0001, Adam, máximo 60 épocas y paciencia 12. Comparar ambas pérdidas. Se mantiene todo salvo lr respecto a `cv5_v3_60ep`.

## Universidad

Desde la carpeta del repositorio, con el dataset local:

```bash
git pull
conda activate cancer
python -u scripts/ejecutar_lr_baja.py
```

El script comprueba GPU, muestra el nombre del dispositivo y no sobrescribe una ejecución anterior. No instala paquetes. Usa `cuda` también con ROCm/AMD. Ejecuta diez entrenamientos secuenciales y el análisis automático; conserva checkpoints por época. Para inspeccionar la receta sin ejecutar: `python scripts/ejecutar_lr_baja.py --mostrar`.

Traer `runs/cv5_simple_lr3e4`. No subir resultados ni pesos a Git. Si el tamaño impide copiar todo, para revisión inicial traer `analisis` y los JSON, CSV y logs de cada fold/pérdida, conservando todos los pesos originales y copiando al menos los seleccionados. No borrar los checkpoints en el equipo de origen.

## Decisión posterior

Al volver al portátil, desde el repositorio: `python scripts/comparar_lr.py`. Requiere los análisis completos de ambas ejecuciones y guarda CSV y PNG en `runs/comparacion_lr_baja`, sin inferencia adicional. Después revisar las curvas individuales antes de decidir.

Comparar con `cv5_v3_60ep`, no con test: AUC media ± desviación, diferencias por fold, PR-AUC/AP, pérdidas y brecha en la época seleccionada, calibración y sensibilidad/especificidad. Menor lr puede aprender más despacio; no interpretar mayor época seleccionada como mejor generalización por sí sola. No prolongar ni cambiar hiperparámetros solo para mejorar la apariencia de las curvas. Registrar si la paciencia limita el aprendizaje.

Si no hay mejora consistente, conservar la CNN simple original y documentar los experimentos negativos. Las comparaciones repetidas sobre los mismos folds tienen optimismo de selección; el test histórico ya se consultó y debe declararse. No se vuelve a consultar test en este experimento.

En paralelo quedan pendientes la aplicación con el mismo pipeline, su validación, la memoria y las cinco diapositivas. La preparación de la entrega debe continuar sin esperar otra arquitectura. La variante final se cerrará después de comparar esta ejecución; no se han cambiado pesos ni umbral del modelo archivado.
