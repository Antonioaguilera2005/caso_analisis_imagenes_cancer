# Presentación visual: cinco diapositivas

El generador `scripts/presentacion_visual.py` utiliza el paquete local y los resultados ya calculados. Requiere Matplotlib, pandas, Playwright, python-pptx y Chrome instalado en Windows. No entrena, no realiza inferencia y no consulta test. Produce datos y figuras únicamente en `runs/`, fuera de Git.

```powershell
.\.venv\Scripts\python.exe scripts/presentacion_visual.py --salida runs/presentacion_defensa
```

Para regenerar una presentación propia creada por el script, añadir `--regenerar`. No sobrescribe el borrador anterior de cinco páginas de texto.

Entregables generados:

- PDF de exactamente cinco diapositivas panorámicas.
- PPTX con cinco imágenes de alta resolución y notas del ponente. El contenido de las imágenes no es texto editable dentro de PowerPoint.
- `PRESENTACION.html`: fuente visual editable; las ecuaciones son SVG de Matplotlib y los gráficos se basan en los historiales reales.
- PNG de cada diapositiva y `GUION_DEFENSA.md` para preparar la exposición.
- Verificación automática de desbordes; completar siempre con inspección visual.

La arquitectura muestra forma de tensores, Conv/BatchNorm/ReLU/MaxPool, parámetros por bloque, GAP, dropout, logit, sigmoid y media de probabilidades por paciente. La pérdida normal/ponderada y la selección de época y umbral se explican aparte. La tabla distingue CV original de la repetición con horizonte más largo. No se afirma que la receta seleccionada tenga el mayor valor en todas las columnas.

La diapositiva final conserva las limitaciones, el test histórico y el aviso docente. La URL de Render no se inventa: se debe incorporar tras desplegar y verificar. El autor debe revisar el contenido y preparar su defensa matemática.
