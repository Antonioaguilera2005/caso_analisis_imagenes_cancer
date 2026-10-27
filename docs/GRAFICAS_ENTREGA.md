# Gráficas de pérdidas y AUC

Desde el repositorio, con los resultados locales:

```bash
python scripts/graficas_presentacion.py --runs runs/cv5_v3_60ep runs/cv5_clinica_v1 runs/cv5_doble_v1 runs/cv5_simple_lr3e4 --salida runs/graficas_entrega
```

La salida debe ser nueva. Se leen historiales y selecciones; no se entrena, no se recalculan predicciones ni se consulta test. El script no sobrescribe las figuras originales.

Por experimento se exportan cinco PNG:

- `perdidas_normal.png` y `perdidas_ponderada.png`: BCE sin ponderar por corte, train y validación sin aumentos. Panel izquierdo con origen cero y escala global común; derecho con escala ampliada compartida entre los cinco folds de esa figura. La escala ampliada puede cambiar entre experimentos, por lo que para compararlos se usa el panel izquierdo.
- `objetivo_ponderado.png`: BCE ponderada evaluada sin aumentos, separada de la anterior. No confundir con la pérdida online con aumentos.
- `auc_normal.png` y `auc_ponderada.png`: AUC por paciente, eje 0–1, sin suavizado.

Todos los puntos y épocas originales permanecen visibles. La línea discontinua marca la época seleccionada por AUC; la zona gris muestra entrenamiento posterior descartado. Las épocas anteriores también son candidatas, no modelos independientes. `epocas_y_brechas.csv` documenta la AUC train/validación en selección.

Usar estas figuras para explicar los resultados, sin afirmar que una mejor presentación mejora el modelo. Los picos y brechas se conservan. El modelo reentrenado con todo train no tiene validación: no construirle una curva de validación ficticia ni mezclar su historial con folds de CV.

Las imágenes y CSV se guardan en `runs/` y no se suben a Git. Solo el generador y esta documentación se publican.
