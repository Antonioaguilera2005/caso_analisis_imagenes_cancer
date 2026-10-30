# Archivos preparados para entregar

La carpeta local `runs/entrega_final` contiene:

| Entregable | Archivo o enlace |
|---|---|
| Repositorio | https://github.com/Antonioaguilera2005/caso_analisis_imagenes_cancer |
| Aplicación publicada | https://breastdce-antonio.onrender.com |
| Presentación | `PRESENTACION_5_DIAPOSITIVAS.pdf`, exactamente cinco páginas, plantilla UAX |
| Pesos | `PESOS_MODELO_FINAL.zip`, checkpoint original de época 5 con receta, arquitectura y verificador |
| Informe | `INFORME_BREASTDCEDL.pdf`, nueve páginas con decisiones, métricas, figuras y limitaciones |

Si el aula solo permite adjuntar un archivo, `ENTREGA_BREASTDCEDL_ANTONIO.zip` reúne los tres entregables, los enlaces y el manifiesto de hashes; ocupa aproximadamente 2 MB. Si tiene campos separados, usar los archivos individuales y copiar los enlaces. No entregar el HTML auxiliar, las carpetas de todos los entrenamientos ni el dataset.

El modelo original está en `runs/modelo_final/checkpoints/epoca_005.pt`. No se ha vuelto a entrenar ni serializado de otra manera: el ZIP conserva los mismos bytes. Si el aula exige expresamente un archivo `.pt`, extraer `modelo_final/checkpoints/epoca_005.pt` del ZIP. Los pesos de las otras épocas se conservan como respaldo local.

SHA-256 del checkpoint: `56f8f2cd767f01e160f03641f9adb3a7c32c7dcf70bec9d410d169b0a235dc73`.

El paquete se ha descomprimido y probado: `comprobar_pesos.py` carga el `state_dict` completo, comprueba arquitectura, receta, época y hash y confirma 97.809 parámetros. No entrena ni evalúa ningún dataset. El ZIP incluye instrucciones para usarlo con el repositorio.

## Informe y revisión antes de enviar

El informe cubre datos y separación por paciente, normalización, arquitectura, hiperparámetros, comparación de cinco experimentos con ambas pérdidas, curvas completas de cinco folds, agregación y umbral, matriz de confusión, métricas por clase, baseline, cohortes, calibración, aplicación y fuentes.

Conserva explícitamente la salvedad de que el test original se consultó antes de explorar variantes. No se debe afirmar cumplimiento metodológico pleno ni ocultar esa secuencia. La prueba privada del profesor no se ha consultado. Leer el informe, confirmar que se pueden explicar sus decisiones y mantener la atribución y el aviso educativo.

Para regenerar desde los artefactos locales ya guardados:

```powershell
.\.venv\Scripts\python.exe scripts/preparar_entrega_final.py --regenerar
```

El generador requiere pandas, matplotlib, pypdf, Playwright y Chrome instalado. Comprueba que el informe no tenga textos desbordados y verifica los números de páginas y los hashes. No realiza inferencia ni vuelve a evaluar test. Los documentos y pesos generados se mantienen en `runs/`, fuera de Git; solo se versionan el generador y esta documentación.
