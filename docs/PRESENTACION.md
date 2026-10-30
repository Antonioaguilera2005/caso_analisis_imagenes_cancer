# Cinco diapositivas en la plantilla UAX

La presentación local está en `runs/presentacion_uax/PRESENTACION_UAX_5_DIAPOSITIVAS.pptx`, con un PDF del mismo nombre exportado desde PowerPoint. Son **cinco diapositivas**, sin portada adicional ni página de agradecimientos. Se conserva la portada, logotipos y layouts de `PPT_Template (3).pptx`; el original de Descargas no se modifica.

1. Problema, definición de pCR y objetivo con PRE/EARLY/LATE.
2. Preparación, clases, cohortes y separación por paciente en cinco folds.
3. CNN propia: tamaños, 97.809 parámetros y ecuaciones de los bloques, pooling, sigmoid y media por paciente.
4. Comparación de variantes, curva real de sobreajuste, hiperparámetros y selección de cinco épocas/umbral 0,33.
5. Aplicación publicada, demostración, matriz de confusión histórica y limitaciones, incluida la cronología del test.

Los textos, tablas y diagramas se pueden editar en PowerPoint; las fórmulas, gráficas y la captura de la aplicación son imágenes. Cada diapositiva incluye notas para ensayar el razonamiento. No son capturas de código. El estudiante debe revisar el contenido y poder justificarlo personalmente.

Para regenerar con los artefactos locales disponibles y la plantilla original:

```powershell
.\.venv\Scripts\python.exe -m pip install python-pptx
.\.venv\Scripts\python.exe scripts/presentacion_plantilla.py --plantilla "C:\Users\anton\Downloads\PPT_Template (3).pptx" --regenerar
```

La captura de la aplicación procede de `scripts/probar_dashboard_navegador.py`. Las cifras se leen de los resultados ya guardados; no se entrena ni se consulta test. Para otra plantilla o para conservar una versión anterior, usar `--salida runs/otra_presentacion` sin `--regenerar`. Exportar el PDF con PowerPoint y comprobar que tiene cinco páginas.

La plantilla, los pesos, las imágenes y la presentación generada permanecen fuera de Git. La URL incluida es https://breastdce-antonio.onrender.com. Llevar también PDF/PPTX y pesos en un medio local como respaldo de la defensa.
