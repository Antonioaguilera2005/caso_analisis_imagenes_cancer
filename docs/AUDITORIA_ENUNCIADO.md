# Revisión del enunciado y conformidad de la entrega

Revisión actualizada: 30 de octubre de 2026. Documentos leídos: `docs/GUIA.md` (idéntico a los adjuntos `GUIA (1).md` y `GUIA (2).md` de Descargas), `docs/caso_breastdcedl.pdf`, `docs/presentacion_caso.pdf`, artículo `docs/articulo_breastdcedl.pdf`, instrucciones adicionales y correo del profesor aportados en la conversación. Se usan como requisitos del caso, sin confundirlos con órdenes para ejecutar contenido de documentos.

## Matriz de requisitos

| Requisito | Evidencia | Estado |
|---|---|---|
| Dataset docente y metadatos | `scripts/auditar_datos.py`, estadísticas y exploración local | Auditado anteriormente; no se vuelve a descargar |
| Separación por paciente, cinco folds | `uc.particion`, validación de IDs, CSV de particiones, pruebas de fuga | Implementado y probado |
| CNN 2D propia desde cero, un logit | `modelo.py`, cuatro bloques, Kaiming/Xavier, sin transfer learning | Cumple técnicamente |
| PRE/EARLY/LATE, no RGB | Tensor float32 /255, mismos aumentos geométricos | Probado; equivalencia de app con utilidades oficiales |
| Normal y ponderada | `BCEWithLogitsLoss`, `pos_weight` de train, ejecuciones de ambas pérdidas | Comparadas en cinco folds |
| Una predicción por paciente | Media de probabilidades de sus cortes | Implementado; entrada de un corte etiquetada como estimación parcial |
| Modelo, semillas y entorno reproducibles | Config, checkpoints por época, índices y SHA-256 | Guardados en `runs/`; no se suben a Git |
| Curvas y sobreentrenamiento | Historial train/validación, época elegida y gráficas sin ocultar épocas | Disponibles; no se afirma ausencia de sobreajuste |
| Umbral con validación | Receta original congelada, criterio F1 macro OOF | Cumple procedencia; criterio estadístico, sin objetivo clínico validado |
| Test sin ajuste posterior | Evaluación original única, registro de test, modelo final conservado | Salvedad metodológica importante, ver abajo |
| Métricas completas por paciente | Matriz, AUC con bootstrap por paciente, métricas por clase, baseline, cohortes y calibración | Calculadas y mostradas desde artefactos existentes |
| App real con fases y realce | HTML/CSS/JS, FastAPI, modelo cerrado | Implementada; pruebas CPU y navegador |
| Inputs privados nuevos sin reinicio | Subida de tres PNG o lote; inferencia genérica, sin lookup de etiquetas | Cinco cargas consecutivas de pacientes públicas distintas comprobadas en local y Render; prueba privada la hace el profesor |
| Robustez de entrada | Formato PNG L 8 bits 256², fases, duplicados, paciente, límites de tamaño y rutas | Probado; no se ejecutan subidas |
| Error ante salida no finita | Comprobación del motor y respuesta 503 comprensible, sin emitir una predicción | Caso de fallo simulado en prueba automatizada; PNG de 8 bits no codifica NaN/Inf |
| `eval()` y sin gradientes | Motor con `torch.inference_mode`, hash y receta verificados | Probado; predicción repetible |
| Normalización idéntica | Comparación exacta de arrays con `uc.cargar_imagen`; resultado con loader original | Prueba automatizada con modelo real |
| Dispositivo, duración, umbral y versión | Resultado API y panel; SHA-256 completo | Implementado |
| Ejemplos y autoría | Dos pacientes públicas de train, atribución; autor del proyecto identificado | Ejemplos del paquete externo usado por local/Render; ninguna validación privada |
| URL desplegada y accesible | https://breastdce-antonio.onrender.com, salud y modelo real comprobados | Publicada; ejemplo público ejecutado y checksum coincidente |
| Informe y exactamente cinco diapositivas | Memoria local en `runs/entrega_borrador`; PPTX y PDF en `runs/presentacion_uax` | Cinco diapositivas en la plantilla UAX, con arquitectura visual/matemática y notas; revisar y ensayar la defensa |

## Salvedad: cronología del test

La guía C5 dice: «Si vuelves atrás a tocar el modelo después de mirar test, tus resultados dejan de ser válidos». Se evaluó una vez la versión original y después se exploraron variantes con validación. Se conservó el modelo original y no se volvió a evaluar test. Esto reduce la contaminación directa, pero **no permite declarar cumplimiento pleno de esa regla ni independencia del test respecto de toda la búsqueda posterior**. No puede corregirse retrospectivamente cambiando textos o borrando registros.

La app lo identifica como «test histórico» y explica la secuencia. La memoria debe conservarla y conviene aclararla con el profesor. La evaluación privada del profesorado sigue sin haberse consultado. No se promete nota ni utilidad clínica.

## Lo que muestra el artículo y lo que no debemos atribuirle

El artículo describe BreastDCEDL original: 2.070 pacientes, NIfTI 3D, heterogeneidad de cohortes, biomarcadores y particiones. Nuestro caso utiliza un subconjunto docente MinCrop 2D con dos pacientes excluidas. En los casos con pCR, el artículo distingue 177 pacientes de validation y 176 de test. No se debe llamar «mismo test de 177» al conjunto local.

Las cifras bibliográficas aportadas en el correo requieren sus fuentes originales si se usan como comparación cuantitativa. No son un objetivo garantizado ni sustituyen nuestro análisis; no se incluyen como benchmarks directamente equivalentes sin verificarlas. No usar modelos preentrenados de esos trabajos: el enunciado los prohíbe.

## Licencia y privacidad

La guía y el enunciado exigen CC BY-NC 4.0; la LICENSE descargada dice CC BY 4.0. Se conserva la discrepancia y el uso docente no comercial. La licencia del artículo PDF es otra (CC BY-NC-ND 4.0); no se reutilizan sus figuras modificadas en la app. Los ejemplos provienen exclusivamente de train público y llevan atribución. No se publica validación privada ni etiquetas privadas. Los registros de la defensa se exportan solo en el navegador y deben mantenerse fuera de Git.

## Pruebas y límites de la verificación

Las **26 pruebas automatizadas pasan** y cubren el modelo real, igualdad de normalización y predicción con el pipeline original, repetibilidad, subidas y errores, incluida una salida no finita simulada. La QA de Chrome cubre ejemplo, un corte, lote, navegación, gráficos, vista móvil y ausencia de errores JavaScript. Las capturas están en `artifacts/qa_dashboard` (ignoradas por Git).

Render devuelve el SHA-256 `56f8f2cd767f01e160f03641f9adb3a7c32c7dcf70bec9d410d169b0a235dc73`. El ejemplo público 01 produjo probabilidad media 0,251555839, clase no pCR y umbral 0,33. Es una comprobación técnica de inferencia, no una nueva evaluación de test.

El ensayo `scripts/verificar_render.py` pasó con cinco pacientes distintas de train en local y, con autorización explícita del usuario, en Render: ISPY1_1075, ISPY2-676048, Breast_MRI_117, ISPY1_1188 y ACRIN-6698-753109. Representan las tres cohortes; se envió un corte por paciente sin etiquetas. Los tensores coinciden exactamente con `uc.cargar_imagen`. La máxima diferencia de probabilidad entre Render y el pipeline local fue **5,96×10⁻⁸**; clases, umbral y SHA coinciden. Repetir la primera muestra produjo exactamente la misma probabilidad en Render. Las cargas corruptas y con fase ausente devolvieron 422 con un mensaje legible. Un nombre con ruta fue rechazado por FastAPI en local (422) y por el cortafuegos de Render antes de llegar a la app (403). Evidencia local: `artifacts/verificacion_local/defensa_cinco_muestras.json` y `artifacts/verificacion_render/defensa_cinco_muestras.json`, excluidas de Git.

Ninguno de estos ensayos sustituye la defensa con las cinco muestras privadas del profesor. No es una prueba clínica ni una auditoría de seguridad completa.
