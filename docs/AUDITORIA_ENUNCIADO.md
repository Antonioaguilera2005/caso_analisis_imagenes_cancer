# Revisión del enunciado y conformidad de la entrega

Revisión: 29 de octubre de 2026. Documentos leídos: `docs/GUIA.md` (idéntico al adjunto `GUIA (1).md` de Descargas), `docs/caso_breastdcedl.pdf`, `docs/presentacion_caso.pdf`, artículo `docs/articulo_breastdcedl.pdf`, instrucciones adicionales y correo del profesor aportados en la conversación. Se usan como requisitos del caso, sin confundirlos con órdenes para ejecutar contenido de documentos.

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
| Inputs privados nuevos sin reinicio | Subida de tres PNG o lote; inferencia genérica, sin lookup de etiquetas | Probado con archivos públicos; prueba privada la hace el profesor |
| Robustez de entrada | Formato PNG L 8 bits 256², fases, duplicados, paciente, límites de tamaño y rutas | Probado; no se ejecutan subidas |
| `eval()` y sin gradientes | Motor con `torch.inference_mode`, hash y receta verificados | Probado; predicción repetible |
| Normalización idéntica | Comparación exacta de arrays con `uc.cargar_imagen`; resultado con loader original | Prueba automatizada con modelo real |
| Dispositivo, duración, umbral y versión | Resultado API y panel; SHA-256 completo | Implementado |
| Ejemplos y autoría | Dos pacientes públicas de train, atribución; autor del proyecto identificado | Ejemplos solo en paquete local, no validación privada |
| URL desplegada y accesible | Dockerfile, Render Blueprint, paquete de pesos externo | Pendiente de creación en la cuenta Render y verificación de la URL |
| Informe y exactamente cinco diapositivas | Documentación científica local; esquema de cinco diapositivas | Se deben cerrar y revisar los entregables finales; el dashboard no los sustituye |

## Salvedad: cronología del test

La guía C5 dice: «Si vuelves atrás a tocar el modelo después de mirar test, tus resultados dejan de ser válidos». Se evaluó una vez la versión original y después se exploraron variantes con validación. Se conservó el modelo original y no se volvió a evaluar test. Esto reduce la contaminación directa, pero **no permite declarar cumplimiento pleno de esa regla ni independencia del test respecto de toda la búsqueda posterior**. No puede corregirse retrospectivamente cambiando textos o borrando registros.

La app lo identifica como «test histórico» y explica la secuencia. La memoria debe conservarla y conviene aclararla con el profesor. La evaluación privada del profesorado sigue sin haberse consultado. No se promete nota ni utilidad clínica.

## Lo que muestra el artículo y lo que no debemos atribuirle

El artículo describe BreastDCEDL original: 2.070 pacientes, NIfTI 3D, heterogeneidad de cohortes, biomarcadores y particiones. Nuestro caso utiliza un subconjunto docente MinCrop 2D con dos pacientes excluidas. En los casos con pCR, el artículo distingue 177 pacientes de validation y 176 de test. No se debe llamar «mismo test de 177» al conjunto local.

Las cifras bibliográficas aportadas en el correo requieren sus fuentes originales si se usan como comparación cuantitativa. No son un objetivo garantizado ni sustituyen nuestro análisis; no se incluyen como benchmarks directamente equivalentes sin verificarlas. No usar modelos preentrenados de esos trabajos: el enunciado los prohíbe.

## Licencia y privacidad

La guía y el enunciado exigen CC BY-NC 4.0; la LICENSE descargada dice CC BY 4.0. Se conserva la discrepancia y el uso docente no comercial. La licencia del artículo PDF es otra (CC BY-NC-ND 4.0); no se reutilizan sus figuras modificadas en la app. Los ejemplos provienen exclusivamente de train público y llevan atribución. No se publica validación privada ni etiquetas privadas. Los registros de la defensa se exportan solo en el navegador y deben mantenerse fuera de Git.

## Pruebas y límites de la verificación

Las pruebas cubren el modelo real, igualdad de normalización y predicción con el pipeline original, repetibilidad, subidas y errores. La QA de Chrome cubre ejemplo, un corte, lote, navegación, gráficos, vista móvil y ausencia de errores JavaScript. Las capturas están en `artifacts/qa_dashboard` (ignoradas por Git). No es una prueba clínica ni una auditoría de seguridad completa. El contenedor en Render y su URL deben verificarse tras desplegar.
