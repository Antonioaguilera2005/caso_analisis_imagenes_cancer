# Criterios de análisis tras la aclaración del profesor

## Problema y evaluación principal

CNN propia 2D desde cero con PRE/EARLY/LATE; sin modelos preentrenados.
La unidad es la paciente. Media de probabilidades de sus cortes y umbral elegido
exclusivamente con validación interna. Las variables clínicas pueden servir como
comparación secundaria, no sustituir el modelo de imagen.
Comprobar qué variables estarían disponibles antes del tratamiento; no asumir
disponibilidad ni añadir variables medidas tras conocer la respuesta.

## Validación cruzada

Cinco folds separados por paciente. Comparar normal/ponderada con igual receta.
Guardar los resultados individuales y media ± desviación muestral entre folds.
La AUC combinada de predicciones fuera de fold es otra medida: no sustituye
la media entre folds, y mezcla probabilidades de modelos diferentes.
Las pacientes del fold elegido para validar nunca se utilizan para entrenar ese modelo.

## Sobreajuste y checkpoints

Guardar un checkpoint independiente por época. Elegir según AUC de validación
por paciente; en empates, la primera época. Mostrar BCE comparable train/val
sin aumentos y AUC por paciente en ambas particiones. Indicar la brecha en la
época elegida; pérdidas online con dropout/aumentos no son la misma evaluación.
No elegir ni descartar épocas a partir de test.

## Métricas y umbral

AUC mide ordenación; el umbral cambia precision, recall, especificidad y F1,
pero no aumenta la AUC. Accuracy requiere comparación con decir siempre no pCR.
Informar precision, recall y F1 de cada clase, y F1 macro: media de ambas clases.
La precisión de una clase sin predicciones de esa clase es indefinida y se
guarda como null; F1 se calcula con la convención cero para ese caso.

El análisis actual explora F1 macro como criterio educativo para dar el mismo
peso a las dos clases. No existe un F1 óptimo universal ni un umbral clínico
validado para este ejercicio. Un FP de pCR atribuye respuesta completa a una
paciente con enfermedad residual; un FN no reconoce una respuesta completa.
La aplicación debe comunicar incertidumbre, sin sugerir cambios de tratamiento.
Si se prioriza alguno de esos errores, justificarlo antes de cerrar el umbral.

Las métricas de un umbral ajustado en las mismas predicciones son optimistas.
Para comparar ajustes con mayor rigor se puede separar una validación de ajuste
interna o usar un esquema anidado. El test público permanece intacto hasta cerrar
el modelo, agregación y umbral; se evalúa una sola vez.

## Más allá de la cifra global

Analizar resultados por cohorte, mostrando los tamaños de grupo. Generar
curvas ROC/PR, calibración y Brier, distribución de probabilidades y ejemplos
de errores con PRE/EARLY/LATE/realce. No explicar causalmente el error de la red
solo observando una imagen. Un corte mostrado es ilustrativo; la probabilidad
del paciente incorpora todos sus cortes.

El IC de AUC usa bootstrap estratificado de pacientes, nunca de cortes.
En validación, es condicional al modelo elegido; no captura selección ni
variabilidad del entrenamiento. La desviación entre folds mide algo distinto.

## Bibliografía y comparabilidad

El correo del profesor aporta orientaciones y referencias; verificar las cifras
en sus fuentes antes de reproducirlas como resultados publicados del informe.
El PDF de Scientific Data adjunto corresponde a 2026 y describe 177 pacientes
con pCR en validation y 176 en test. No se debe identificar automáticamente un
resultado citado para 177 pacientes con nuestro test de 176.
El PDF adjunto es principalmente un descriptor de datos: las cifras predictivas
citadas en el correo requieren localizar el benchmark o estudio específico.
Comparar modalidad, uso de clínica, cohortes, tamaño y evaluación independiente;
no presentar una comparación bibliográfica como competición en condiciones idénticas.

## Transporte y publicación

Solo código e instrucciones van a GitHub. Entrenamientos, pesos, métricas y PNG
derivados se guardan en `runs/`, ignorado por Git. Copiar la carpeta completa y
el diagnóstico por USB, correo o almacenamiento personal. No eliminar la copia
de clase hasta verificar la del portátil. No publicar la validación privada.
