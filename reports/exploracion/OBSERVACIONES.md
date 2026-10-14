# Exploración inicial de train

Se seleccionó por identificador una paciente de cada combinación cohorte/pCR:
seis pacientes, diez cortes cada una. Se comprobaron los 180 PNG de esta selección:
formato PNG, escala de grises, 256×256 y tensores finitos dentro de [0,1].
La auditoría anterior de la descarga comprobó presencia y tamaño no vacío de
las 38.109 imágenes, no su decodificación completa.

En las seis pacientes, el promedio de intensidad EARLY supera PRE. Las medias
se calculan dentro de PRE > 0.1 para excluir fondo, primero por corte y después
por paciente. Tres presentan LATE menor que EARLY. Estos recuentos describen
la selección ilustrativa; no estiman frecuencias poblacionales.

La figura muestra las tres fases y EARLY−PRE de un corte central de cada paciente.
La escala de grises es [0,1] para todas las fases. El mapa de realce tiene escala
simétrica ajustada a cada corte: su color no permite comparar magnitudes entre filas.
No se puede concluir de estas imágenes que una clase tenga mayor realce que otra.

Clases de entrenamiento a nivel paciente: 775 no pCR y 322 pCR, aproximadamente
70.6% y 29.4%. Esta es la distribución relevante para contextualizar las métricas.
Las cohortes difieren en adquisición y selección de cortes; conviene comprobar
más adelante si el modelo aprende diferencias entre fuentes.

Datos derivados de BreastDCEDL, Fridman y colaboradores,
DOI 10.1038/s41597-026-06589-6. Uso educativo sin validez clínica.
