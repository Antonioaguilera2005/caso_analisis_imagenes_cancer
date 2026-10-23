# Comparación clínica, después de cv5_v3_60ep

La CNN solo imagen se conserva como referencia principal. Esta comparación adicional utiliza exclusivamente `age` y `menopause`, nunca variables de respuesta ni de anatomía patológica. No se modifica `modelo.py` ni sus pesos históricos. No se ejecuta test: ya fue consultado para la primera versión y debe declararse en la memoria.

## Diseño congelado antes de entrenar

- Mismos cinco folds oficiales por paciente y semilla 42.
- Extractor de imagen idéntico: cuatro convoluciones 16/32/64/128, sin preentrenamiento.
- Fusión tardía: vector de imagen de 128 componentes + cuatro valores clínicos, dropout 0,3 y capa lineal a un logit. Total: 97.813 parámetros.
- Valores clínicos: edad estandarizada, indicador de edad ausente, menopausia binaria original e indicador de menopausia ausente.
- Mediana de edad, media y desviación de la edad imputada, y moda de menopausia calculadas exclusivamente sobre pacientes únicos de train de cada fold. Se guardan el estado y los IDs de ajuste. No se recalculan con validación.
- Se conserva cada paciente y todos sus cortes. La misma clínica se repite por corte y se promedian las probabilidades por paciente, como en la referencia. La pérdida por corte mantiene la ponderación implícita por número de cortes del modelo original.
- Adam, lr 0,001, weight decay 0,0001, batch 16; máximo 60 épocas, paciencia 12 por AUC de validación. Pérdidas normal y ponderada secuenciales, checkpoint por época.
- Comparar AUC media ± desviación y PR-AUC/AP, curvas y brechas. Umbral 0,5 como referencia; los umbrales exploratorios OOF no son una evaluación independiente ni una decisión clínica.

## Ausentes y limitaciones

En train hay 3 edades ausentes y 134 menopausias ausentes. Las 104 pacientes I-SPY1 carecen de menopausia; las otras 30 corresponden a I-SPY2. El indicador de ausencia puede actuar como marcador de cohorte. Informar rendimiento por cohorte y, si la mejora parece depender de este patrón, estudiar una ablación solo edad. No inferir menopausia a partir de edad como si fuera un dato observado.

La selección de época se hace con cada fold de validación: estas cifras no son una validación cruzada anidada. Los experimentos adicionales pueden introducir optimismo de selección. No elegir modelos por su resultado en el test histórico.

## Universidad: Linux y GPU AMD

Desde el repositorio, con el entorno GPU existente:

```bash
git pull
conda activate cancer
python -u validacion_cruzada.py --clinica --datos breastdcedl --salida runs/cv5_clinica_v1 --dispositivo cuda --epocas 60 --paciencia 12 --workers 2
```

No reinstalar PyTorch CPU en el entorno GPU. `cuda` es el identificador de PyTorch también con ROCm. La salida debe ser nueva. Son diez entrenamientos; al final se generan PNG y métricas en `runs/cv5_clinica_v1/analisis`. La cabecera debe indicar `entrada=imagenes + edad + menopausia` y 97.813 parámetros.

Traer la carpeta por USB u otra transferencia; no subir resultados ni pesos a Git. Para inspeccionar primero basta copiar `analisis` y, por fold/pérdida, los JSON, CSV y log; conservar todos los checkpoints en el equipo de origen y traer al menos el seleccionado si el espacio es limitado. Sin los pesos no se puede verificar su checksum ni reutilizar el modelo.

## Referencia solo clínica

```bash
python baseline_clinica.py --salida runs/baseline_clinica_v1
```

Regresión logística por paciente, C=1, versiones normal y balanced, mismos folds, preprocesado ajustado dentro de cada fold. Produce predicciones OOF, matrices de confusión y curvas. No sirve como prueba de que la red combinada necesariamente mejorará.

El cierre de `finalizar.py` solo admite la arquitectura histórica de imagen: rechaza esta variante clínica para evitar un reentrenamiento incompatible. Primero comparar la CV. La doble convolución será un experimento posterior separado.
