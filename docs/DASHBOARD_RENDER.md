# Dashboard HTML + FastAPI

Interfaz propia HTML/CSS/JavaScript, sin Streamlit. FastAPI ejecuta la CNN definitiva en CPU. La imagen no se redimensiona ni se normaliza por fase: PNG L de 8 bits, 256×256, orden PRE/EARLY/LATE, float32 /255. Se promedian las probabilidades de los cortes y se aplica el umbral de la receta verificada. No se usan variables clínicas en el modelo final.

## Abrir en el portátil

Desde el repositorio, PowerShell (no requiere activar scripts):

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-web.txt
# Solo si todavía no existe artifacts/dashboard:
.\.venv\Scripts\python.exe scripts/preparar_dashboard.py
.\.venv\Scripts\python.exe -m uvicorn webapp.main:app --host 127.0.0.1 --port 8000 --workers 1 --no-access-log
```

Abrir <http://127.0.0.1:8000>. El paquete local ya se preparó durante el desarrollo; no hace falta generarlo otra vez. No instalar `requirements-web.txt` en el entorno GPU de la universidad: contiene PyTorch CPU. Para ese equipo se conservan sus dependencias ROCm.

## Funciones

- Un corte con casillas explícitas PRE/EARLY/LATE, o lote de hasta 32 cortes de una paciente con los nombres oficiales. Tres fases por corte; hasta 1 MB por PNG y 16 MB por solicitud.
- Dos ejemplos públicos de train. Son demostrativos, no evaluación independiente.
- PRE/EARLY/LATE y mapa firmado EARLY−PRE, escala fija; no es Grad-CAM ni segmentación.
- Resultado por corte y media por paciente, clase, umbral, checksum, CPU y duración de inferencia. Un solo corte se identifica como parcial.
- Registro en memoria de la pestaña y descarga JSON para las cinco muestras de la defensa. No guarda archivos subidos en disco. El proveedor de hosting puede registrar metadatos de peticiones, fuera del control de la app.
- Métricas históricas, baseline, matrices, cohortes y calibración; arquitectura, hiperparámetros, CV, curvas reales y limitaciones. No se recalculan las métricas ni se consulta test al usar la app.

## Paquete externo: conservar resultados fuera de Git

`scripts/preparar_dashboard.py` crea `artifacts/dashboard` y `artifacts/dashboard_bundle.zip` y su SHA-256. Contiene el checkpoint final seleccionado, el resumen derivado de los resultados ya calculados, tres figuras y ejemplos públicos de train. `artifacts/` está ignorado por Git y Docker. No se necesita subir el dataset completo.

El paquete es serialización PyTorch de origen propio; nunca se acepta un modelo subido por visitantes. Al iniciar se verifica SHA del paquete remoto, SHA del peso, tipo de checkpoint, receta y hash del código de arquitectura. Se usa `weights_only=True`. Guardar la URL del paquete solo en la configuración de Render, no en el código. El contenido puede descargarse desde un almacenamiento con URL HTTPS directa (no página de vista previa). Elegir un acceso conforme a la privacidad deseada y a las condiciones del proveedor.

## Render, paso a paso

No hay una URL pública creada todavía. Se necesita acceso a la cuenta Render y alojar el ZIP fuera de Git.

1. Subir `artifacts/dashboard_bundle.zip` a un almacenamiento que entregue el archivo por URL HTTPS directa. Conservar la huella de `artifacts/dashboard_bundle.sha256`.
2. Entrar en Render, conectar el repositorio `Antonioaguilera2005/caso_analisis_imagenes_cancer` y crear un **Blueprint** usando `render.yaml` (alternativa: Web Service con runtime Docker).
3. Introducir `MODEL_BUNDLE_URL` y `MODEL_BUNDLE_SHA256` cuando lo solicite Render. No colocar estos valores en un commit. `MODEL_DIR` lo fija Docker por defecto.
4. Desplegar. El Dockerfile instala solo las dependencias del backend y escucha en `$PORT`, con un worker. Al arrancar descarga el paquete y verifica sus hashes; si fallan no sirve predicciones.
5. Abrir la URL HTTPS que devuelva Render. Comprobar `/api/health`, ejecutar un ejemplo y un corte nuevo, navegar por resultados y descargar un JSON. Comparar probabilidad y SHA con la prueba local; pueden existir pequeñas diferencias numéricas entre versiones/plataformas.
6. Ensayar cinco cargas consecutivas sin reiniciar ni tocar código. Abrir también la URL desde otro dispositivo. Añadir la URL verificada al README, memoria y diapositivas.

El Blueprint usa plan gratuito; no se ha contratado nada. Render documenta suspensión por inactividad en servicios gratuitos: abrir y comprobar la app antes de la exposición; no confundir un arranque lento con una predicción fallida. Ver condiciones vigentes: [FastAPI en Render](https://render.com/docs/deploy-fastapi) y [límites del plan gratuito](https://render.com/docs/free). El consumo real y los tiempos deben verificarse en Render, no extrapolar los del portátil.

## Verificación reproducible

```powershell
.\.venv\Scripts\python.exe -m pip install httpx
.\.venv\Scripts\python.exe -m unittest discover -s tests
```

Las pruebas reales de app requieren el paquete local; si no existe, se omiten explícitamente. QA de navegador opcional: instalar Playwright y ejecutar `scripts/probar_dashboard_navegador.py` con Chrome instalado y el servidor en puerto 8000. No se incluye Playwright en la imagen Docker.

## Antes de defender

Revisar `AUDITORIA_ENUNCIADO.md`, especialmente la cronología del test. Conservar la atribución, la licencia, los pesos y su checksum. El resultado no constituye diagnóstico, pronóstico de supervivencia ni recomendación terapéutica. El dashboard no sustituye la memoria ni las exactamente cinco diapositivas.
