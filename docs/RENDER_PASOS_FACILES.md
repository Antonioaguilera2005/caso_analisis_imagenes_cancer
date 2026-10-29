# Render sin pasos técnicos innecesarios

La app ejecuta tus pesos reales. Un pendrive permite mover el paquete entre ordenadores, pero no da a Render un enlace para descargarlo. No hay que subir el dataset ni las carpetas de runs.

## 1. Localiza el ZIP

En el Explorador de VS Code abre la carpeta `PARA_RENDER`, dentro del repositorio. Allí está `1_SUBIR_ESTE_ARCHIVO.zip`. Es una copia del paquete verificado de aproximadamente 3,8 MB, preparada para que sea fácil localizarlo. No lo descomprimas. La carpeta está excluida de Git.

## 2. Ponlo en Drive

Si tienes cuenta de Google, abre https://drive.google.com e inicia sesión.

1. Pulsa **Nuevo → Subir archivo**.
2. Selecciona `PARA_RENDER/1_SUBIR_ESTE_ARCHIVO.zip`.
3. Cuando termine, clic derecho en el archivo → **Compartir**.
4. En **Acceso general**, cambia **Restringido** a **Cualquier persona con el enlace**, con permiso **Lector**. Esto permite descargar el paquete a quien tenga el enlace: solo contiene el modelo docente, resultados agregados y ejemplos públicos de train, no validación privada.
5. Pulsa **Copiar enlace** y pégalo en la conversación. No envíes contraseñas ni claves. Se comprobará que el archivo puede descargarse sin iniciar sesión, y que su SHA coincide, antes de usarlo en Render. Si una cuenta de universidad no permite compartir externamente, hará falta otro almacenamiento o una cuenta con esa posibilidad.

No tienes que construir una URL de descarga a mano. El enlace de vista previa de Drive no siempre sirve directamente para Render; hay que comprobarlo y convertirlo cuando corresponda. No iniciar el despliegue hasta tener ese enlace probado.

Referencias oficiales: [subir archivos](https://support.google.com/drive/answer/2424368?hl=es) y [compartir por enlace](https://support.google.com/drive/answer/2494822?hl=es).

## 3. En la pantalla de Render que has enviado

1. Pulsa **New Web Service**, en la tarjeta **Web Services**.
2. Elige el repositorio **Antonioaguilera2005/caso_analisis_imagenes_cancer**. Si no aparece, usa **Configure GitHub** para autorizar ese repositorio y vuelve a seleccionarlo.
3. Configura **Name**: `breastdce-antonio`; **Branch**: `main`; **Language / Runtime**: `Docker`. Deja **Root Directory** vacío (el repositorio ya contiene el Dockerfile en su raíz). Elige **Free**, si aparece disponible. No necesitas base de datos.
4. En **Environment Variables** añade:
   - `MODEL_BUNDLE_URL`: enlace de descarga HTTPS que hayamos comprobado.
   - `MODEL_BUNDLE_SHA256`: el contenido de `PARA_RENDER/2_CONFIGURACION_RENDER.txt` para esa variable.
5. Solo cuando ambas variables estén correctas, pulsa **Deploy Web Service**. El Dockerfile ya define el arranque; no escribas comandos de Streamlit.
6. Espera a que Render indique **Live**. Abre la URL `https://...onrender.com` que muestre. Si falla, conserva el mensaje del registro: no cambies pesos ni receta para intentar arreglar el despliegue.

Render no tiene acceso a los archivos de tu portátil ni del pendrive. Usa GitHub para obtener el código y la URL externa para obtener el paquete. [Documentación oficial de Web Services](https://render.com/docs/web-services), [Docker](https://render.com/docs/docker) y [variables](https://render.com/docs/configure-environment-variables).

## 4. Comprueba la defensa

En la URL pública: ejecutar un ejemplo, cargar un corte nuevo, comprobar hash/umbral, repetir cinco muestras sin reiniciar y descargar el registro JSON. El aviso educativo y las limitaciones deben permanecer visibles. La cuenta y las restricciones del plan de Render pueden afectar el arranque; comprobar la URL antes de la exposición.
