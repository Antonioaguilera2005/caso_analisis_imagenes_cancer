"""Dashboard educativo HTML + FastAPI; un único modelo cerrado, sin guardar subidas."""
from contextlib import asynccontextmanager
import io
import json
import os
from pathlib import Path
import zipfile
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.datastructures import UploadFile
from starlette.concurrency import run_in_threadpool
from starlette.formparsers import MultiPartParser, MultiPartException
from webapp.inferencia import Motor, preparar_archivos, MAX_ARCHIVO, MAX_CORTES

RAIZ = Path(__file__).resolve().parents[1]
ESTATICOS = Path(__file__).parent/'static'
MAX_CUERPO = 16 * 1024 * 1024


class FormularioEnMemoria(MultiPartParser):
    # El límite global impide llegar al umbral de volcado a un archivo temporal.
    spool_max_size = MAX_CUERPO + 1


@asynccontextmanager
async def formulario_memoria(request):
    parser = FormularioEnMemoria(request.headers, request.stream(), max_files=MAX_CORTES*3, max_fields=3, max_part_size=MAX_ARCHIVO)
    form = await parser.parse()
    try:
        yield form
    finally:
        await form.close()


@asynccontextmanager
async def lifespan(app):
    from webapp.paquete import preparar_paquete_remoto
    carpeta = Path(os.environ.get('MODEL_DIR', str(RAIZ/'artifacts/dashboard')))
    preparar_paquete_remoto(carpeta)
    app.state.motor = Motor(carpeta)
    app.state.carpeta = carpeta
    yield


app = FastAPI(title='BreastDCE · Laboratorio de respuesta', lifespan=lifespan, docs_url='/api/docs', redoc_url=None)


class LimitarCuerpo:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope['type'] != 'http':
            return await self.app(scope, receive, send)
        headers = dict(scope.get('headers', []))
        try:
            largo = int(headers.get(b'content-length', b'0'))
        except ValueError:
            return await JSONResponse({'detail':'Tamaño de solicitud no válido.'}, status_code=400)(scope, receive, send)
        if largo > MAX_CUERPO:
            return await JSONResponse({'detail':'El conjunto de archivos supera el límite de 16 MB.'}, status_code=413)(scope, receive, send)
        total = 0

        async def recibir():
            nonlocal total
            mensaje = await receive()
            total += len(mensaje.get('body', b''))
            if total > MAX_CUERPO:
                raise HTTPException(413, 'El conjunto de archivos supera el límite de 16 MB.')
            return mensaje
        await self.app(scope, recibir, send)


app.add_middleware(LimitarCuerpo)


@app.middleware('http')
async def cabeceras(request, call_next):
    response = await call_next(request)
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['Referrer-Policy'] = 'no-referrer'
    response.headers['Content-Security-Policy'] = "default-src 'self'; img-src 'self' blob: data:; style-src 'self'; script-src 'self'; connect-src 'self'; object-src 'none'; base-uri 'self'; frame-ancestors 'none'"
    if request.url.path.startswith('/api/'):
        response.headers['Cache-Control'] = 'no-store'
    return response


@app.get('/')
def index():
    return FileResponse(ESTATICOS/'index.html')


@app.get('/api/health')
def health(request: Request):
    return dict(estado='listo', modelo=request.app.state.motor.sha)


@app.get('/api/modelo')
def modelo(request: Request):
    return request.app.state.motor.resumen


@app.get('/api/ejemplos')
def ejemplos(request: Request):
    return request.app.state.motor.resumen.get('ejemplos', [])


def imagenes_ejemplo(request, ejemplo):
    datos = request.app.state.motor.resumen.get('ejemplos', [])
    ficha = next((v for v in datos if v['id'] == ejemplo), None)
    if not ficha:
        raise HTTPException(404, 'Ejemplo no disponible.')
    carpeta = request.app.state.carpeta/'ejemplos'/ejemplo
    archivos = []
    for nombre in ficha['archivos']:
        if Path(nombre).name != nombre:
            raise HTTPException(500, 'Paquete de ejemplo no válido.')
        archivos.append((nombre, (carpeta/nombre).read_bytes(), None))
    return archivos


@app.get('/api/ejemplos/{ejemplo}/archivo/{nombre}')
def archivo_ejemplo(request: Request, ejemplo: str, nombre: str):
    archivos = imagenes_ejemplo(request, ejemplo)
    contenido = next((b for n,b,_ in archivos if n == nombre), None)
    if contenido is None:
        raise HTTPException(404, 'Imagen no disponible.')
    from fastapi.responses import Response
    return Response(contenido, media_type='image/png', headers={'Cache-Control':'public, max-age=3600'})


@app.post('/api/ejemplos/{ejemplo}/predecir')
async def predecir_ejemplo(request: Request, ejemplo: str):
    x, ids = preparar_archivos(imagenes_ejemplo(request, ejemplo))
    return await ejecutar_inferencia(request, x, ids)


async def ejecutar_inferencia(request, x, ids):
    try:
        return await run_in_threadpool(request.app.state.motor.predecir, x, ids)
    except ValueError as exc:
        raise HTTPException(503, 'El modelo ha producido una salida no válida o no finita. No se ha emitido una predicción; revisa las imágenes o vuelve a intentarlo.') from exc


@app.post('/api/predecir')
async def predecir(request: Request):
    archivos = []
    try:
        if not request.headers.get('content-type','').lower().startswith('multipart/form-data'):
            raise ValueError('Carga las imágenes como formulario multipart.')
        async with formulario_memoria(request) as form:
            for campo, valor in form.multi_items():
                if campo not in ('PRE','EARLY','LATE','archivos') or not isinstance(valor, UploadFile):
                    raise ValueError('Solo se admiten imágenes en las casillas de fase o en el lote.')
                datos = await valor.read(MAX_ARCHIVO+1)
                archivos.append((valor.filename, datos, None if campo == 'archivos' else campo))
            x, ids = preparar_archivos(archivos)
    except (ValueError, MultiPartException) as exc:
        raise HTTPException(422, str(exc)) from exc
    return await ejecutar_inferencia(request, x, ids)


@app.get('/api/figuras/{nombre}')
def figura(request: Request, nombre: str):
    permitidos = request.app.state.motor.resumen.get('figuras', [])
    if nombre not in permitidos or Path(nombre).name != nombre:
        raise HTTPException(404, 'Figura no disponible.')
    return FileResponse(request.app.state.carpeta/'figuras'/nombre)


app.mount('/static', StaticFiles(directory=ESTATICOS), name='static')
