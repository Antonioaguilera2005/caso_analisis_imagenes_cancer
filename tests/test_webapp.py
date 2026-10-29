import io
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import torch
from fastapi.testclient import TestClient
from webapp.inferencia import cargar_png, preparar_archivos, Motor
from webapp.main import app

REPO=Path(__file__).resolve().parents[1]


def png(size=(256,256), mode='L', valor=100):
    f=io.BytesIO()
    Image.new(mode,size,valor).save(f,format='PNG')
    return f.getvalue()


class ValidacionWeb(unittest.TestCase):
    def test_formato_dimensiones_y_tipo(self):
        for datos in [b'no soy png',png((128,256)),png(mode='RGB'),b'x'*(1024*1024+1)]:
            with self.assertRaises(ValueError):
                cargar_png(datos)

    def test_fases_y_paciente(self):
        archivos=[(f'A_z001_{f}.png',png(),None) for f in ['PRE','EARLY','LATE']]
        for incorrecto in [archivos[:2],archivos+[archivos[0]],archivos[:2]+[('B_z001_LATE.png',png(),None)], [('..\\x.png',png(),'PRE')]]:
            with self.assertRaises(ValueError):
                preparar_archivos(incorrecto)
        x, ids=preparar_archivos(archivos)
        self.assertEqual(x.shape,(1,3,256,256))
        self.assertEqual(ids,['z001'])
        np.testing.assert_array_equal(x,np.full_like(x,np.float32(100)/255))

    def test_fase_mal_asignada(self):
        archivos=[('A_z001_PRE.png',png(),'EARLY'),('A_z001_EARLY.png',png(),'PRE'),('A_z001_LATE.png',png(),'LATE')]
        with self.assertRaises(ValueError):
            preparar_archivos(archivos)


@unittest.skipUnless((REPO/'artifacts/dashboard/resumen.json').exists(),'Preparar paquete local para pruebas reales')
class IntegracionWeb(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.context=TestClient(app)
        cls.client=cls.context.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.context.__exit__(None,None,None)

    def test_modelo_real_y_repetibilidad(self):
        c=self.client
        self.assertEqual(c.get('/').status_code,200)
        self.assertEqual(c.get('/api/health').json()['estado'],'listo')
        meta=c.get('/api/modelo').json()
        eid=meta['ejemplos'][0]['id']
        a=c.post(f'/api/ejemplos/{eid}/predecir')
        b=c.post(f'/api/ejemplos/{eid}/predecir')
        self.assertEqual(a.status_code,200,a.text)
        self.assertEqual(a.json()['probabilidad'],b.json()['probabilidad'])
        self.assertEqual(a.json()['umbral'],.33)
        self.assertEqual(a.json()['sha256'],meta['sha256_checkpoint'])

    def test_equivalencia_utilidades_y_modelo_original(self):
        import sys
        sys.path.insert(0,str(REPO/'breastdcedl'))
        import utils_caso as uc
        from finalizar import cargar_modelo_final
        fila=pd.read_csv(REPO/'breastdcedl/metadata/samples.csv').query("split == 'train'").iloc[0]
        archivos=[(Path(fila[f'path_{f.lower()}']).name,(REPO/'breastdcedl'/fila[f'path_{f.lower()}']).read_bytes(),None) for f in ['PRE','EARLY','LATE']]
        x,ids=preparar_archivos(archivos)
        np.testing.assert_array_equal(x[0],uc.cargar_imagen(fila,REPO/'breastdcedl'))
        m,r,a=cargar_modelo_final(REPO/'runs/modelo_final')
        with torch.inference_mode():
            esperado=torch.sigmoid(m(torch.from_numpy(x))).item()
        real=self.client.app.state.motor.predecir(x,ids)
        self.assertAlmostEqual(real['probabilidad'],esperado,places=7)

    def test_errores_api_y_subida(self):
        c=self.client
        self.assertEqual(c.post('/api/predecir').status_code,422)
        files=[(f,(f'{f}.png',png(valor=v),'image/png')) for f,v in [('PRE',50),('EARLY',100),('LATE',90)]]
        r=c.post('/api/predecir',files=files)
        self.assertEqual(r.status_code,200,r.text)
        self.assertEqual(r.json()['cortes'],1)
        self.assertEqual(c.post('/api/predecir',files=[('PRE',('bad.png',b'bad','image/png'))]).status_code,422)
        self.assertEqual(c.get('/api/ejemplos/no-existe/archivo/no.png').status_code,404)
        self.assertEqual(c.get('/api/figuras/no.png').status_code,404)
        self.assertEqual(c.post('/api/predecir',headers={'Content-Length':str(17*1024*1024)}).status_code,413)


if __name__=='__main__':
    unittest.main()
