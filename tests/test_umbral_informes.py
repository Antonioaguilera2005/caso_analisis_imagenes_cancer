"""Los informes deben respetar el umbral congelado en lugar de asumir 0.5."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from analizar_resultados import casos_fallados, metricas


class UmbralInformeTest(unittest.TestCase):
    def test_f1_se_calcula_con_umbral_indicado(self):
        y, prob = [0,1,0], [0.2,0.4,0.3]
        self.assertEqual(metricas(y,prob,0.33)["f1_macro"],1.0)
        self.assertAlmostEqual(metricas(y,prob,0.5)["f1_macro"],0.4)

    def test_casos_fallados_respeta_umbral_final(self):
        with tempfile.TemporaryDirectory() as temporal:
            raiz = Path(temporal)
            filas = []
            for pid in ("a","b"):
                fila = dict(patient_id=pid,slice_index=0)
                for fase,columna in (("PRE","path_pre"),("EARLY","path_early"),("LATE","path_late")):
                    nombre=f"{pid}_{fase}.png"
                    Image.fromarray(np.full((256,256),128,dtype=np.uint8)).save(raiz/nombre)
                    fila[columna]=nombre
                filas.append(fila)
            tabla=pd.DataFrame(dict(patient_id=["a","b"],pCR=[1,0],prob=[0.4,0.4]))
            casos_fallados(tabla,"prueba",pd.DataFrame(filas),raiz,raiz,umbral=0.33)
            fallos=pd.read_csv(raiz/"fallos_prueba.csv")
            self.assertEqual(fallos.patient_id.tolist(),["b"])
            self.assertEqual(fallos.tipo.tolist(),["FP"])


if __name__=="__main__":
    unittest.main()
