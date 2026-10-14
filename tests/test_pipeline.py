"""Comprobaciones de alineación, agregación y propagación de gradientes."""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "breastdcedl"))
import utils_caso as uc
from modelo import CNNBreastDCE, aumento_geometrico
from entrenar import validar_particion


class PipelineTest(unittest.TestCase):
    def test_aumentos_conservan_alineacion_y_realce(self):
        patron = torch.arange(64, dtype=torch.float32).reshape(8, 8) / 100
        x = torch.stack([patron, patron + 0.1, patron + 0.2])
        for _ in range(20):
            aumentado = aumento_geometrico(x)
            torch.testing.assert_close(aumentado[1] - aumentado[0], torch.full((8, 8), 0.1))
            torch.testing.assert_close(aumentado[2] - aumentado[0], torch.full((8, 8), 0.2))

    def test_agregacion_cuenta_pacientes_no_cortes(self):
        filas = pd.DataFrame({"patient_id": ["a", "a", "b", "c"], "pCR": [1, 1, 0, 1]})
        resultado = uc.evaluar_por_paciente([0.2, 0.8, 0.1, 0.9], filas)
        self.assertEqual(resultado["pacientes"], 3)
        self.assertEqual(resultado["matriz_confusion"], {"VP": 2, "VN": 1, "FP": 0, "FN": 0})
        self.assertEqual(resultado["auc"], 1)

    def test_rechaza_solape_y_test(self):
        tr = pd.DataFrame({"patient_id": ["a", "b"], "pCR": [0, 1], "split": ["train", "train"]})
        with self.assertRaisesRegex(ValueError, "Fuga"):
            validar_particion(tr, tr)
        va = tr.assign(patient_id=["c", "d"], split="test")
        with self.assertRaisesRegex(ValueError, "Solo"):
            validar_particion(tr, va)

    def test_cnn_produce_gradientes_finitos(self):
        modelo = CNNBreastDCE()
        x = torch.rand(2, 3, 256, 256)
        logits = modelo(x)
        self.assertEqual(logits.shape, (2,))
        loss = torch.nn.BCEWithLogitsLoss()(logits, torch.tensor([0.0, 1.0]))
        loss.backward()
        for parametro in modelo.parameters():
            self.assertIsNotNone(parametro.grad)
            self.assertTrue(torch.isfinite(parametro.grad).all())


if __name__ == "__main__":
    unittest.main()
