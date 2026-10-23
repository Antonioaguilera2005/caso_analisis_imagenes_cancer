import unittest
import numpy as np
import pandas as pd
import torch
from clinica import ajustar_clinica, transformar_clinica, CNNConClinica


class TestClinica(unittest.TestCase):
    def setUp(self):
        self.tr = pd.DataFrame(dict(pid=["a", "b", "c"], split=["train"] * 3,
                                    age=[30., 50., np.nan], menopause=[0., 1., np.nan]))

    def test_no_reajuste_con_validacion(self):
        estado = ajustar_clinica(self.tr)
        va = self.tr.copy()
        va.age = [100., np.nan, 1000.]
        x = transformar_clinica(va, estado)
        self.assertEqual(estado["edad_mediana"], 40.)
        self.assertAlmostEqual(x[1, 0], 0.)
        self.assertEqual(x[1, 1], 1.)
        self.assertEqual(x[2, 3], 1.)
        self.assertTrue(np.isfinite(x).all())

    def test_rechaza_test_y_duplicados(self):
        with self.assertRaises(ValueError):
            ajustar_clinica(self.tr.assign(split="test"))
        with self.assertRaises(ValueError):
            ajustar_clinica(pd.concat([self.tr, self.tr]))

    def test_fusion_recibe_gradientes(self):
        modelo = CNNConClinica()
        x = torch.randn(2, 3, 32, 32)
        c = torch.randn(2, 4, requires_grad=True)
        y = modelo((x, c))
        self.assertEqual(tuple(y.shape), (2,))
        y.sum().backward()
        self.assertGreater(c.grad.abs().sum().item(), 0.)
        self.assertGreater(modelo.features[0].weight.grad.abs().sum().item(), 0.)


if __name__ == "__main__":
    unittest.main()
