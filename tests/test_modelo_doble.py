import io
import unittest
import torch
from torch import nn
from modelo_doble import CNNBreastDCEDoble


class TestDoble(unittest.TestCase):
    def test_bloques_y_gradientes(self):
        m = CNNBreastDCEDoble()
        self.assertEqual(sum(isinstance(c, nn.Conv2d) for c in m.modules()), 8)
        self.assertEqual(sum(isinstance(c, nn.MaxPool2d) for c in m.modules()), 4)
        x = torch.rand(2, 3, 64, 64)
        y = m(x)
        self.assertEqual(tuple(y.shape), (2,))
        nn.functional.binary_cross_entropy_with_logits(y, torch.tensor([0., 1.])).backward()
        for c in m.modules():
            if isinstance(c, nn.Conv2d):
                self.assertTrue(torch.isfinite(c.weight.grad).all())
                self.assertGreater(c.weight.grad.abs().sum().item(), 0.)

    def test_checkpoint_reproduce_predicciones(self):
        m = CNNBreastDCEDoble().eval()
        x = torch.rand(2, 3, 256, 256)
        with torch.inference_mode():
            esperado = m(x)
            f = io.BytesIO()
            torch.save(m.state_dict(), f)
            f.seek(0)
            n = CNNBreastDCEDoble().eval()
            n.load_state_dict(torch.load(f, weights_only=True))
            torch.testing.assert_close(n(x), esperado)
