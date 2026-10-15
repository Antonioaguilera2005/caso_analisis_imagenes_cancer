"""Comprueba que el cierre y las pruebas técnicas no permiten abrir test accidentalmente."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from finalizar import (cargar_receta, evaluar_test, sha, sha_modelo, validar_checkpoint_final)


def receta_ejemplo():
    return dict(version=1, modo="cnn_unica_todo_train", agregacion="mean", canales=["PRE", "EARLY", "LATE"],
                normalizacion="float32 / 255", perdida="normal", epocas=5, umbral=0.33,
                pacientes_train=1097, sha256_modelo_texto=sha_modelo())


class CierreTest(unittest.TestCase):
    def test_rechaza_receta_modificada(self):
        with tempfile.TemporaryDirectory() as temporal:
            ruta = Path(temporal) / "receta.json"
            ruta.write_text(json.dumps(receta_ejemplo()), encoding="utf-8")
            ruta.with_suffix(".sha256").write_text(sha(ruta), encoding="utf-8")
            r = cargar_receta(ruta)
            r["umbral"] = 0.7
            ruta.write_text(json.dumps(r), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "modificado"):
                cargar_receta(ruta)

    def test_rechaza_epoca_no_fijada_por_cv(self):
        cp = dict(tipo="cnn_final", prueba=False, receta=receta_ejemplo(), epoca=4, pacientes_train=1097)
        with self.assertRaisesRegex(ValueError, "última época"):
            validar_checkpoint_final(cp)

    def test_rechaza_modelo_de_fold(self):
        with self.assertRaisesRegex(ValueError, "modelo final"):
            validar_checkpoint_final(dict(state_dict={}, epoca=3))

    def test_prueba_tecnica_no_accede_a_test(self):
        with tempfile.TemporaryDirectory() as temporal:
            raiz = Path(temporal)
            torch.save(dict(tipo="cnn_final", prueba=True), raiz / "prueba.pt")
            (raiz / "artefacto.json").write_text(json.dumps(dict(checkpoint="prueba.pt", sha256_checkpoint=sha(raiz/"prueba.pt"))), encoding="utf-8")
            args = SimpleNamespace(modelo=raiz, datos=raiz/"dataset_inexistente", salida=raiz/"salida", dispositivo="cpu")
            with patch("finalizar.uc.conjunto_test") as acceso:
                with self.assertRaisesRegex(ValueError, "modelo final"):
                    evaluar_test(args)
                acceso.assert_not_called()
            self.assertFalse(args.salida.exists())


if __name__ == "__main__":
    unittest.main()
