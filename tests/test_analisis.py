"""Evita conclusiones basadas en predicciones desalineadas o agregaciones incorrectas."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from analizar_resultados import cargar_ejecucion, intervalo_auc, metricas


class AnalisisTest(unittest.TestCase):
    def preparar(self, raiz):
        samples = pd.DataFrame({"sample_id": ["a0", "a1", "b0", "b1"],
                                "patient_id": ["a0", "a1", "b0", "b1"],
                                "pCR": [0, 1, 0, 1], "split": ["train"] * 4, "fold": [0, 0, 1, 1]})
        pacientes = pd.DataFrame({"pid": samples.patient_id, "dataset": ["spy1"] * 4})
        samples[samples.fold == 0].drop(columns="split").to_csv(raiz / "particion_val.csv", index=False)
        samples[samples.fold == 1].drop(columns="split").to_csv(raiz / "particion_train.csv", index=False)
        cortes = samples[samples.fold == 0][["sample_id", "patient_id", "pCR"]].assign(prob=[0.2, 0.8])
        cortes.to_csv(raiz / "predicciones_cortes_val.csv", index=False)
        cortes.drop(columns="sample_id").to_csv(raiz / "predicciones_pacientes_mean.csv", index=False)
        for nombre, datos in {
            "config.json": {"prueba": False, "uso": "validacion_interna", "fold": 0},
            "ejecucion.json": {"epoca_elegida": 1},
            "historial.json": [{"epoca": 1, "auc": 1.0}],
        }.items():
            (raiz / nombre).write_text(json.dumps(datos), encoding="utf-8")
        return samples, pacientes

    def test_rechaza_etiqueta_cambiada(self):
        with tempfile.TemporaryDirectory() as temporal:
            raiz = Path(temporal)
            samples, pacientes = self.preparar(raiz)
            pred = pd.read_csv(raiz / "predicciones_cortes_val.csv")
            pred.loc[0, "pCR"] = 1
            pred.to_csv(raiz / "predicciones_cortes_val.csv", index=False)
            with self.assertRaisesRegex(ValueError, "etiqueta"):
                cargar_ejecucion(raiz, samples, pacientes)

    def test_rechaza_media_guardada_incorrecta(self):
        with tempfile.TemporaryDirectory() as temporal:
            raiz = Path(temporal)
            samples, pacientes = self.preparar(raiz)
            pred = pd.read_csv(raiz / "predicciones_pacientes_mean.csv")
            pred.loc[0, "prob"] = 0.9
            pred.to_csv(raiz / "predicciones_pacientes_mean.csv", index=False)
            with self.assertRaisesRegex(ValueError, "media"):
                cargar_ejecucion(raiz, samples, pacientes)

    def test_f1_macro_no_oculta_clase_minoritaria(self):
        r = metricas([0, 0, 0, 1], [0.1, 0.2, 0.3, 0.4], 0.5)
        self.assertEqual(r["accuracy"], 0.75)
        self.assertEqual(r["sensibilidad"], 0)
        self.assertEqual(r["f1_pcr"], 0)
        self.assertAlmostEqual(r["f1_macro"], 3 / 7)
        self.assertIsNone(r["precision_pcr"])

    def test_bootstrap_remuestrea_pacientes(self):
        intervalo = intervalo_auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9], 20)
        np.testing.assert_allclose(intervalo, [1.0, 1.0])


if __name__ == "__main__":
    unittest.main()
