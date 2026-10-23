"""Edad y menopausia: preprocesado ajustado SOLO en pacientes de train del fold."""
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset
from modelo import CNNBreastDCE


def ajustar_clinica(pacientes):
    if pacientes.pid.duplicated().any() or set(pacientes.split) != {"train"}:
        raise ValueError("Ajustar con pacientes train únicos")
    edad = pd.to_numeric(pacientes.age, errors="raise")
    meno = pd.to_numeric(pacientes.menopause, errors="raise")
    if edad.notna().sum() == 0 or meno.notna().sum() == 0:
        raise ValueError("No hay observaciones para ajustar la clínica")
    if not meno.dropna().isin([0, 1]).all() or not np.isfinite(edad.dropna()).all():
        raise ValueError("Valores clínicos no válidos")
    mediana = float(edad.median())
    completa = edad.fillna(mediana)
    return dict(edad_mediana=mediana, edad_media=float(completa.mean()),
                edad_std=float(completa.std(ddof=0)) or 1.0,
                menopausia_moda=float(meno.mode().iloc[0]),
                variables=["edad_z", "edad_ausente", "menopausia", "menopausia_ausente"],
                pacientes_ajuste=sorted(pacientes.pid.tolist()))


def transformar_clinica(pacientes, estado):
    edad = pd.to_numeric(pacientes.age, errors="raise")
    meno = pd.to_numeric(pacientes.menopause, errors="raise")
    if not meno.dropna().isin([0, 1]).all() or not np.isfinite(edad.dropna()).all():
        raise ValueError("Valores clínicos no válidos")
    x = np.column_stack([
        (edad.fillna(estado["edad_mediana"]) - estado["edad_media"]) / estado["edad_std"],
        edad.isna().astype(float), meno.fillna(estado["menopausia_moda"]),
        meno.isna().astype(float)]).astype(np.float32)
    if not np.isfinite(x).all():
        raise ValueError("Clínica transformada no finita")
    return x


class DatasetClinico(Dataset):
    def __init__(self, imagenes, pacientes, estado):
        self.imagenes = imagenes
        ids = imagenes.filas[["patient_id"]]
        unidos = ids.merge(pacientes, left_on="patient_id", right_on="pid", how="left", validate="many_to_one", sort=False)
        if unidos.pid.isna().any() or unidos.patient_id.tolist() != ids.patient_id.tolist():
            raise ValueError("Clínica incompleta o desordenada")
        self.clinica = torch.from_numpy(transformar_clinica(unidos, estado))

    def __len__(self):
        return len(self.imagenes)

    def __getitem__(self, i):
        x, y = self.imagenes[i]
        return (x, self.clinica[i]), y


class CNNConClinica(nn.Module):
    """Mismo extractor de 4 convoluciones; fusión tardía con cuatro valores clínicos."""
    def __init__(self, dropout=0.3):
        super().__init__()
        base = CNNBreastDCE(dropout)
        self.features = base.features
        self.pool = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(132, 1))
        self.head.apply(CNNBreastDCE._inicializar)

    def forward(self, x):
        imagen, clinica = x
        return self.head(torch.cat([self.pool(self.features(imagen)), clinica], dim=1)).squeeze(1)
