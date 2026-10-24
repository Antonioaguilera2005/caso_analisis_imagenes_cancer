"""Variante propia desde cero: dos convoluciones por bloque, solo imágenes DCE."""
from torch import nn
from modelo import CNNBreastDCE


class CNNBreastDCEDoble(nn.Module):
    def __init__(self, dropout=0.3):
        super().__init__()
        bloques = []
        entrada = 3
        for salida in (16, 32, 64, 128):
            for canales in (entrada, salida):
                bloques.extend([
                    nn.Conv2d(canales, salida, 3, padding=1, bias=False),
                    nn.BatchNorm2d(salida), nn.ReLU(inplace=True),
                ])
            bloques.append(nn.MaxPool2d(2))
            entrada = salida
        self.features = nn.Sequential(*bloques)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                  nn.Dropout(dropout), nn.Linear(128, 1))
        self.apply(CNNBreastDCE._inicializar)

    def forward(self, x):
        return self.head(self.features(x)).squeeze(1)
