"""CNN propia: tres fases DCE, un logit; sin pesos preentrenados."""
import torch
from torch import nn


class CNNBreastDCE(nn.Module):
    def __init__(self, dropout=0.3):
        super().__init__()
        bloques = []
        entrada = 3
        for salida in (16, 32, 64, 128):
            bloques.extend([
                nn.Conv2d(entrada, salida, 3, padding=1, bias=False),
                nn.BatchNorm2d(salida), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            ])
            entrada = salida
        self.features = nn.Sequential(*bloques)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                  nn.Dropout(dropout), nn.Linear(128, 1))
        self.apply(self._inicializar)

    @staticmethod
    def _inicializar(modulo):
        if isinstance(modulo, nn.Conv2d):
            nn.init.kaiming_normal_(modulo.weight, mode="fan_out", nonlinearity="relu")
        elif isinstance(modulo, nn.Linear):
            nn.init.xavier_uniform_(modulo.weight)
            nn.init.zeros_(modulo.bias)

    def forward(self, x):
        return self.head(self.features(x)).squeeze(1)


def aumento_geometrico(x):
    """Misma transformación para PRE/EARLY/LATE; conserva sus intensidades."""
    if torch.rand(()) < 0.5:
        x = torch.flip(x, dims=(-1,))
    if torch.rand(()) < 0.5:
        x = torch.flip(x, dims=(-2,))
    return x
