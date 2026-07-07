"""
model_dlinear.py — DLinear physics-informed (dans l'esprit du papier PI-DLinear).

Décomposition trend/seasonal de l'historique observé (moyenne mobile pour la
tendance), puis une couche linéaire PAR canal (P, T_die, T_hs) et par
composante (trend, seasonal) qui mappe directement la longueur d'observation
éparse (L_obs) vers l'horizon de prédiction dense (pred_len). Les deux
composantes sont sommées pour former la prédiction finale.

Le caractère "physics-informed" vient de la loss partagée (losses.py), pas de
l'architecture elle-même : c'est voulu, pour une comparaison équitable avec le
NODE (même input, même horizon, même fonction de coût).

Exécutable seul : `python model_dlinear.py` fait un forward pass sanity-check.
"""

import time

import torch
import torch.nn as nn


class SeriesDecomposition(nn.Module):
    """Décompose une série en tendance (moyenne mobile) + résidu saisonnier."""

    def __init__(self, kernel_size=5):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg_pool = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0)

    def forward(self, x):
        # x: (batch, L, C) -> pad de façon symétrique pour garder la longueur L
        pad_left = (self.kernel_size - 1) // 2
        pad_right = self.kernel_size - 1 - pad_left
        x_t = x.permute(0, 2, 1)  # (batch, C, L)
        x_padded = torch.cat([
            x_t[:, :, :1].repeat(1, 1, pad_left),
            x_t,
            x_t[:, :, -1:].repeat(1, 1, pad_right),
        ], dim=-1)
        trend = self.avg_pool(x_padded)  # (batch, C, L)
        trend = trend.permute(0, 2, 1)   # (batch, L, C)
        seasonal = x - trend
        return trend, seasonal


class PhysicsDLinear(nn.Module):
    """DLinear "individual" : poids séparés par canal (P, T_die, T_hs)."""

    def __init__(self, l_obs, pred_len, kernel_size=5, n_channels=3):
        super().__init__()
        self.decomp = SeriesDecomposition(kernel_size)
        self.n_channels = n_channels
        self.pred_len = pred_len

        # une couche linéaire par canal, pour chaque composante
        self.linear_trend = nn.ModuleList([nn.Linear(l_obs, pred_len) for _ in range(n_channels)])
        self.linear_seasonal = nn.ModuleList([nn.Linear(l_obs, pred_len) for _ in range(n_channels)])

    def forward(self, obs):
        # obs: (batch, L_obs, 3) -> historique éparse/bruité (même input que le NODE)
        trend, seasonal = self.decomp(obs)

        outputs = []
        for c in range(self.n_channels):
            trend_c = self.linear_trend[c](trend[:, :, c])       # (batch, pred_len)
            seasonal_c = self.linear_seasonal[c](seasonal[:, :, c])
            outputs.append(trend_c + seasonal_c)

        return torch.stack(outputs, dim=-1)  # (batch, pred_len, 3)


if __name__ == "__main__":
    torch.manual_seed(0)
    batch, l_obs, pred_len = 4, 30, 300
    obs = torch.randn(batch, l_obs, 3) * 5 + torch.tensor([50.0, 40.0, 30.0])

    model = PhysicsDLinear(l_obs=l_obs, pred_len=pred_len)
    start = time.perf_counter()
    out = model(obs)
    elapsed = time.perf_counter() - start
    print(f"sortie DLinear: {out.shape} (attendu ({batch}, {pred_len}, 3)) en {elapsed:.3f}s")
