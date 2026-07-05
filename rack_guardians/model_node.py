"""
model_node.py — Physics-Informed Neural ODE (RK4 fait maison).

Système intégré : [P, T_die, T_hs].
  - dT_die/dt, dT_hs/dt : équations RC connues, avec R1, R2, C_die, C_hs
    APPRENABLES (paramétrées en log-espace pour rester positives).
  - dP/dt : terme de relaxation appris (P_eq, tau_P) + petit MLP de correction
    qui capture la dynamique de charge/throttling non observée directement
    (c'est le "terme de couplage P<->T" que le MLP vient corriger).

Un encodeur MLP transforme l'historique éparse/bruité observé en un état
initial corrigé (P0, T_die0, T_hs0) d'où démarre l'intégration RK4.

Exécutable seul : `python model_node.py` fait un forward pass sanity-check.
"""

import time

import torch
import torch.nn as nn


class NodeEncoder(nn.Module):
    """Encode l'historique éparse/bruité (longueur fixe L_obs, 3 canaux + temps) -> état initial.

    BUG CORRIGÉ : obs (P~25-100, T~20-90) entrait brut dans les Linear+Tanh, contrairement
    à CouplingMLP qui normalise déjà state/norm_scale. Avec des entrées d'échelle ~100,
    tanh sature immédiatement -> gradients quasi nuls -> encodeur figé dès les premières
    epochs. On normalise ici par le même norm_scale, pour rester homogène avec CouplingMLP.
    """

    def __init__(self, l_obs, hidden=32):
        super().__init__()
        in_dim = l_obs * 4  # (t_norm, P, T_die, T_hs) par point d'observation
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 3),
        )

    def forward(self, t_obs_norm, obs, norm_scale):
        # obs: (batch, L_obs, 3), t_obs_norm: (batch, L_obs)
        batch = obs.shape[0]
        obs_norm = obs / norm_scale
        flat = torch.cat([t_obs_norm.unsqueeze(-1), obs_norm], dim=-1).reshape(batch, -1)
        correction_norm = self.net(flat)  # correction en espace normalisé
        last_obs = obs[:, -1, :]  # dernier point observé (brut) comme base
        return last_obs + correction_norm * norm_scale


class CouplingMLP(nn.Module):
    """Petit MLP de correction pour dP/dt, ajouté au terme de relaxation appris."""

    def __init__(self, hidden=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, t_norm, state_norm):
        x = torch.cat([t_norm.unsqueeze(-1), state_norm], dim=-1)
        return self.net(x).squeeze(-1)


class ThermalNODE(nn.Module):
    def __init__(self, l_obs, norm_scale=(250.0, 100.0, 100.0), horizon_s=150.0,
                 t_amb_init=20.0, p_eq_init=60.0):
        """norm_scale et p_eq_init doivent correspondre à l'échelle réelle de calibration.json
        (ex. p_load_max~242W -> norm_scale[0]~250, pas l'ancienne échelle nominale ~100W).
        Sans ça, l'encodeur/coupling_mlp reçoivent des états normalisés > 2 (au lieu de ~1),
        ce qui pousse les tanh vers la saturation et ralentit fortement la convergence
        (observé sur le run précédent : plateau de loss ~90 epochs avant de percer).
        """
        super().__init__()
        self.encoder = NodeEncoder(l_obs)
        self.coupling_mlp = CouplingMLP()
        self.horizon_s = horizon_s
        self.register_buffer("norm_scale", torch.tensor(norm_scale))

        # équation RC connue, R/C entraînables en log-espace (positivité garantie)
        nominal = {"R1": 0.06, "R2": 0.14, "C_die": 45.0, "C_hs": 220.0}
        self.T_amb = t_amb_init  # ambiant fixe, non appris
        self.log_R1 = nn.Parameter(torch.log(torch.tensor(nominal["R1"])))
        self.log_R2 = nn.Parameter(torch.log(torch.tensor(nominal["R2"])))
        self.log_Cdie = nn.Parameter(torch.log(torch.tensor(nominal["C_die"])))
        self.log_Chs = nn.Parameter(torch.log(torch.tensor(nominal["C_hs"])))

        # dynamique de puissance : relaxation vers un équilibre appris, initialisé
        # proche de la puissance de charge réelle (calibration.json) plutôt que d'une
        # valeur arbitraire à l'ancienne échelle (~100W).
        self.log_tau_P = nn.Parameter(torch.log(torch.tensor(1.5)))
        self.P_eq = nn.Parameter(torch.tensor(float(p_eq_init)))

    def thermal_params(self):
        """Constantes physiques courantes (pour le résidu physique dans losses.py)."""
        return {
            "T_amb": torch.tensor(self.T_amb, device=self.log_R1.device),
            "R1": torch.exp(self.log_R1), "R2": torch.exp(self.log_R2),
            "C_die": torch.exp(self.log_Cdie), "C_hs": torch.exp(self.log_Chs),
        }

    def derivative(self, t_norm, state):
        """dstate/dt, state: (batch, 3) -> [P, T_die, T_hs]."""
        p, t_die, t_hs = state[:, 0], state[:, 1], state[:, 2]
        R1, R2 = torch.exp(self.log_R1), torch.exp(self.log_R2)
        C_die, C_hs = torch.exp(self.log_Cdie), torch.exp(self.log_Chs)
        tau_P = torch.exp(self.log_tau_P)

        q_die_hs = (t_die - t_hs) / R1
        q_hs_amb = (t_hs - self.T_amb) / R2
        dtdie_dt = (p - q_die_hs) / C_die
        dths_dt = (q_die_hs - q_hs_amb) / C_hs

        state_norm = state / self.norm_scale
        correction = self.coupling_mlp(t_norm, state_norm)
        dp_dt = (self.P_eq - p) / tau_P + correction

        return torch.stack([dp_dt, dtdie_dt, dths_dt], dim=-1)

    def rollout(self, state0, t0, t1, dt, n_substeps=1):
        """Intégration RK4 de t0 à t1, renvoie la trajectoire (batch, n_steps, 3).

        n_substeps : nombre de pas RK4 internes par intervalle de sortie dt (h = dt/n_substeps).
        Avant fix : n_substeps=1 implicite (aucun sous-pas), pas RK4 = dt = 0.5s.
        """
        n_steps = int(round((t1 - t0) / dt))
        h = dt / n_substeps
        state = state0
        traj = []
        for i in range(n_steps):
            for j in range(n_substeps):
                ti = t0 + i * dt + j * h
                t_norm = torch.full((state.shape[0],), ti / self.horizon_s, device=state.device)
                t_norm_mid = torch.full((state.shape[0],), (ti + h / 2) / self.horizon_s, device=state.device)
                t_norm_end = torch.full((state.shape[0],), (ti + h) / self.horizon_s, device=state.device)

                k1 = self.derivative(t_norm, state)
                k2 = self.derivative(t_norm_mid, state + h / 2 * k1)
                k3 = self.derivative(t_norm_mid, state + h / 2 * k2)
                k4 = self.derivative(t_norm_end, state + h * k3)
                state = state + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            traj.append(state)
        return torch.stack(traj, dim=1)

    def forward(self, t_obs, obs, pred_len, dt, obs_duration, n_substeps=1):
        """t_obs/obs: historique éparse/bruité -> trajectoire prédite (batch, pred_len, 3)."""
        t_obs_norm = t_obs / obs_duration
        state0 = self.encoder(t_obs_norm, obs, self.norm_scale)
        t1 = pred_len * dt
        return self.rollout(state0, 0.0, t1, dt, n_substeps=n_substeps)


if __name__ == "__main__":
    torch.manual_seed(0)
    batch, l_obs, pred_len, dt = 4, 30, 300, 0.5
    t_obs = torch.linspace(0, 150, l_obs).unsqueeze(0).repeat(batch, 1)
    obs = torch.randn(batch, l_obs, 3) * 5 + torch.tensor([50.0, 40.0, 30.0])

    model = ThermalNODE(l_obs=l_obs)
    start = time.perf_counter()
    out = model(t_obs, obs, pred_len=pred_len, dt=dt, obs_duration=150.0)
    elapsed = time.perf_counter() - start
    print(f"sortie NODE: {out.shape} (attendu ({batch}, {pred_len}, 3)) en {elapsed:.3f}s")
