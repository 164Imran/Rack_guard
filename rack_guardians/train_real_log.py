"""
train_real_log.py — Test rapide : entraîne un NODE directement sur la trace réelle
`gpu_log_stress2.csv` (nvidia-smi perso), à la place des scénarios simulés/calibrés de
train.py. Script séparé et autonome : ne touche PAS à train.py/compare.py/model_node.py
(pipeline de démo principal), qui reste basé sur les scénarios calibrés (seul jeu de
données garantissant des franchissements de seuil pour la démo).

Limites assumées de ce test (documentées explicitement, cf. échange avec l'utilisateur) :
  - Pas de canal T_hs (heatsink) dans ce log -> modèle à 2 canaux [P, T_die] seulement
    (ThermalNODE2Ch ci-dessous, RC à un seul noeud die->ambiant, pas le RC à 2 noeuds
    de model_node.py).
  - fan.speed jamais lu (toujours [N/A] dans ce log).
  - Ce log ne contient AUCUN franchissement de seuil de throttling (plafonne à 61°C) :
    aucune métrique de détection de seuil n'est calculée ici, impossible par construction.
  - Une seule trace continue de 330s : les fenêtres obs/pred sont découpées par
    fenêtre glissante sur CETTE trace, avec split TEMPOREL (pas aléatoire) train/test
    pour éviter toute fuite entre fenêtres qui se chevauchent.

Exécution : `python train_real_log.py`
Sorties : checkpoints_real_log/{node2ch.pt, loss_curve.png, comparison.png}
"""

import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

CSV_PATH = "gpu_log_stress2.csv"
CKPT_DIR = "checkpoints_real_log"

OBS_DURATION = 60.0     # s
PRED_DURATION = 60.0    # s
SAMPLE_DT = 3.0         # s, sous-échantillonnage de l'historique observé (bruit réel du capteur)
WINDOW_STRIDE = 10.0    # s, pas de la fenêtre glissante
TRAIN_TEST_SPLIT_S = 150.0  # fenêtres démarrant avant -> train, après -> test (split temporel)

N_EPOCHS = 500
LR = 1e-2
N_SUBSTEPS = 4
NORM_SCALE = (50.0, 100.0)   # (P, T_die) : marge au-dessus des maxima observés (44.5W, 61°C)


def load_real_trace(path=CSV_PATH):
    """Charge la trace dense (t, P, T_die), fan.speed ignoré (toujours [N/A] dans ce log)."""
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    df["power"] = pd.to_numeric(
        df["power.draw [W]"].astype(str).str.replace("W", "", regex=False).str.strip(), errors="coerce")
    df["temp"] = pd.to_numeric(df["temperature.gpu"], errors="coerce")
    df["ts"] = pd.to_datetime(df["timestamp"].str.strip(), format="%Y/%m/%d %H:%M:%S.%f")
    df["t_s"] = (df["ts"] - df["ts"].iloc[0]).dt.total_seconds()
    return df["t_s"].to_numpy(), df["power"].to_numpy(), df["temp"].to_numpy()


def build_windows(t, p, temp, obs_duration, pred_duration, sample_dt, start_times):
    """Découpe la trace réelle en fenêtres (obs éparse, pred dense) à des instants de départ donnés."""
    dt_full = np.median(np.diff(t))
    step = max(1, int(round(sample_dt / dt_full)))
    windows = []
    for s in start_times:
        obs_mask = (t >= s) & (t < s + obs_duration)
        pred_mask = (t >= s + obs_duration) & (t < s + obs_duration + pred_duration)
        idx_obs = np.where(obs_mask)[0][::step]
        idx_pred = np.where(pred_mask)[0]
        if len(idx_obs) < 3 or len(idx_pred) < 3:
            continue
        t_obs = t[idx_obs] - s
        obs = np.stack([p[idx_obs], temp[idx_obs]], axis=-1)
        true_pred = np.stack([p[idx_pred], temp[idx_pred]], axis=-1)
        windows.append({"start": s, "t_obs": t_obs, "obs": obs, "true_pred": true_pred})
    return windows


class Encoder2Ch(nn.Module):
    def __init__(self, l_obs, hidden=32):
        super().__init__()
        in_dim = l_obs * 3  # (t_norm, P, T_die)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 2),
        )

    def forward(self, t_obs_norm, obs, norm_scale):
        batch = obs.shape[0]
        obs_norm = obs / norm_scale
        flat = torch.cat([t_obs_norm.unsqueeze(-1), obs_norm], dim=-1).reshape(batch, -1)
        correction_norm = self.net(flat)
        last_obs = obs[:, -1, :]
        return last_obs + correction_norm * norm_scale


class Coupling2Ch(nn.Module):
    def __init__(self, hidden=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, t_norm, state_norm):
        x = torch.cat([t_norm.unsqueeze(-1), state_norm], dim=-1)
        return self.net(x).squeeze(-1)


class ThermalNODE2Ch(nn.Module):
    """RC à un seul noeud (die -> ambiant directement) : pas de heatsink séparé, ce log
    n'ayant qu'un capteur de température (temperature.gpu)."""

    def __init__(self, l_obs, norm_scale, horizon_s, t_amb_init, p_eq_init, r_init=0.4, c_init=8.0):
        super().__init__()
        self.encoder = Encoder2Ch(l_obs)
        self.coupling_mlp = Coupling2Ch()
        self.horizon_s = horizon_s
        self.register_buffer("norm_scale", torch.tensor(norm_scale))
        self.T_amb = float(t_amb_init)
        self.log_R = nn.Parameter(torch.log(torch.tensor(float(r_init))))
        self.log_C = nn.Parameter(torch.log(torch.tensor(float(c_init))))
        self.log_tau_P = nn.Parameter(torch.log(torch.tensor(1.5)))
        self.P_eq = nn.Parameter(torch.tensor(float(p_eq_init)))

    def derivative(self, t_norm, state):
        p, t_die = state[:, 0], state[:, 1]
        R, C, tau_P = torch.exp(self.log_R), torch.exp(self.log_C), torch.exp(self.log_tau_P)
        q_amb = (t_die - self.T_amb) / R
        dtdie_dt = (p - q_amb) / C
        state_norm = state / self.norm_scale
        correction = self.coupling_mlp(t_norm, state_norm)
        dp_dt = (self.P_eq - p) / tau_P + correction
        return torch.stack([dp_dt, dtdie_dt], dim=-1)

    def rollout(self, state0, t0, t1, dt, n_substeps=1):
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
        t_obs_norm = t_obs / obs_duration
        state0 = self.encoder(t_obs_norm, obs, self.norm_scale)
        t1 = pred_len * dt
        return self.rollout(state0, 0.0, t1, dt, n_substeps=n_substeps)


def windows_to_tensors(windows):
    obs = torch.tensor(np.stack([w["obs"] for w in windows]), dtype=torch.float32)
    t_obs = torch.tensor(np.stack([w["t_obs"] for w in windows]), dtype=torch.float32)
    true_pred = torch.tensor(np.stack([w["true_pred"] for w in windows]), dtype=torch.float32)
    return t_obs, obs, true_pred


def main():
    os.makedirs(CKPT_DIR, exist_ok=True)
    t, p, temp = load_real_trace()
    total_duration = t[-1]
    dt_full = float(np.median(np.diff(t)))
    print(f"[train_real_log] trace réelle chargée : {len(t)} points, {total_duration:.1f}s, dt≈{dt_full:.2f}s")

    max_start = total_duration - (OBS_DURATION + PRED_DURATION)
    all_starts = np.arange(0.0, max_start + 1e-6, WINDOW_STRIDE)
    train_starts = all_starts[all_starts < TRAIN_TEST_SPLIT_S]
    test_starts = all_starts[all_starts >= TRAIN_TEST_SPLIT_S]

    train_windows = build_windows(t, p, temp, OBS_DURATION, PRED_DURATION, SAMPLE_DT, train_starts)
    test_windows = build_windows(t, p, temp, OBS_DURATION, PRED_DURATION, SAMPLE_DT, test_starts)
    print(f"[train_real_log] fenêtres : {len(train_windows)} train / {len(test_windows)} test "
          f"(split temporel à t={TRAIN_TEST_SPLIT_S:.0f}s, pas de fuite entre fenêtres chevauchantes)")

    t_obs, obs, true_pred = windows_to_tensors(train_windows)
    l_obs, pred_len = obs.shape[1], true_pred.shape[1]
    pred_dt = PRED_DURATION / pred_len

    p_idle_perso = float(p[temp < 50].mean()) if (temp < 50).any() else float(p.min())
    t_amb_perso = float(temp[p < 10].mean()) if (p < 10).any() else float(temp.min())
    p_eq_init = float(p[p > 30].mean()) if (p > 30).any() else float(p.mean())
    print(f"[train_real_log] init : T_amb={t_amb_perso:.1f}°C P_eq={p_eq_init:.1f}W "
          f"norm_scale={NORM_SCALE}")

    model = ThermalNODE2Ch(l_obs=l_obs, norm_scale=NORM_SCALE, horizon_s=PRED_DURATION,
                            t_amb_init=t_amb_perso, p_eq_init=p_eq_init)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    losses = []

    start = time.perf_counter()
    pbar = tqdm(range(N_EPOCHS), desc="[NODE-2ch]", unit="epoch")
    for epoch in pbar:
        optimizer.zero_grad()
        pred = model(t_obs, obs, pred_len=pred_len, dt=pred_dt, obs_duration=OBS_DURATION, n_substeps=N_SUBSTEPS)
        loss = torch.nn.functional.mse_loss(pred, true_pred)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        pbar.set_postfix(mse=f"{loss.item():.3f}")
    elapsed = time.perf_counter() - start
    print(f"[train_real_log] entraînement terminé en {elapsed:.2f}s ({N_EPOCHS} epochs)")

    torch.save(model.state_dict(), os.path.join(CKPT_DIR, "node2ch.pt"))

    plt.figure(figsize=(7, 4.5))
    plt.plot(losses)
    plt.yscale("log")
    plt.xlabel("epoch")
    plt.ylabel("MSE (log)")
    plt.title("NODE 2 canaux — entraîné sur gpu_log_stress2.csv (données réelles)")
    plt.tight_layout()
    plt.savefig(os.path.join(CKPT_DIR, "loss_curve.png"), dpi=120)

    # --- évaluation sur les fenêtres de test (temporellement disjointes du train) ---
    model.eval()
    t_obs_te, obs_te, true_pred_te = windows_to_tensors(test_windows)
    with torch.no_grad():
        pred_te = model(t_obs_te, obs_te, pred_len=pred_len, dt=pred_dt, obs_duration=OBS_DURATION,
                         n_substeps=N_SUBSTEPS)
    mse_p = torch.mean((pred_te[:, :, 0] - true_pred_te[:, :, 0]) ** 2).item()
    mae_p = torch.mean(torch.abs(pred_te[:, :, 0] - true_pred_te[:, :, 0])).item()
    mse_t = torch.mean((pred_te[:, :, 1] - true_pred_te[:, :, 1]) ** 2).item()
    mae_t = torch.mean(torch.abs(pred_te[:, :, 1] - true_pred_te[:, :, 1])).item()
    print(f"\n=== Test ({len(test_windows)} fenêtres, réel, aucun franchissement possible dans ce log) ===")
    print(f"MSE_P={mse_p:.2f}  MAE_P={mae_p:.2f}  MSE_T={mse_t:.2f}  MAE_T={mae_t:.2f}")

    n_test = len(test_windows)
    fig, axes = plt.subplots(n_test, 2, figsize=(10, 2.8 * n_test), squeeze=False)
    t_axis = np.arange(1, pred_len + 1) * pred_dt
    for i in range(n_test):
        ax_t, ax_p = axes[i, 0], axes[i, 1]
        ax_t.plot(t_axis, true_pred_te[i, :, 1], "k-", label="vérité T_die", linewidth=2)
        ax_t.plot(t_axis, pred_te[i, :, 1], "--", label="NODE-2ch")
        ax_t.set_ylabel("T_die (°C)")
        ax_t.set_title(f"fenêtre test start={test_windows[i]['start']:.0f}s")
        if i == 0:
            ax_t.legend(fontsize=8)
        ax_p.plot(t_axis, true_pred_te[i, :, 0], "k-", label="vérité P", linewidth=2)
        ax_p.plot(t_axis, pred_te[i, :, 0], "--", label="NODE-2ch")
        ax_p.set_ylabel("P (W)")
        if i == n_test - 1:
            ax_t.set_xlabel("temps depuis fin d'observation (s)")
            ax_p.set_xlabel("temps depuis fin d'observation (s)")
    plt.tight_layout()
    out_path = os.path.join(CKPT_DIR, "comparison.png")
    plt.savefig(out_path, dpi=110)
    print(f"[train_real_log] graphique sauvegardé : {out_path}")


if __name__ == "__main__":
    main()
