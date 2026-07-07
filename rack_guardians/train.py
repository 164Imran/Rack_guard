"""
train.py — Entraîne le NODE physics-informed sur des scénarios calibrés sur données
réelles (MIT Supercloud + log nvidia-smi personnel, cf. fetch_real_data.py et
generate_realistic_data.py). Pipeline unique pour la démo : DLinear a été retiré
(décision produit : NODE seul, cf. commentaires ci-dessous) pour limiter la
complexité, la baseline naïve reste dans compare.py comme repère.

Exécution : `python train.py`
Sorties :
  - checkpoints/node.pt (poids)
  - checkpoints/loss_curves.png (courbe de loss)
  - data/scenarios.pkl (scénarios + split train/test, pour compare.py)
"""

import os
import pickle
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from generate_realistic_data import (
    T_THRESH_MIN, T_THRESH_MAX, build_windows_realistic,
    generate_realistic_scenarios, load_calibration,
)
from losses import combined_loss
from model_node import ThermalNODE

# ============================================================
# Config
# ============================================================
N_SCENARIOS = 50         # cf. revue du run précédent : 15 scénarios -> jeu de test trop
                          # pauvre pour juger la détection de throttling (1 seul cas croisé)
MIN_THROTTLED = 18       # garantit un nombre suffisant de scénarios avec franchissement réel
SEED = 42
DURATION = 300.0
DT = 0.5
OBS_DURATION = 150.0
PRED_DURATION = 150.0
SAMPLE_DT = 5.0

LAMBDA_PHYS = 0.01
LAMBDA_THROTTLE = 0.1
N_EPOCHS = 300
LR = 1e-2
N_SUBSTEPS = 4

CKPT_DIR = "checkpoints"
DATA_DIR = "data"


def prepare_batch(scenarios, rng, calib):
    """Construit les tenseurs batch (obs, t_obs, true_pred, t_thresh) à partir d'une liste de
    scénarios. t_thresh est désormais tiré par scénario (cf. generate_realistic_data.py) : on
    renvoie un tenseur (batch, 1) plutôt qu'un scalaire global, pour la throttle_loss batchée.
    p_scale_w (bruit capteur) est lui aussi par scénario (TDP du GPU tiré), pas une valeur
    MIT globale (cf. generate_realistic_data.sample_gpu_power_scale).
    """
    obs_list, t_obs_list, true_pred_list, t_thresh_list = [], [], [], []
    for s in scenarios:
        win = build_windows_realistic(s, OBS_DURATION, PRED_DURATION, SAMPLE_DT,
                                       p_scale_w=s["p_load_max"], calib=calib, rng=rng)
        obs_list.append(win["obs"])
        t_obs_list.append(win["t_obs"])
        true_pred_list.append(win["true_pred"])
        t_thresh_list.append(s["t_thresh"])

    obs = torch.tensor(np.stack(obs_list), dtype=torch.float32)
    t_obs = torch.tensor(np.stack(t_obs_list), dtype=torch.float32)
    true_pred = torch.tensor(np.stack(true_pred_list), dtype=torch.float32)
    t_thresh = torch.tensor(t_thresh_list, dtype=torch.float32).unsqueeze(-1)  # (batch, 1)
    return t_obs, obs, true_pred, t_thresh


def train_node(t_obs, obs, true_pred, l_obs, pred_len, t_thresh, norm_scale, p_eq_init):
    model = ThermalNODE(l_obs=l_obs, horizon_s=PRED_DURATION,
                         norm_scale=norm_scale, p_eq_init=p_eq_init)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    losses = []

    start = time.perf_counter()
    pbar = tqdm(range(N_EPOCHS), desc="[NODE]", unit="epoch")
    for epoch in pbar:
        optimizer.zero_grad()
        pred = model(t_obs, obs, pred_len=pred_len, dt=DT, obs_duration=OBS_DURATION, n_substeps=N_SUBSTEPS)
        total, comp = combined_loss(pred, true_pred, DT, model.thermal_params(),
                                     t_thresh, LAMBDA_PHYS, LAMBDA_THROTTLE)
        total.backward()
        optimizer.step()
        losses.append(comp["total"])
        pbar.set_postfix(total=f"{comp['total']:.3f}", data=f"{comp['data']:.3f}",
                          phys=f"{comp['phys']:.4f}", throttle=f"{comp['throttle']:.4f}")
    elapsed = time.perf_counter() - start
    print(f"  [NODE] entraînement terminé en {elapsed:.1f}s")
    return model, losses, elapsed


def main():
    os.makedirs(CKPT_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)
    rng = np.random.default_rng(SEED)

    calib = load_calibration()
    gpu_table = calib.get("gpu_tdp_table_w")
    if gpu_table:
        # norm_scale/P_eq_init sont des constantes UNIQUES du modèle (partagées par tous les
        # scénarios d'un même run) : norm_scale doit couvrir le GPU le plus puissant de la
        # table (marge 5%), P_eq_init part de la moyenne pondérée des p_load_mean par modèle
        # (cf. sample_gpu_power_scale) plutôt que d'une seule valeur MIT générique.
        p_load_max_for_norm = max(gpu_table.values())
        weights_map = calib.get("gpu_tdp_weights") or {}
        models = list(gpu_table.keys())
        w = np.array([weights_map.get(m, 1.0) for m in models])
        w = w / w.sum()
        mit_p_idle = calib.get("power_idle_mean_w") or 25.0
        mit_p_load_mean = calib.get("power_load_mean_w") or 60.0
        mit_p_load_max = calib.get("power_load_max_w") or 100.0
        mean_ratio = mit_p_load_mean / mit_p_load_max
        p_load_mean = float(np.sum(w * np.array([gpu_table[m] for m in models]))) * mean_ratio
    else:
        p_load_max_for_norm = calib.get("power_load_max_w") or 100.0
        p_load_mean = calib.get("power_load_mean_w") or 60.0
    norm_scale = (round(p_load_max_for_norm * 1.05, -1), 100.0, 100.0)  # marge 5% au-dessus du GPU le + puissant

    print(f"génération de {N_SCENARIOS} scénarios réalistes (seed={SEED}, "
          f"t_thresh~U({T_THRESH_MIN:.0f},{T_THRESH_MAX:.0f})°C par scénario, "
          f"min_throttled={MIN_THROTTLED})...")
    scenarios = generate_realistic_scenarios(n_scenarios=N_SCENARIOS, seed=SEED, duration=DURATION,
                                              dt=DT, calib=calib, min_throttled=MIN_THROTTLED)

    n_train = int(round(0.8 * N_SCENARIOS))
    perm = rng.permutation(N_SCENARIOS)
    train_idx, test_idx = perm[:n_train], perm[n_train:]
    train_scenarios = [scenarios[i] for i in train_idx]
    test_scenarios = [scenarios[i] for i in test_idx]
    n_test_throttled = sum(1 for s in test_scenarios if s["traj"][:, 1].max() > s["t_thresh"])
    print(f"split : {len(train_scenarios)} train / {len(test_scenarios)} test "
          f"({n_test_throttled}/{len(test_scenarios)} scénarios de test avec franchissement réel)")

    with open(os.path.join(DATA_DIR, "scenarios.pkl"), "wb") as f:
        pickle.dump({"train": train_scenarios, "test": test_scenarios,
                     "config": {"obs_duration": OBS_DURATION, "pred_duration": PRED_DURATION,
                                "sample_dt": SAMPLE_DT, "dt": DT,
                                "t_thresh_range": (T_THRESH_MIN, T_THRESH_MAX)}}, f)

    t_obs, obs, true_pred, t_thresh_batch = prepare_batch(train_scenarios, rng, calib)
    l_obs, pred_len = obs.shape[1], true_pred.shape[1]
    print(f"tenseurs batch : obs={tuple(obs.shape)} true_pred={tuple(true_pred.shape)}")
    print(f"échelle NODE : norm_scale={norm_scale} p_eq_init={p_load_mean:.1f}W")

    print("\n=== Entraînement NODE ===")
    node_model, node_losses, node_time = train_node(t_obs, obs, true_pred, l_obs, pred_len,
                                                      t_thresh_batch, norm_scale, p_load_mean)
    torch.save(node_model.state_dict(), os.path.join(CKPT_DIR, "node.pt"))

    with open(os.path.join(CKPT_DIR, "meta.pkl"), "wb") as f:
        pickle.dump({"l_obs": l_obs, "pred_len": pred_len, "n_substeps": N_SUBSTEPS,
                     "node_time": node_time, "norm_scale": norm_scale}, f)

    plt.figure(figsize=(8, 5))
    plt.plot(node_losses, label=f"NODE (train={node_time:.1f}s)")
    plt.yscale("log")
    plt.xlabel("epoch")
    plt.ylabel("loss totale (log)")
    plt.title("Courbe de loss — NODE (données réelles)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(CKPT_DIR, "loss_curves.png"), dpi=120)
    print(f"\nrésumé : NODE {node_time:.1f}s")
    print("poids sauvegardés dans checkpoints/, courbe dans checkpoints/loss_curves.png")


if __name__ == "__main__":
    main()
