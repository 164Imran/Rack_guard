"""
train.py — Entraîne le NODE et le DLinear physics-informed sur les mêmes scénarios.

Exécution : `python train.py`
Sorties :
  - checkpoints/node.pt, checkpoints/dlinear.pt (poids)
  - checkpoints/loss_curves.png (courbes de loss des deux modèles)
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

import simulator
from losses import combined_loss
from model_dlinear import PhysicsDLinear
from model_node import ThermalNODE

# ============================================================
# Config
# ============================================================
N_SCENARIOS = 15          # 15 scénarios : assez pour couvrir les 4 profils x incident on/off
                           # tout en restant entraînable en quelques dizaines de secondes
SEED = 42
DURATION = 300.0          # s, durée totale simulée par scénario
DT = 0.5                  # s, pas de temps dense
OBS_DURATION = 150.0      # s, fenêtre d'historique éparse observée
PRED_DURATION = 150.0     # s, horizon de prédiction dense
SAMPLE_DT = 5.0           # s, période d'échantillonnage éparse (capteurs réels)

LAMBDA_PHYS = 0.01        # poids modeste : le résidu physique a une échelle numérique
                          # bien plus grande que la MSE, on l'atténue pour ne pas dominer
LAMBDA_THROTTLE = 0.1     # poids plus fort : la contrainte de throttling est rare
                          # (peu de pas au-dessus du seuil) donc on la renforce
N_EPOCHS = 300
LR = 1e-2
N_SUBSTEPS = 4            # TEST étape 4 : sous-pas RK4 internes par intervalle de sortie
                          # (h = DT/N_SUBSTEPS = 0.125s au lieu de 0.5s), pour vérifier
                          # si un pas trop grossier déstabilise l'intégration du NODE

CKPT_DIR = "checkpoints"
DATA_DIR = "data"


def prepare_batch(scenarios, rng):
    """Construit les tenseurs batch (obs, t_obs, true_pred) à partir d'une liste de scénarios."""
    obs_list, t_obs_list, true_pred_list = [], [], []
    for s in scenarios:
        win = simulator.build_windows(s, OBS_DURATION, PRED_DURATION, SAMPLE_DT, rng=rng)
        obs_list.append(win["obs"])
        t_obs_list.append(win["t_obs"])
        true_pred_list.append(win["true_pred"])

    obs = torch.tensor(np.stack(obs_list), dtype=torch.float32)
    t_obs = torch.tensor(np.stack(t_obs_list), dtype=torch.float32)
    true_pred = torch.tensor(np.stack(true_pred_list), dtype=torch.float32)
    return t_obs, obs, true_pred


def train_node(t_obs, obs, true_pred, l_obs, pred_len):
    model = ThermalNODE(l_obs=l_obs, horizon_s=PRED_DURATION)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    losses = []

    start = time.perf_counter()
    for epoch in range(N_EPOCHS):
        optimizer.zero_grad()
        pred = model(t_obs, obs, pred_len=pred_len, dt=DT, obs_duration=OBS_DURATION, n_substeps=N_SUBSTEPS)
        total, comp = combined_loss(pred, true_pred, DT, model.thermal_params(),
                                     simulator.T_THRESH, LAMBDA_PHYS, LAMBDA_THROTTLE)
        total.backward()
        optimizer.step()
        losses.append(comp["total"])
        if epoch % 50 == 0 or epoch == N_EPOCHS - 1:
            print(f"  [NODE] epoch {epoch:4d} | total={comp['total']:.3f} | "
                  f"data={comp['data']:.3f} | phys={comp['phys']:.4f} | throttle={comp['throttle']:.4f}")
    elapsed = time.perf_counter() - start
    print(f"  [NODE] entraînement terminé en {elapsed:.1f}s")
    return model, losses, elapsed


def train_dlinear(t_obs, obs, true_pred, l_obs, pred_len):
    model = PhysicsDLinear(l_obs=l_obs, pred_len=pred_len)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    losses = []

    # constantes physiques nominales fixes (le DLinear n'a pas de paramètres RC internes)
    thermal_params = {k: torch.tensor(v) for k, v in simulator.THERMAL_PARAMS_NOMINAL.items()}

    start = time.perf_counter()
    for epoch in range(N_EPOCHS):
        optimizer.zero_grad()
        pred = model(obs)
        total, comp = combined_loss(pred, true_pred, DT, thermal_params,
                                     simulator.T_THRESH, LAMBDA_PHYS, LAMBDA_THROTTLE)
        total.backward()
        optimizer.step()
        losses.append(comp["total"])
        if epoch % 50 == 0 or epoch == N_EPOCHS - 1:
            print(f"  [DLinear] epoch {epoch:4d} | total={comp['total']:.3f} | "
                  f"data={comp['data']:.3f} | phys={comp['phys']:.4f} | throttle={comp['throttle']:.4f}")
    elapsed = time.perf_counter() - start
    print(f"  [DLinear] entraînement terminé en {elapsed:.1f}s")
    return model, losses, elapsed


def main():
    os.makedirs(CKPT_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)
    rng = np.random.default_rng(SEED)

    print(f"génération de {N_SCENARIOS} scénarios (seed={SEED})...")
    scenarios = simulator.generate_scenarios(n_scenarios=N_SCENARIOS, seed=SEED,
                                              duration=DURATION, dt=DT)

    # split 80/20, scénarios de test jamais vus à l'entraînement
    n_train = int(round(0.8 * N_SCENARIOS))
    perm = rng.permutation(N_SCENARIOS)
    train_idx, test_idx = perm[:n_train], perm[n_train:]
    train_scenarios = [scenarios[i] for i in train_idx]
    test_scenarios = [scenarios[i] for i in test_idx]
    print(f"split : {len(train_scenarios)} train / {len(test_scenarios)} test")

    with open(os.path.join(DATA_DIR, "scenarios.pkl"), "wb") as f:
        pickle.dump({"train": train_scenarios, "test": test_scenarios,
                     "config": {"obs_duration": OBS_DURATION, "pred_duration": PRED_DURATION,
                                "sample_dt": SAMPLE_DT, "dt": DT}}, f)

    t_obs, obs, true_pred = prepare_batch(train_scenarios, rng)
    l_obs, pred_len = obs.shape[1], true_pred.shape[1]
    print(f"tenseurs batch : obs={tuple(obs.shape)} true_pred={tuple(true_pred.shape)}")

    print("\n=== Entraînement NODE ===")
    node_model, node_losses, node_time = train_node(t_obs, obs, true_pred, l_obs, pred_len)
    torch.save(node_model.state_dict(), os.path.join(CKPT_DIR, "node.pt"))

    print("\n=== Entraînement DLinear ===")
    dlinear_model, dlinear_losses, dlinear_time = train_dlinear(t_obs, obs, true_pred, l_obs, pred_len)
    torch.save(dlinear_model.state_dict(), os.path.join(CKPT_DIR, "dlinear.pt"))

    with open(os.path.join(CKPT_DIR, "meta.pkl"), "wb") as f:
        pickle.dump({"l_obs": l_obs, "pred_len": pred_len, "n_substeps": N_SUBSTEPS,
                     "node_time": node_time, "dlinear_time": dlinear_time}, f)

    plt.figure(figsize=(8, 5))
    plt.plot(node_losses, label=f"NODE (train={node_time:.1f}s)")
    plt.plot(dlinear_losses, label=f"DLinear (train={dlinear_time:.1f}s)")
    plt.yscale("log")
    plt.xlabel("epoch")
    plt.ylabel("loss totale (log)")
    plt.title("Courbes de loss — NODE vs DLinear physics-informed")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(CKPT_DIR, "loss_curves.png"), dpi=120)
    print(f"\nrésumé : NODE {node_time:.1f}s | DLinear {dlinear_time:.1f}s")
    print("poids sauvegardés dans checkpoints/, courbes dans checkpoints/loss_curves.png")


if __name__ == "__main__":
    main()
