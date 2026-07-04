"""
compare.py — Compare NODE, DLinear physics-informed et baseline naïve sur les scénarios de test.

Exécution : `python compare.py` (nécessite d'avoir lancé `python train.py` avant).
Sorties : tableau récapitulatif en console + checkpoints/comparison.png
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
from model_dlinear import PhysicsDLinear
from model_node import ThermalNODE

CKPT_DIR = "checkpoints"
DATA_DIR = "data"


def naive_polynomial_baseline(t_obs, obs, pred_len, dt, degree=2):
    """Extrapolation polynomiale simple par canal, à partir de l'historique observé."""
    n_scenarios = obs.shape[0]
    t_pred = np.arange(1, pred_len + 1) * dt  # temps relatifs après la fin de l'observation
    preds = np.zeros((n_scenarios, pred_len, 3))
    t0 = time.perf_counter()
    for i in range(n_scenarios):
        t_hist = t_obs[i]
        for c in range(3):
            coeffs = np.polyfit(t_hist, obs[i, :, c], deg=degree)
            preds[i, :, c] = np.polyval(coeffs, t_hist[-1] + t_pred)
    elapsed = time.perf_counter() - t0
    return preds, elapsed


def throttle_crossing_time(traj, t_thresh, dt):
    """Premier instant (en s, relatif au début de la prédiction) où T_die dépasse le seuil, ou None."""
    t_die = traj[:, 1]
    idx = np.where(t_die > t_thresh)[0]
    if len(idx) == 0:
        return None
    return idx[0] * dt


def summarize_errors(name, pred, true, dt, t_thresh, inference_time):
    """Calcule MSE/MAE (température, puissance) + erreur de franchissement de seuil."""
    mse_p = np.mean((pred[:, :, 0] - true[:, :, 0]) ** 2)
    mae_p = np.mean(np.abs(pred[:, :, 0] - true[:, :, 0]))
    # "température" = moyenne des deux canaux T_die et T_hs
    mse_t = np.mean((pred[:, :, 1:3] - true[:, :, 1:3]) ** 2)
    mae_t = np.mean(np.abs(pred[:, :, 1:3] - true[:, :, 1:3]))

    crossing_errors = []
    for i in range(pred.shape[0]):
        t_true = throttle_crossing_time(true[i], t_thresh, dt)
        t_pred = throttle_crossing_time(pred[i], t_thresh, dt)
        if t_true is not None and t_pred is not None:
            crossing_errors.append(abs(t_true - t_pred))
        elif t_true is None and t_pred is None:
            continue  # pas de throttling ni prédit ni réel : rien à comparer
        else:
            crossing_errors.append(None)  # un throttling manqué ou halluciné

    n_missed = sum(1 for e in crossing_errors if e is None)
    valid_errors = [e for e in crossing_errors if e is not None]
    mean_crossing_err = np.mean(valid_errors) if valid_errors else float("nan")

    return {
        "name": name,
        "mse_power": mse_p, "mae_power": mae_p,
        "mse_temp": mse_t, "mae_temp": mae_t,
        "crossing_err_s": mean_crossing_err,
        "crossing_missed": n_missed,
        "crossing_total": len(crossing_errors),
        "inference_time_s": inference_time,
    }


def print_summary_table(results):
    headers = ["Modèle", "MSE_P", "MAE_P", "MSE_T", "MAE_T", "Err. seuil (s)", "Ratés", "T. inférence (s)"]
    rows = []
    for r in results:
        rows.append([
            r["name"], f"{r['mse_power']:.2f}", f"{r['mae_power']:.2f}",
            f"{r['mse_temp']:.2f}", f"{r['mae_temp']:.2f}",
            f"{r['crossing_err_s']:.1f}" if not np.isnan(r["crossing_err_s"]) else "N/A",
            f"{r['crossing_missed']}/{r['crossing_total']}",
            f"{r['inference_time_s']:.4f}",
        ])

    widths = [max(len(h), *(len(row[i]) for row in rows)) for i, h in enumerate(headers)]
    line = " | ".join(h.ljust(w) for h, w in zip(headers, widths))
    print(line)
    print("-" * len(line))
    for row in rows:
        print(" | ".join(c.ljust(w) for c, w in zip(row, widths)))


def main():
    with open(os.path.join(DATA_DIR, "scenarios.pkl"), "rb") as f:
        data = pickle.load(f)
    test_scenarios = data["test"]
    cfg = data["config"]

    with open(os.path.join(CKPT_DIR, "meta.pkl"), "rb") as f:
        meta = pickle.load(f)
    l_obs, pred_len = meta["l_obs"], meta["pred_len"]

    rng = np.random.default_rng(123)  # bruit d'observation différent de l'entraînement
    obs_list, t_obs_list, true_pred_list = [], [], []
    for s in test_scenarios:
        win = simulator.build_windows(s, cfg["obs_duration"], cfg["pred_duration"], cfg["sample_dt"], rng=rng)
        obs_list.append(win["obs"])
        t_obs_list.append(win["t_obs"])
        true_pred_list.append(win["true_pred"])

    obs_np = np.stack(obs_list)
    t_obs_np = np.stack(t_obs_list)
    true_pred = np.stack(true_pred_list)

    obs = torch.tensor(obs_np, dtype=torch.float32)
    t_obs = torch.tensor(t_obs_np, dtype=torch.float32)

    # --- NODE ---
    node_model = ThermalNODE(l_obs=l_obs, horizon_s=cfg["pred_duration"])
    node_model.load_state_dict(torch.load(os.path.join(CKPT_DIR, "node.pt")))
    node_model.eval()
    with torch.no_grad():
        t0 = time.perf_counter()
        node_pred = node_model(t_obs, obs, pred_len=pred_len, dt=cfg["dt"], obs_duration=cfg["obs_duration"],
                                n_substeps=meta.get("n_substeps", 1))
        node_time = time.perf_counter() - t0
    node_pred_np = node_pred.numpy()

    # --- DLinear ---
    dlinear_model = PhysicsDLinear(l_obs=l_obs, pred_len=pred_len)
    dlinear_model.load_state_dict(torch.load(os.path.join(CKPT_DIR, "dlinear.pt")))
    dlinear_model.eval()
    with torch.no_grad():
        t0 = time.perf_counter()
        dlinear_pred = dlinear_model(obs)
        dlinear_time = time.perf_counter() - t0
    dlinear_pred_np = dlinear_pred.numpy()

    # --- baseline naïve (extrapolation polynomiale) ---
    naive_pred, naive_time = naive_polynomial_baseline(t_obs_np, obs_np, pred_len, cfg["dt"])

    results = [
        summarize_errors("NODE (physics-informed)", node_pred_np, true_pred, cfg["dt"], simulator.T_THRESH, node_time),
        summarize_errors("DLinear (physics-informed)", dlinear_pred_np, true_pred, cfg["dt"], simulator.T_THRESH, dlinear_time),
        summarize_errors("Baseline polynomiale (naïve)", naive_pred, true_pred, cfg["dt"], simulator.T_THRESH, naive_time),
    ]

    print(f"\n=== Comparaison sur {len(test_scenarios)} scénarios de test ===\n")
    print_summary_table(results)

    best = min(results[:2], key=lambda r: r["mse_temp"])
    print(f"\n>>> Meilleur modèle sur MSE température : {best['name']}")

    # --- graphique comparatif sur le premier scénario de test ---
    i = 0
    t_axis = np.arange(1, pred_len + 1) * cfg["dt"]
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

    axes[0].plot(t_axis, true_pred[i, :, 1], "k-", label="vérité terrain T_die", linewidth=2)
    axes[0].plot(t_axis, node_pred_np[i, :, 1], "--", label="NODE")
    axes[0].plot(t_axis, dlinear_pred_np[i, :, 1], "--", label="DLinear")
    axes[0].plot(t_axis, naive_pred[i, :, 1], ":", label="Baseline naïve")
    axes[0].axhline(simulator.T_THRESH, color="r", linestyle=":", alpha=0.5, label="seuil throttling")
    axes[0].set_ylabel("T_die (°C)")
    axes[0].legend(fontsize=8)
    axes[0].set_title(f"Scénario test #{test_scenarios[i]['id']} (profil={test_scenarios[i]['profile']})")

    axes[1].plot(t_axis, true_pred[i, :, 0], "k-", label="vérité terrain P", linewidth=2)
    axes[1].plot(t_axis, node_pred_np[i, :, 0], "--", label="NODE")
    axes[1].plot(t_axis, dlinear_pred_np[i, :, 0], "--", label="DLinear")
    axes[1].plot(t_axis, naive_pred[i, :, 0], ":", label="Baseline naïve")
    axes[1].set_ylabel("P (W)")
    axes[1].set_xlabel("temps depuis fin d'observation (s)")
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    out_path = os.path.join(CKPT_DIR, "comparison.png")
    plt.savefig(out_path, dpi=120)
    print(f"graphique sauvegardé : {out_path}")


if __name__ == "__main__":
    main()
