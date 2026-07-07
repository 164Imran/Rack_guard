"""
compare.py — Compare le NODE physics-informed à une baseline polynomiale naïve sur les
scénarios de test (calibrés sur données réelles, cf. train.py/generate_realistic_data.py).
DLinear a été retiré (décision produit : NODE seul pour la démo) ; la baseline naïve
reste comme repère de fond.

Exécution : `python compare.py` (nécessite d'avoir lancé `python train.py` avant).
Sorties : tableau récapitulatif en console + checkpoints/comparison.png (tous les
scénarios de test, pas un seul, pour juger la diversité des cas).
"""

import os
import pickle
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from generate_realistic_data import build_windows_realistic, load_calibration
from model_node import ThermalNODE

CKPT_DIR = "checkpoints"
DATA_DIR = "data"


def naive_polynomial_baseline(t_obs, obs, pred_len, dt, degree=2):
    """Extrapolation polynomiale simple par canal, à partir de l'historique observé."""
    n_scenarios = obs.shape[0]
    t_pred = np.arange(1, pred_len + 1) * dt
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


def summarize_errors(name, pred, true, dt, t_thresh_list, inference_time):
    """Calcule MSE/MAE (température, puissance) + erreur de franchissement de seuil.

    t_thresh_list : seuil de throttling PAR scénario (tiré aléatoirement, cf.
    generate_realistic_data.py) -- un scalaire global ne serait plus correct.
    """
    mse_p = np.mean((pred[:, :, 0] - true[:, :, 0]) ** 2)
    mae_p = np.mean(np.abs(pred[:, :, 0] - true[:, :, 0]))
    mse_t = np.mean((pred[:, :, 1:3] - true[:, :, 1:3]) ** 2)
    mae_t = np.mean(np.abs(pred[:, :, 1:3] - true[:, :, 1:3]))

    crossing_errors = []
    n_true_crossings = 0
    n_detected = 0
    for i in range(pred.shape[0]):
        t_thresh = t_thresh_list[i]
        t_true = throttle_crossing_time(true[i], t_thresh, dt)
        t_pred = throttle_crossing_time(pred[i], t_thresh, dt)
        if t_true is not None:
            n_true_crossings += 1
            if t_pred is not None:
                n_detected += 1
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
        "n_true_crossings": n_true_crossings,
        "n_detected": n_detected,
        "inference_time_s": inference_time,
    }


def print_summary_table(results, n_test):
    headers = ["Modèle", "MSE_P", "MAE_P", "MSE_T", "MAE_T", "Err. seuil (s)", "Détectés", "T. inférence (s)"]
    rows = []
    for r in results:
        rows.append([
            r["name"], f"{r['mse_power']:.2f}", f"{r['mae_power']:.2f}",
            f"{r['mse_temp']:.2f}", f"{r['mae_temp']:.2f}",
            f"{r['crossing_err_s']:.1f}" if not np.isnan(r["crossing_err_s"]) else "N/A",
            f"{r['n_detected']}/{r['n_true_crossings']}",
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
    t_thresh_list = [s["t_thresh"] for s in test_scenarios]  # seuil tiré par scénario

    calib = load_calibration()

    with open(os.path.join(CKPT_DIR, "meta.pkl"), "rb") as f:
        meta = pickle.load(f)
    l_obs, pred_len = meta["l_obs"], meta["pred_len"]

    rng = np.random.default_rng(123)  # bruit d'observation différent de l'entraînement
    obs_list, t_obs_list, true_pred_list = [], [], []
    for s in test_scenarios:
        win = build_windows_realistic(s, cfg["obs_duration"], cfg["pred_duration"], cfg["sample_dt"],
                                       p_scale_w=s["p_load_max"], calib=calib, rng=rng)
        obs_list.append(win["obs"])
        t_obs_list.append(win["t_obs"])
        true_pred_list.append(win["true_pred"])

    obs_np, t_obs_np, true_pred = np.stack(obs_list), np.stack(t_obs_list), np.stack(true_pred_list)
    obs, t_obs = torch.tensor(obs_np, dtype=torch.float32), torch.tensor(t_obs_np, dtype=torch.float32)

    # --- NODE ---
    node_model = ThermalNODE(l_obs=l_obs, horizon_s=cfg["pred_duration"], norm_scale=meta["norm_scale"])
    node_model.load_state_dict(torch.load(os.path.join(CKPT_DIR, "node.pt")))
    node_model.eval()
    with torch.no_grad():
        t0 = time.perf_counter()
        node_pred = node_model(t_obs, obs, pred_len=pred_len, dt=cfg["dt"], obs_duration=cfg["obs_duration"],
                                n_substeps=meta.get("n_substeps", 1))
        node_time = time.perf_counter() - t0
    node_pred_np = node_pred.numpy()

    # --- baseline naïve (extrapolation polynomiale) ---
    naive_pred, naive_time = naive_polynomial_baseline(t_obs_np, obs_np, pred_len, cfg["dt"])

    results = [
        summarize_errors("NODE (physics-informed)", node_pred_np, true_pred, cfg["dt"], t_thresh_list, node_time),
        summarize_errors("Baseline polynomiale (naïve)", naive_pred, true_pred, cfg["dt"], t_thresh_list, naive_time),
    ]

    print(f"\n=== Comparaison sur {len(test_scenarios)} scénarios de test ===\n")
    print_summary_table(results, len(test_scenarios))

    print(f"\n>>> Franchissements réels dans le jeu de test : {results[0]['n_true_crossings']}/{len(test_scenarios)}")
    print(f">>> NODE en détecte : {results[0]['n_detected']}/{results[0]['n_true_crossings']}")
    print(f">>> Baseline en détecte : {results[1]['n_detected']}/{results[1]['n_true_crossings']}")

    # --- graphique pour TOUS les scénarios de test (seul vrai benchmark visuel) ---
    n_test = len(test_scenarios)
    fig, axes = plt.subplots(n_test, 2, figsize=(11, 3.0 * n_test), squeeze=False)
    t_axis = np.arange(1, pred_len + 1) * cfg["dt"]

    for i in range(n_test):
        ax_t, ax_p = axes[i, 0], axes[i, 1]
        ax_t.plot(t_axis, true_pred[i, :, 1], "k-", label="vérité T_die", linewidth=2)
        ax_t.plot(t_axis, node_pred_np[i, :, 1], "--", label="NODE")
        ax_t.plot(t_axis, naive_pred[i, :, 1], ":", label="Baseline naïve")
        ax_t.axhline(t_thresh_list[i], color="r", linestyle=":", alpha=0.5)
        ax_t.set_ylabel("T_die (°C)")
        ax_t.set_title(f"#{test_scenarios[i]['id']} profil={test_scenarios[i]['profile']} "
                        f"amb={test_scenarios[i].get('amb_baseline', float('nan')):.1f}°C "
                        f"GPU={test_scenarios[i].get('gpu_model', '?')}")
        if i == 0:
            ax_t.legend(fontsize=7)

        ax_p.plot(t_axis, true_pred[i, :, 0], "k-", label="vérité P", linewidth=2)
        ax_p.plot(t_axis, node_pred_np[i, :, 0], "--", label="NODE")
        ax_p.plot(t_axis, naive_pred[i, :, 0], ":", label="Baseline naïve")
        ax_p.set_ylabel("P (W)")
        if i == n_test - 1:
            ax_t.set_xlabel("temps depuis fin d'observation (s)")
            ax_p.set_xlabel("temps depuis fin d'observation (s)")

    plt.tight_layout()
    out_path = os.path.join(CKPT_DIR, "comparison.png")
    plt.savefig(out_path, dpi=110)
    print(f"graphique sauvegardé : {out_path}")


if __name__ == "__main__":
    main()
