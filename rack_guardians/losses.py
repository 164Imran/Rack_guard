"""
losses.py — Loss combinée physics-informed, partagée par le NODE et le DLinear.

L = L_data + lambda_phys * L_residu_physique + lambda_throttle * L_throttle

- L_data : MSE standard entre trajectoire prédite et vérité terrain.
- L_residu_physique : résidu des équations RC (die, heatsink). Ces équations sont
  entièrement déterminées par l'état [P, T_die, T_hs] (pas de charge externe
  inconnue), donc on peut les imposer comme contrainte dure sur TOUTE trajectoire
  prédite, que le modèle la connaisse ou non.
- L_throttle : reprend l'esprit de la contrainte du papier PI-DLinear :
  max(0, delta_P_prédit)^2 quand T_die dépasse le seuil de throttling (la
  puissance ne devrait pas continuer à augmenter une fois le throttling actif).

Exécutable seul : `python losses.py` fait un test sanity-check sur un batch random.
"""

import torch


def physics_residual_loss(traj, dt, thermal_params):
    """Résidu des équations RC sur une trajectoire prédite.

    traj : tenseur (batch, T, 3) -> canaux [P, T_die, T_hs]
    thermal_params : dict de tenseurs scalaires {T_amb, R1, R2, C_die, C_hs}
    """
    p = traj[:, :, 0]
    t_die = traj[:, :, 1]
    t_hs = traj[:, :, 2]

    # dérivée numérique (différence avant), un pas de moins que la trajectoire
    dtdie_dt_num = (t_die[:, 1:] - t_die[:, :-1]) / dt
    dths_dt_num = (t_hs[:, 1:] - t_hs[:, :-1]) / dt

    q_die_hs = (t_die - t_hs) / thermal_params["R1"]
    q_hs_amb = (t_hs - thermal_params["T_amb"]) / thermal_params["R2"]

    dtdie_dt_phys = (p - q_die_hs) / thermal_params["C_die"]
    dths_dt_phys = (q_die_hs - q_hs_amb) / thermal_params["C_hs"]

    resid_die = dtdie_dt_num - dtdie_dt_phys[:, :-1]
    resid_hs = dths_dt_num - dths_dt_phys[:, :-1]

    return (resid_die ** 2).mean() + (resid_hs ** 2).mean()


def throttle_loss(traj, t_thresh):
    """Pénalise max(0, delta_P)^2 quand T_die dépasse le seuil de throttling."""
    p = traj[:, :, 0]
    t_die = traj[:, :, 1]

    delta_p = p[:, 1:] - p[:, :-1]
    above_thresh = (t_die[:, 1:] > t_thresh).float()

    penalty = torch.relu(delta_p) ** 2 * above_thresh
    return penalty.mean()


def combined_loss(pred_traj, true_traj, dt, thermal_params, t_thresh,
                   lambda_phys=0.01, lambda_throttle=0.1):
    """Loss totale + dict des composantes (pour le logging)."""
    l_data = torch.nn.functional.mse_loss(pred_traj, true_traj)
    l_phys = physics_residual_loss(pred_traj, dt, thermal_params)
    l_throttle = throttle_loss(pred_traj, t_thresh)

    total = l_data + lambda_phys * l_phys + lambda_throttle * l_throttle
    components = {
        "data": l_data.item(),
        "phys": l_phys.item(),
        "throttle": l_throttle.item(),
        "total": total.item(),
    }
    return total, components


if __name__ == "__main__":
    torch.manual_seed(0)
    batch, horizon = 4, 50
    pred = torch.randn(batch, horizon, 3) * 5 + torch.tensor([50.0, 40.0, 30.0])
    true = torch.randn(batch, horizon, 3) * 5 + torch.tensor([50.0, 40.0, 30.0])

    thermal_params = {
        "T_amb": torch.tensor(20.0), "R1": torch.tensor(0.06),
        "R2": torch.tensor(0.14), "C_die": torch.tensor(45.0), "C_hs": torch.tensor(220.0),
    }
    total, comp = combined_loss(pred, true, dt=0.5, thermal_params=thermal_params, t_thresh=70.0)
    print("sanity check combined_loss:", comp)
