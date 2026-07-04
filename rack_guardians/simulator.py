"""
simulator.py — Simulateur de télémétrie GPU (puissance + températures couplées).

Système à 3 états couplés : P (puissance, W), T_die (°C), T_hs (°C, heatsink).
Couplage bidirectionnel :
  - P -> T : dissipation thermique classique (réseau RC 2 noeuds).
  - T -> P : throttling (P chute quand T_die dépasse un seuil) + fuite (leakage)
    qui augmente légèrement P avec T.

Exécutable seul : `python simulator.py` génère quelques scénarios et affiche un résumé.
"""

import numpy as np

# ============================================================
# CALIBRATION RÉELLE (ordre de grandeur)
# Source : MIT Lincoln Laboratory Supercomputing Center — "MIT Supercloud
# Dataset" (télémétrie GPU datacenter, power draw / die temp / utilisation).
# On ne vise pas une réplique exacte du dataset (temps de hackathon limité),
# seulement le bon ordre de grandeur pour que la dynamique simulée soit
# physiquement plausible :
#   - Puissance idle GPU ~ 25 W
#   - Puissance pic (charge soutenue) ~ 100 W / GPU
#   - Température idle ~ 25 °C (proche ambiant, refroidissement actif)
# ============================================================
P_IDLE = 25.0          # W
P_PEAK = 100.0         # W (charge soutenue)
P_BURST = 120.0        # W (pics courts, > P_PEAK)
T_AMB = 20.0           # °C, température ambiante datacenter

# Réseau RC thermique (die -> heatsink -> ambiant)
R1_DIE_HS = 0.06       # °C/W, résistance thermique die -> heatsink
R2_HS_AMB = 0.14       # °C/W, résistance thermique heatsink -> ambiant
C_DIE = 45.0           # J/°C, capacité thermique du die (faible inertie, réagit vite)
C_HS = 220.0           # J/°C, capacité thermique du heatsink (inertie plus grande)

# Dynamique de la puissance (contrôleur d'alimentation, pas instantané)
TAU_P = 1.5            # s, constante de temps de réponse de P vers sa cible

# Throttling thermique
T_THRESH = 70.0        # °C, seuil de déclenchement du throttling (marge vs 83-90°C réel,
                       # abaissé pour observer le phénomène sur une fenêtre de simulation courte)
K_THROTTLE = 0.05      # 1/°C, pente de réduction de puissance au-dessus du seuil
P_MIN_FRAC = 0.4       # fraction minimale de puissance conservée même en throttling fort

# Courant de fuite (leakage) : croît légèrement avec T_die
LEAK_COEF = 0.03       # W/°C au-dessus de l'ambiant

# Incident : dégradation ventilateur -> réduction de l'efficacité de refroidissement
FAN_DEGRADATION_FACTOR = 5.0  # multiplie R2 (heatsink->ambiant) pendant l'incident


def load_profile(t, profile="ramp", duration=300.0):
    """Puissance demandée (avant throttling/leakage) au temps t, selon le profil de charge."""
    if profile == "idle":
        return P_IDLE
    if profile == "ramp":
        ramp_end = duration * 0.4
        if t < ramp_end:
            return P_IDLE + (P_PEAK - P_IDLE) * (t / ramp_end)
        return P_PEAK
    if profile == "sustained":
        ramp_end = duration * 0.15
        if t < ramp_end:
            return P_IDLE + (P_PEAK - P_IDLE) * (t / ramp_end)
        return P_PEAK
    if profile == "burst":
        # baseline idle avec pics courts périodiques
        period = 40.0
        burst_len = 8.0
        phase = t % period
        if phase < burst_len:
            return P_BURST
        return P_IDLE
    raise ValueError(f"profil inconnu: {profile}")


def throttle_factor(t_die):
    """Fraction de puissance conservée en fonction de T_die (1.0 = pas de throttling)."""
    if t_die <= T_THRESH:
        return 1.0
    factor = 1.0 - K_THROTTLE * (t_die - T_THRESH)
    return max(P_MIN_FRAC, factor)


def leakage(t_die):
    """Puissance de fuite additionnelle, croît avec T_die au-dessus de l'ambiant."""
    return LEAK_COEF * max(0.0, t_die - T_AMB)


def r2_effective(t, incident_time=None, incident_duration=None):
    """Résistance heatsink->ambiant, dégradée si un incident ventilateur est actif."""
    if incident_time is None:
        return R2_HS_AMB
    if incident_duration is None:
        # incident permanent à partir de incident_time
        active = t >= incident_time
    else:
        active = incident_time <= t < incident_time + incident_duration
    return R2_HS_AMB * FAN_DEGRADATION_FACTOR if active else R2_HS_AMB


def derivative(t, state, profile, duration, incident_time=None, incident_duration=None):
    """dstate/dt pour le système [P, T_die, T_hs]."""
    p, t_die, t_hs = state
    p_req = load_profile(t, profile, duration)
    p_target = throttle_factor(t_die) * p_req + leakage(t_die)
    dp_dt = (p_target - p) / TAU_P

    r2 = r2_effective(t, incident_time, incident_duration)
    q_die_hs = (t_die - t_hs) / R1_DIE_HS
    q_hs_amb = (t_hs - T_AMB) / r2

    dtdie_dt = (p - q_die_hs) / C_DIE
    dths_dt = (q_die_hs - q_hs_amb) / C_HS

    return np.array([dp_dt, dtdie_dt, dths_dt])


def simulate(profile="ramp", duration=300.0, dt=0.5, incident_time=None,
             incident_duration=40.0, state0=None):
    """Intègre la trajectoire complète (RK4) et renvoie t, P, T_die, T_hs (arrays denses)."""
    n_steps = int(round(duration / dt))
    t = np.linspace(0.0, duration, n_steps + 1)
    if state0 is None:
        state0 = np.array([P_IDLE, T_AMB + 3.0, T_AMB + 1.0])  # démarrage proche équilibre idle

    traj = np.zeros((n_steps + 1, 3))
    traj[0] = state0
    state = state0.copy()
    for i in range(n_steps):
        ti = t[i]
        k1 = derivative(ti, state, profile, duration, incident_time, incident_duration)
        k2 = derivative(ti + dt / 2, state + dt / 2 * k1, profile, duration, incident_time, incident_duration)
        k3 = derivative(ti + dt / 2, state + dt / 2 * k2, profile, duration, incident_time, incident_duration)
        k4 = derivative(ti + dt, state + dt * k3, profile, duration, incident_time, incident_duration)
        state = state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        traj[i + 1] = state

    return t, traj  # traj[:,0]=P, traj[:,1]=T_die, traj[:,2]=T_hs


def generate_scenarios(n_scenarios=15, seed=42, duration=300.0, dt=0.5):
    """Génère N scénarios variés (profils différents, avec/sans incident, seeds différentes)."""
    rng = np.random.default_rng(seed)
    profiles = ["idle", "ramp", "sustained", "burst"]
    scenarios = []
    for i in range(n_scenarios):
        profile = profiles[i % len(profiles)]
        has_incident = rng.random() < 0.5 and profile != "idle"
        incident_time = float(rng.uniform(duration * 0.3, duration * 0.5)) if has_incident else None
        # incident permanent une fois déclenché (dégradation ventilateur non résolue en 24h)
        t, traj = simulate(profile=profile, duration=duration, dt=dt,
                            incident_time=incident_time, incident_duration=None)
        scenarios.append({
            "id": i,
            "profile": profile,
            "incident_time": incident_time,
            "t": t,
            "traj": traj,  # (n_steps+1, 3) -> P, T_die, T_hs
            "dt": dt,
        })
    return scenarios


def sample_sparse(t, traj, sample_dt=5.0, noise_std=(0.5, 0.3, 0.3), t_start=0.0, t_end=None, rng=None):
    """Échantillonnage éparse + bruité d'une trajectoire dense (simule des capteurs réels).

    noise_std : écarts-types du bruit gaussien pour (P, T_die, T_hs).
    """
    if rng is None:
        rng = np.random.default_rng()
    if t_end is None:
        t_end = t[-1]
    dt_full = t[1] - t[0]
    step = max(1, int(round(sample_dt / dt_full)))
    mask = (t >= t_start) & (t <= t_end)
    idx = np.where(mask)[0][::step]
    t_sparse = t[idx].copy()
    obs = traj[idx].copy()
    obs[:, 0] += rng.normal(0, noise_std[0], size=obs.shape[0])
    obs[:, 1] += rng.normal(0, noise_std[1], size=obs.shape[0])
    obs[:, 2] += rng.normal(0, noise_std[2], size=obs.shape[0])
    return t_sparse, obs


def build_windows(scenario, obs_duration=150.0, pred_duration=150.0, sample_dt=5.0, rng=None):
    """Découpe un scénario en fenêtre d'observation (éparse/bruitée) + fenêtre de prédiction (dense/vérité terrain).

    Utilisé par train.py et compare.py pour préparer les tenseurs d'entrée/sortie
    de façon identique (évite toute divergence entre entraînement et évaluation).
    """
    t, traj = scenario["t"], scenario["traj"]
    t_obs, obs = sample_sparse(t, traj, sample_dt=sample_dt, t_start=0.0, t_end=obs_duration, rng=rng)

    pred_mask = (t > obs_duration) & (t <= obs_duration + pred_duration)
    t_pred = t[pred_mask]
    true_pred = traj[pred_mask]

    return {
        "t_obs": t_obs,
        "obs": obs,                # (L_obs, 3) éparse + bruité
        "t_pred": t_pred,
        "true_pred": true_pred,    # (pred_len, 3) dense, vérité terrain
        "dt_pred": t[1] - t[0],
        "profile": scenario["profile"],
        "incident_time": scenario["incident_time"],
    }


THERMAL_PARAMS_NOMINAL = {
    "T_amb": T_AMB, "R1": R1_DIE_HS, "R2": R2_HS_AMB, "C_die": C_DIE, "C_hs": C_HS,
}


if __name__ == "__main__":
    scenarios = generate_scenarios(n_scenarios=5, seed=0)
    for s in scenarios:
        win = build_windows(s)
        t_die_max = s["traj"][:, 1].max()
        throttled = t_die_max > T_THRESH
        print(f"scenario {s['id']:2d} | profil={s['profile']:10s} | incident={'oui' if s['incident_time'] else 'non':3s} "
              f"| T_die max={t_die_max:5.1f}°C | throttling={'OUI' if throttled else 'non'} "
              f"| obs pts={len(win['t_obs']):3d} | pred pts={len(win['t_pred']):3d}")
