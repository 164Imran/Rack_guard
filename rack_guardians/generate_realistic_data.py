"""
generate_realistic_data.py — Scénarios synthétiques avec couches physiques réalistes.

Complète simulator.py (non modifié) avec des couches supplémentaires documentées :
  1. Courbe de ventilateur avec hystérésis (vitesse fonction de T_die, seuil d'arrêt
     < seuil de démarrage pour éviter le cyclage).
  2. Température ambiante bornée ASHRAE (18-27°C) + incident de climatisation
     (excursion contrôlée au-delà), avec effet mesurable sur R2 (résistance
     heatsink -> ambiant), via un facteur multiplicatif dépendant de l'écart à 25°C.
  3. Throttling thermique par scénario : seuil, sévérité et hystérésis tirés
     aléatoirement (cf. `sample_throttle_params`) au lieu d'un kink linéaire à
     t_thresh=70°C fixe — aucune source réelle ne calibre ce phénomène (ni MIT
     Supercloud, sans colonne température, ni gpu_log_perso, qui ne montre jamais
     de throttling), donc cette partie reste une hypothèse raisonnée documentée,
     pas une mesure.
  4. Domain randomization du réseau RC (R1, R2 nominal, C_die, C_hs) par scénario,
     pour que le NODE apprenne une famille de dynamiques plausibles plutôt qu'une
     trajectoire figée.
  5. Bruit de capteur réaliste (quantification température à 1°C, bruit puissance
     calibré sur gpu_log_perso et mis à l'échelle du scénario) au lieu d'un bruit
     gaussien générique, cf. `sample_sparse_realistic`.
  6. Échelle de puissance (p_idle/p_load_mean/p_load_max) tirée PAR SCÉNARIO dans une
     table de TDP publics des GPU ciblés par le pitch Crusoe (H100/H200/B200/GB200/
     MI300X/MI355X, cf. `sample_gpu_power_scale` et calibration.json["gpu_tdp_table_w"],
     ajoutée par fetch_real_data.py) au lieu d'une valeur MIT générique unique (~242W,
     qui ne correspond à aucun GPU réellement visé). La FORME de la dynamique (pente
     rampe, tau, bruit) reste calibrée sur gpu_log_perso/MIT comme avant ; seule
     l'ÉCHELLE change, ancrée sur les specs matérielles réelles des GPU cibles -- un
     argument de pitch plus fort que "GPU générique datacenter". p_idle/p_load_mean par
     modèle ne sont PAS mesurés (aucune source publique fiable par modèle) : dérivés du
     TDP via le ratio idle/max calibré sur MIT Supercloud, cf. `sample_gpu_power_scale`.
     R1/R2 nominal sont mis à l'échelle inversement au TDP (cf. `sample_rc_params`) :
     un GPU 4-5x plus puissant a nécessairement un refroidissement proportionnellement
     meilleur (plus gros dissipateur/vapor chamber/liquide) pour tenir une marge
     thermique comparable -- principe de conception réel, pas un ajustement ad hoc.

Standalone : n'importe PAS simulator.py, réimplémente sa propre intégration RK4 et
ses propres constantes (mêmes ordres de grandeur, cf. calibration.json si présent).

Lit calibration.json (produit par fetch_real_data.py) s'il existe, sinon utilise le
même fallback codé en dur. Sauvegarde les scénarios dans data/realistic_scenarios.pkl,
même structure de dict que simulator.generate_scenarios() (id, profile, incident_time,
t, traj, dt) + champs additionnels (t_thresh, throttle_severity, throttle_hyst_delta,
rc_params) -> réutilisable tel quel par train.py à la place de simulator.py.

Exécutable seul : `python generate_realistic_data.py` (quelques secondes).
"""

import json
import os
import pickle
import time

import numpy as np

CALIBRATION_PATH = "calibration.json"
OUTPUT_PATH = "data/realistic_scenarios.pkl"

# ============================================================
# FALLBACK — identique à fetch_real_data.py, utilisé si calibration.json est absent.
# ============================================================
FALLBACK_CALIBRATION = {
    "power_idle_mean_w": 25.0,
    "power_load_mean_w": 85.0,
    "power_load_max_w": 100.0,
    "temp_gpu_load_p95_c": 70.0,
}

# Constantes physiques non calibrables depuis le dataset réel (pas de température
# dans ce mirroir, cf. fetch_real_data.py) : reprises des mêmes ordres de grandeur
# documentés dans simulator.py. Servent de CENTRE pour la domain randomization
# par scénario (cf. `sample_rc_params`), pas de valeur fixe unique.
R1_DIE_HS_NOMINAL = 0.06        # °C/W
R2_NOMINAL_NOMINAL = 0.14       # °C/W, à 25°C ambiant et ventilateur pleine vitesse
C_DIE_NOMINAL = 45.0            # J/°C
C_HS_NOMINAL = 220.0            # J/°C
RC_JITTER_FRAC = 0.25           # domain randomization : +/-25% autour du nominal
TAU_P = 1.5             # s
LEAK_COEF = 0.03        # W/°C

# --- Throttling thermique (100% synthétique, cf. docstring du module) ---
# Plage de seuil de déclenchement : littérature générale sur le DVFS thermique GPU
# (consumer et datacenter confondus, les fiches techniques placent typiquement le
# throttling entre ~80-90°C côté consumer et un peu plus bas côté datacenter avec
# marge de sécurité) -> on prend une plage 75-85°C comme ordre de grandeur plausible,
# tirée par scénario plutôt qu'une valeur unique choisie à la main.
T_THRESH_MIN, T_THRESH_MAX = 75.0, 85.0
# Sévérité (fraction de puissance perdue à l'intensité maximale du throttling) :
# ancrée sur les 3 segments M100 observés (chute de puissance nœud entière -19% à
# -26%, proxy indirect car pas de métrique de clock GPU dans ce dataset public) ;
# on élargit légèrement la plage (15-30%) pour couvrir l'incertitude de ce proxy.
THROTTLE_SEVERITY_MIN, THROTTLE_SEVERITY_MAX = 0.15, 0.30
# Hypothèse (non mesurée) : l'intensité de la sévérité est atteinte progressivement
# sur les 10°C au-dessus du seuil (palier DVFS typique), pas instantanément.
THROTTLE_SEVERITY_SPAN_C = 10.0
# Hystérésis : le GPU ne revient en puissance normale qu'en repassant sous
# (t_thresh - delta), delta tiré par scénario -> évite un retour instantané dès
# qu'un bruit fait osciller T d'un degré autour du seuil (même logique que le
# ventilateur ci-dessous, hypothèse raisonnée, pas mesurée).
THROTTLE_HYST_MIN, THROTTLE_HYST_MAX = 3.0, 8.0

# --- Ambiant ASHRAE ---
T_AMB_MIN, T_AMB_MAX = 18.0, 27.0    # plage standard datacenter (ASHRAE TC9.9)
T_AMB_REF = 25.0                     # référence pour le facteur de dégradation R2
AC_INCIDENT_T_AMB = 36.0             # excursion pendant un incident de climatisation
K_AMBIENT_R2 = 0.03                  # /°C, dégradation de R2 par °C au-dessus de T_AMB_REF

# --- Forme de la rampe de charge (repli si aucune calibration perso disponible) ---
# Ces fractions gouvernent la FORME de la montée en charge (durée relative avant
# stabilisation), pas l'échelle de puissance. Par défaut recalibrées depuis
# calibration.json["gpu_log_perso"] (log nvidia-smi réel, cf. fetch_real_data.py) ;
# ces constantes ne servent que si cette calibration perso est absente.
RAMP_FRAC_DEFAULT = 0.4
SUSTAINED_FRAC_DEFAULT = 0.15

# --- Ventilateur (hystérésis) ---
T_FAN_STOP = 40.0       # °C, en dessous -> vitesse minimale (si déjà actif, se désactive)
T_FAN_START = 45.0      # °C, au-dessus -> le ventilateur s'active/accélère (> T_FAN_STOP : évite le cyclage)
T_FAN_MAX = 75.0        # °C, température à laquelle le ventilateur est à pleine vitesse
FAN_SPEED_MIN = 0.4     # vitesse minimale (jamais à l'arrêt total, sécurité)
FAN_SPEED_MAX = 1.0

# --- Bruit de capteur (calibré sur gpu_log_perso, cf. sample_sparse_realistic) ---
TEMP_QUANT_STEP_C = 1.0            # quantification entière observée dans gpu_log_perso
POWER_NOISE_MIN_W_PERSO = 0.1      # amplitude bruit puissance à l'échelle laptop (gpu_log_perso)
POWER_NOISE_MAX_W_PERSO = 0.5
GPU_LOG_PERSO_P_LOAD_MAX_W_DEFAULT = 44.51   # repli si calibration.json["gpu_log_perso"] absent


def load_calibration():
    """Charge calibration.json si présent, sinon fallback. Affiche la source utilisée."""
    if os.path.exists(CALIBRATION_PATH):
        with open(CALIBRATION_PATH) as f:
            calib = json.load(f)
        print(f"[generate_realistic_data] calibration chargée depuis {CALIBRATION_PATH} "
              f"(fallback_used={calib.get('fallback_used')})")
        return calib
    print(f"[generate_realistic_data] {CALIBRATION_PATH} introuvable -> fallback codé en dur.")
    return FALLBACK_CALIBRATION


def fan_speed_step(t_die, fan_active_prev):
    """Un pas de la machine à états du ventilateur (hystérésis). Renvoie (vitesse, nouvel état actif)."""
    if fan_active_prev:
        fan_active = t_die >= T_FAN_STOP     # se désactive seulement en dessous du seuil bas
    else:
        fan_active = t_die >= T_FAN_START    # s'active seulement au-dessus du seuil haut

    if not fan_active:
        return FAN_SPEED_MIN, fan_active

    frac = np.clip((t_die - T_FAN_START) / (T_FAN_MAX - T_FAN_START), 0.0, 1.0)
    speed = FAN_SPEED_MIN + (FAN_SPEED_MAX - FAN_SPEED_MIN) * frac
    return speed, fan_active


def throttle_step(t_die, t_thresh, severity, hyst_delta, throttle_active_prev):
    """Un pas de la machine à états du throttling (contrôle proportionnel + hystérésis).

    Remplace le kink linéaire instantané par un comportement plus proche d'un vrai
    DVFS thermique : la puissance ne chute pas d'un coup dès T>=t_thresh, elle décroît
    progressivement avec l'écart au seuil (jusqu'à `severity` de perte max, atteinte
    sur THROTTLE_SEVERITY_SPAN_C °C au-dessus du seuil), et le contrôleur ne repasse
    en régime normal qu'en repassant sous (t_thresh - hyst_delta) : évite un retour
    instantané en puissance normale au moindre °C d'oscillation autour du seuil.
    Comme pour le ventilateur, l'état est évalué une fois par pas de sortie et tenu
    constant sur tout le pas RK4 (cohérent avec un pas de sortie déjà fin, dt=0.5s).
    """
    if throttle_active_prev:
        active = t_die >= (t_thresh - hyst_delta)
    else:
        active = t_die >= t_thresh

    if not active:
        return 1.0, active

    frac_over = np.clip((t_die - t_thresh) / THROTTLE_SEVERITY_SPAN_C, 0.0, 1.0)
    factor = 1.0 - severity * frac_over
    return factor, active


def ambient_r2_factor(t_amb):
    """Facteur multiplicatif sur R2 selon l'écart à l'ambiant de référence (25°C).

    Un ambiant plus chaud réduit l'efficacité du radiateur -> R2 effectif augmente.
    Effet mesurable et non décoratif : à +10°C d'ambiant, R2 augmente de 30%.
    """
    return 1.0 + K_AMBIENT_R2 * (t_amb - T_AMB_REF)


def sample_throttle_params(rng):
    """Tire (t_thresh, severity, hyst_delta) par scénario, cf. constantes documentées ci-dessus."""
    t_thresh = float(rng.uniform(T_THRESH_MIN, T_THRESH_MAX))
    severity = float(rng.uniform(THROTTLE_SEVERITY_MIN, THROTTLE_SEVERITY_MAX))
    hyst_delta = float(rng.uniform(THROTTLE_HYST_MIN, THROTTLE_HYST_MAX))
    return t_thresh, severity, hyst_delta


def sample_rc_params(rng, p_load_max=None, p_load_max_ref=None):
    """Domain randomization du réseau RC thermique : +/-RC_JITTER_FRAC autour du nominal,
    par scénario, pour que le NODE apprenne une famille de dynamiques plausibles plutôt
    qu'une seule trajectoire figée (pas de source réelle pour calibrer R/C directement,
    cf. commentaire sur les constantes nominales ci-dessus).

    Si p_load_max/p_load_max_ref sont fournis (GPU cible Crusoe, cf. `sample_gpu_power_scale`),
    R1/R2 nominal sont mis à l'échelle par (p_load_max_ref / p_load_max) AVANT jitter : un
    GPU plus puissant que la référence MIT (p_load_max_ref) a un refroidissement
    proportionnellement meilleur (principe de conception réel des GPU datacenter, qui
    opèrent près de leur limite thermique quel que soit leur TDP), pas un ajustement ad hoc.
    C_die/C_hs sont mis à l'échelle par le facteur INVERSE (p_load_max / p_load_max_ref) :
    sans cette compensation, réduire R seul réduit aussi tau=R*C (le système réagit plus
    vite), ce qui dégrade l'accord de constante de temps avec gpu_log_perso (vérifié : tau
    passait de -8.8% à -40.8% d'écart sans cette compensation). Physiquement cohérent aussi :
    un GPU plus puissant a une masse thermique (die/heatsink) proportionnellement plus
    grande, pas seulement un refroidissement plus efficace.
    """
    r1_center, r2_center = R1_DIE_HS_NOMINAL, R2_NOMINAL_NOMINAL
    c_die_center, c_hs_center = C_DIE_NOMINAL, C_HS_NOMINAL
    if p_load_max is not None and p_load_max_ref:
        power_scale = p_load_max_ref / p_load_max
        r1_center *= power_scale
        r2_center *= power_scale
        c_die_center /= power_scale
        c_hs_center /= power_scale

    def jitter(nominal):
        return float(nominal * rng.uniform(1.0 - RC_JITTER_FRAC, 1.0 + RC_JITTER_FRAC))

    return {
        "R1": jitter(r1_center),
        "R2_nominal": jitter(r2_center),
        "C_die": jitter(c_die_center),
        "C_hs": jitter(c_hs_center),
    }


def sample_gpu_power_scale(rng, calib):
    """Tire le modèle de GPU cible (table TDP Crusoe, cf. calibration.json["gpu_tdp_table_w"]
    ajoutée par fetch_real_data.py) et renvoie (gpu_model, p_idle, p_load_mean, p_load_max)
    pour CE scénario. p_load_max = TDP officiel du modèle tiré. p_idle/p_load_mean ne sont
    PAS mesurés par modèle (aucune source publique fiable) : dérivés du TDP via le ratio
    idle/max et moyenne/max calibrés sur MIT Supercloud, appliqué tel quel -- approximation
    documentée, pas une mesure par modèle.
    """
    table = calib.get("gpu_tdp_table_w")
    mit_p_idle = calib.get("power_idle_mean_w") or FALLBACK_CALIBRATION["power_idle_mean_w"]
    mit_p_load_mean = calib.get("power_load_mean_w") or FALLBACK_CALIBRATION["power_load_mean_w"]
    mit_p_load_max = calib.get("power_load_max_w") or FALLBACK_CALIBRATION["power_load_max_w"]

    if not table:
        # calibration.json pas régénéré avec la table Crusoe -> repli MIT générique
        return "generic_mit", mit_p_idle, mit_p_load_mean, mit_p_load_max

    weights_map = calib.get("gpu_tdp_weights") or {}
    models = list(table.keys())
    weights = np.array([weights_map.get(m, 1.0) for m in models], dtype=float)
    weights = weights / weights.sum()
    model = models[int(rng.choice(len(models), p=weights))]
    p_load_max = float(table[model])

    idle_ratio = mit_p_idle / mit_p_load_max
    mean_ratio = mit_p_load_mean / mit_p_load_max
    p_idle = p_load_max * idle_ratio
    p_load_mean = p_load_max * mean_ratio
    return model, p_idle, p_load_mean, p_load_max


def load_profile(t, profile, duration, p_idle, p_load_mean, p_load_max,
                  ramp_frac=RAMP_FRAC_DEFAULT, sustained_frac=SUSTAINED_FRAC_DEFAULT):
    """Puissance demandée (avant throttling/leakage), calibrée sur les valeurs réelles/fallback.

    ramp_frac/sustained_frac : fraction de `duration` avant stabilisation, calibrée par
    défaut sur la FORME observée dans le log nvidia-smi perso (temps relatif pour
    atteindre 90% de l'excursion thermique après le début de charge), cf.
    fetch_real_data.compute_perso_gpu_calibration(). Les VALEURS de puissance (p_idle,
    p_load_mean, p_load_max) restent, elles, à l'échelle datacenter (MIT Supercloud).
    """
    if profile == "idle":
        return p_idle
    if profile == "ramp":
        ramp_end = duration * ramp_frac
        target = p_load_mean
        if t < ramp_end:
            return p_idle + (target - p_idle) * (t / ramp_end)
        return target
    if profile == "sustained":
        ramp_end = duration * sustained_frac
        target = p_load_max
        if t < ramp_end:
            return p_idle + (target - p_idle) * (t / ramp_end)
        return target
    if profile == "burst":
        period, burst_len = 40.0, 8.0
        phase = t % period
        return p_load_max if phase < burst_len else p_idle
    raise ValueError(f"profil inconnu: {profile}")


def ambient_at(t, amb_baseline, ac_incident_time):
    """Température ambiante au temps t : baseline ASHRAE, ou excursion si incident actif."""
    if ac_incident_time is not None and t >= ac_incident_time:
        return AC_INCIDENT_T_AMB
    return amb_baseline


def simulate_realistic(profile, duration, dt, p_idle, p_load_mean, p_load_max,
                        t_thresh, severity, hyst_delta, rc_params,
                        amb_baseline, ac_incident_time=None, rng=None,
                        ramp_frac=RAMP_FRAC_DEFAULT, sustained_frac=SUSTAINED_FRAC_DEFAULT):
    """Intègre la trajectoire complète (RK4, sans sous-pas -- cohérent avec simulator.py)
    avec ventilateur hystérétique + ambiant dynamique couplés à R2 + throttling
    proportionnel/hystérétique (cf. `throttle_step`), sur un réseau RC randomisé par
    scénario (cf. `rc_params` / `sample_rc_params`).
    """
    r1 = rc_params["R1"]
    r2_nominal = rc_params["R2_nominal"]
    c_die = rc_params["C_die"]
    c_hs = rc_params["C_hs"]

    n_steps = int(round(duration / dt))
    t = np.linspace(0.0, duration, n_steps + 1)

    state = np.array([p_idle, amb_baseline + 3.0, amb_baseline + 1.0])
    traj = np.zeros((n_steps + 1, 3))
    fan_speed_series = np.zeros(n_steps + 1)
    amb_series = np.zeros(n_steps + 1)
    traj[0] = state
    fan_active = False
    throttle_active = False

    def derivative(ti, s, r2_eff, throttle_factor):
        p, t_die, t_hs = s
        p_req = load_profile(ti, profile, duration, p_idle, p_load_mean, p_load_max,
                              ramp_frac=ramp_frac, sustained_frac=sustained_frac)
        leak = LEAK_COEF * max(0.0, t_die - amb_baseline)
        p_target = throttle_factor * p_req + leak
        dp_dt = (p_target - p) / TAU_P

        q_die_hs = (t_die - t_hs) / r1
        t_amb_now = ambient_at(ti, amb_baseline, ac_incident_time)
        q_hs_amb = (t_hs - t_amb_now) / r2_eff
        dtdie_dt = (p - q_die_hs) / c_die
        dths_dt = (q_die_hs - q_hs_amb) / c_hs
        return np.array([dp_dt, dtdie_dt, dths_dt])

    for i in range(n_steps):
        ti = t[i]
        # ventilateur + ambiant + throttling évalués une fois par pas de sortie (tenus
        # constants sur tout le pas RK4, cohérent avec un pas de sortie dt=0.5s déjà fin)
        fan_speed, fan_active = fan_speed_step(state[1], fan_active)
        t_amb_now = ambient_at(ti, amb_baseline, ac_incident_time)
        r2_eff = (r2_nominal * ambient_r2_factor(t_amb_now)) / fan_speed
        throttle_factor, throttle_active = throttle_step(state[1], t_thresh, severity,
                                                           hyst_delta, throttle_active)

        fan_speed_series[i] = fan_speed
        amb_series[i] = t_amb_now

        k1 = derivative(ti, state, r2_eff, throttle_factor)
        k2 = derivative(ti + dt / 2, state + dt / 2 * k1, r2_eff, throttle_factor)
        k3 = derivative(ti + dt / 2, state + dt / 2 * k2, r2_eff, throttle_factor)
        k4 = derivative(ti + dt, state + dt * k3, r2_eff, throttle_factor)
        state = state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        traj[i + 1] = state

    fan_speed_series[-1] = fan_speed_series[-2]
    amb_series[-1] = amb_series[-2]
    return t, traj, fan_speed_series, amb_series


def sample_sparse_realistic(t, traj, sample_dt, p_scale_w, calib, rng, t_start=0.0, t_end=None):
    """Échantillonnage éparse + bruit de capteur réaliste (remplace le bruit gaussien
    générique de simulator.sample_sparse) :
      - Température : quantification à TEMP_QUANT_STEP_C (1°C), comme observé sur les
        valeurs entières de gpu_log_perso (pas de bruit gaussien continu ensuite : le
        capteur réel ne renvoie que des entiers, la quantification EST le bruit).
      - Puissance : bruit gaussien d'amplitude POWER_NOISE_MIN/MAX_W_PERSO (0.1-0.5W),
        mesuré à l'échelle laptop de gpu_log_perso, mis à l'échelle du scénario
        (p_scale_w) proportionnellement plutôt que recopié brut.
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

    perso_p_scale = calib.get("gpu_log_perso", {}).get(
        "power_load_max_w_perso", GPU_LOG_PERSO_P_LOAD_MAX_W_DEFAULT)
    scale_factor = p_scale_w / perso_p_scale
    noise_amp_w = rng.uniform(POWER_NOISE_MIN_W_PERSO, POWER_NOISE_MAX_W_PERSO) * scale_factor
    obs[:, 0] += rng.normal(0, noise_amp_w, size=obs.shape[0])

    obs[:, 1] = np.round(obs[:, 1] / TEMP_QUANT_STEP_C) * TEMP_QUANT_STEP_C
    obs[:, 2] = np.round(obs[:, 2] / TEMP_QUANT_STEP_C) * TEMP_QUANT_STEP_C
    return t_sparse, obs


def build_windows_realistic(scenario, obs_duration, pred_duration, sample_dt, p_scale_w, calib, rng=None):
    """Équivalent de simulator.build_windows, avec bruit de capteur réaliste (cf.
    `sample_sparse_realistic`) au lieu du bruit gaussien générique. true_pred reste la
    trajectoire dense SANS bruit (vérité terrain), comme dans simulator.build_windows.
    """
    t, traj = scenario["t"], scenario["traj"]
    t_obs, obs = sample_sparse_realistic(t, traj, sample_dt=sample_dt, p_scale_w=p_scale_w,
                                          calib=calib, rng=rng, t_start=0.0, t_end=obs_duration)

    pred_mask = (t > obs_duration) & (t <= obs_duration + pred_duration)
    t_pred = t[pred_mask]
    true_pred = traj[pred_mask]

    return {
        "t_obs": t_obs,
        "obs": obs,                # (L_obs, 3) éparse + bruit réaliste
        "t_pred": t_pred,
        "true_pred": true_pred,    # (pred_len, 3) dense, vérité terrain
        "dt_pred": t[1] - t[0],
        "profile": scenario["profile"],
        "incident_time": scenario["incident_time"],
    }


def generate_realistic_scenarios(n_scenarios=50, seed=42, duration=300.0, dt=0.5, calib=None,
                                  min_throttled=18):
    """Génère N scénarios réalistes (mêmes profils que simulator.py + ambiant/ventilo/incident AC
    + throttling/RC randomisés par scénario, cf. `sample_throttle_params`/`sample_rc_params`).

    min_throttled : nombre minimum de scénarios devant réellement franchir leur t_thresh
    (T_die max de la trajectoire complète > t_thresh du scénario). Sans cette garantie, le
    tirage aléatoire (profil/ambiant/incident/seuil) ne produit qu'une petite fraction de
    franchissements, ce qui laisse un jeu de test bien trop pauvre pour juger la détection de
    throttling. Les scénarios manquants sont forcés déterministement (profil ramp/sustained +
    incident climatisation), cf. `_force_throttle()` ci-dessous.
    """
    if calib is None:
        calib = load_calibration()

    mit_p_load_max_ref = calib.get("power_load_max_w") or FALLBACK_CALIBRATION["power_load_max_w"]
    gpu_table = calib.get("gpu_tdp_table_w")

    perso = calib.get("gpu_log_perso", {})
    ramp_frac = perso.get("ramp_frac_perso", RAMP_FRAC_DEFAULT)
    sustained_frac = perso.get("sustained_frac_perso", SUSTAINED_FRAC_DEFAULT)
    if gpu_table:
        print(f"[generate_realistic_data] échelle de puissance tirée PAR SCÉNARIO dans la table "
              f"TDP Crusoe : {', '.join(f'{k}={v:.0f}W' for k, v in gpu_table.items())} "
              f"(p_idle/p_load_mean dérivés via ratio idle/max calibré MIT, réf={mit_p_load_max_ref:.1f}W)")
    else:
        print(f"[generate_realistic_data] table TDP Crusoe absente de calibration.json -> repli "
              f"MIT générique p_load_max={mit_p_load_max_ref:.1f}W (relancer fetch_real_data.py "
              f"pour régénérer la table).")
    print(f"[generate_realistic_data] t_thresh~U({T_THRESH_MIN:.0f},{T_THRESH_MAX:.0f})°C (tiré par "
          f"scénario) | forme rampe (perso) ramp_frac={ramp_frac:.3f} sustained_frac={sustained_frac:.3f}")

    rng = np.random.default_rng(seed)
    profiles = ["idle", "ramp", "sustained", "burst"]

    def build_one(i, profile, amb_baseline, ac_incident_time, throttle_params=None, rc_params=None,
                  gpu_power=None):
        if gpu_power is None:
            gpu_power = sample_gpu_power_scale(rng, calib)
        gpu_model, p_idle, p_load_mean, p_load_max = gpu_power

        if throttle_params is None:
            throttle_params = sample_throttle_params(rng)
        if rc_params is None:
            rc_params = sample_rc_params(rng, p_load_max=p_load_max, p_load_max_ref=mit_p_load_max_ref)
        t_thresh, severity, hyst_delta = throttle_params

        t, traj, fan_speed, amb_series = simulate_realistic(
            profile, duration, dt, p_idle, p_load_mean, p_load_max,
            t_thresh, severity, hyst_delta, rc_params,
            amb_baseline, ac_incident_time, rng=rng, ramp_frac=ramp_frac, sustained_frac=sustained_frac,
        )
        return {
            "id": i,
            "profile": profile,
            "incident_time": ac_incident_time,   # même clé que simulator.py (compat train.py/build_windows)
            "t": t,
            "traj": traj,                        # (n_steps+1, 3) -> P, T_die, T_hs (même ordre que simulator.py)
            "dt": dt,
            "amb_baseline": amb_baseline,         # métadonnées additionnelles, ignorées par build_windows
            "fan_speed": fan_speed,
            "amb_series": amb_series,
            "t_thresh": t_thresh,                 # seuil de throttling tiré pour CE scénario
            "throttle_severity": severity,
            "throttle_hyst_delta": hyst_delta,
            "rc_params": rc_params,
            "gpu_model": gpu_model,                # GPU cible Crusoe tiré pour CE scénario
            "p_idle": p_idle,
            "p_load_mean": p_load_mean,
            "p_load_max": p_load_max,              # TDP officiel du modèle tiré (échelle du scénario)
        }

    scenarios = []
    for i in range(n_scenarios):
        profile = profiles[i % len(profiles)]
        amb_baseline = float(rng.uniform(T_AMB_MIN, T_AMB_MAX))
        has_ac_incident = rng.random() < 0.4 and profile != "idle"
        ac_incident_time = float(rng.uniform(duration * 0.3, duration * 0.5)) if has_ac_incident else None
        scenarios.append(build_one(i, profile, amb_baseline, ac_incident_time))

    n_cross = sum(1 for s in scenarios if s["traj"][:, 1].max() > s["t_thresh"])
    if n_cross < min_throttled:
        # idle/burst ne franchissent quasiment jamais le seuil (puissance trop faible/trop
        # intermittente, vérifié empiriquement) : on force ramp/sustained + incident AC sur les
        # scénarios non-idle/burst déjà présents en priorité, sinon on convertit un idle/burst.
        # Le seuil/sévérité/hystérésis et le RC restant tirés aléatoirement (cf. domain
        # randomization), certaines combinaisons (t_thresh haut + RC très "bien refroidi")
        # ne franchissent toujours pas même en sustained+incident AC : on retire alors ces
        # paramètres et on retire (jusqu'à ATTEMPTS_CAP essais), plutôt que de forcer une
        # valeur fixe qui romprait la variété voulue par le tirage aléatoire.
        ATTEMPTS_CAP = 30
        candidates = [s for s in scenarios if s["traj"][:, 1].max() <= s["t_thresh"]]
        candidates.sort(key=lambda s: 0 if s["profile"] in ("ramp", "sustained") else 1)
        needed = min_throttled - n_cross
        forced = 0
        for s in candidates:
            if forced >= needed:
                break
            forced_profile = s["profile"] if s["profile"] in ("ramp", "sustained") else "ramp"
            throttle_params = (s["t_thresh"], s["throttle_severity"], s["throttle_hyst_delta"])
            rc_params = s["rc_params"]
            # le GPU (modèle/échelle de puissance) tiré pour ce scénario reste fixe pendant les
            # essais : seuls le throttling/RC sont retirés, pas le GPU lui-même.
            gpu_power = (s["gpu_model"], s["p_idle"], s["p_load_mean"], s["p_load_max"])
            for _attempt in range(ATTEMPTS_CAP):
                forced_incident = float(rng.uniform(duration * 0.3, duration * 0.5))
                candidate_scn = build_one(s["id"], forced_profile, s["amb_baseline"], forced_incident,
                                           throttle_params=throttle_params, rc_params=rc_params,
                                           gpu_power=gpu_power)
                if candidate_scn["traj"][:, 1].max() > candidate_scn["t_thresh"]:
                    scenarios[s["id"]] = candidate_scn
                    forced += 1
                    break
                # combinaison infranchissable -> re-tirage complet (nouveau seuil/sévérité/RC,
                # même GPU/échelle de puissance)
                throttle_params = sample_throttle_params(rng)
                rc_params = sample_rc_params(rng, p_load_max=gpu_power[3], p_load_max_ref=mit_p_load_max_ref)
        n_cross = sum(1 for s in scenarios if s["traj"][:, 1].max() > s["t_thresh"])

    print(f"[generate_realistic_data] {n_cross}/{len(scenarios)} scénarios franchissent leur t_thresh "
          f"(min_throttled={min_throttled}).")
    return scenarios


def coherence_self_test(duration=300.0, dt=0.5, calib=None):
    """Vérifie que la vitesse du ventilateur augmente nettement entre 25°C et 30°C d'ambiant."""
    if calib is None:
        calib = load_calibration()
    p_idle = calib.get("power_idle_mean_w") or FALLBACK_CALIBRATION["power_idle_mean_w"]
    p_load_mean = calib.get("power_load_mean_w") or FALLBACK_CALIBRATION["power_load_mean_w"]
    p_load_max = calib.get("power_load_max_w") or FALLBACK_CALIBRATION["power_load_max_w"]
    # paramètres FIXES (pas de tirage aléatoire) : ce test ne doit isoler que l'effet de
    # l'ambiant sur le ventilateur, pas être confondu par un tirage RC/throttle différent.
    t_thresh = 0.5 * (T_THRESH_MIN + T_THRESH_MAX)
    severity = 0.5 * (THROTTLE_SEVERITY_MIN + THROTTLE_SEVERITY_MAX)
    hyst_delta = 0.5 * (THROTTLE_HYST_MIN + THROTTLE_HYST_MAX)
    rc_params = {"R1": R1_DIE_HS_NOMINAL, "R2_nominal": R2_NOMINAL_NOMINAL,
                 "C_die": C_DIE_NOMINAL, "C_hs": C_HS_NOMINAL}

    _, _, fan_25, _ = simulate_realistic("sustained", duration, dt, p_idle, p_load_mean, p_load_max,
                                          t_thresh, severity, hyst_delta, rc_params, amb_baseline=25.0)
    _, _, fan_30, _ = simulate_realistic("sustained", duration, dt, p_idle, p_load_mean, p_load_max,
                                          t_thresh, severity, hyst_delta, rc_params, amb_baseline=30.0)

    mean_25, mean_30 = fan_25.mean(), fan_30.mean()
    increased = mean_30 > mean_25 * 1.05  # au moins +5% relatif pour parler d'effet "net"

    print("\n=== Test de cohérence : effet ambiant -> ventilateur ===")
    print(f"vitesse ventilo moyenne @ ambiant=25°C : {mean_25:.3f}")
    print(f"vitesse ventilo moyenne @ ambiant=30°C : {mean_30:.3f}")
    print(f"la vitesse augmente nettement avec l'ambiant : {'OUI' if increased else 'NON'}")
    return increased


def main():
    t0 = time.perf_counter()
    print("[generate_realistic_data] démarrage...")

    calib = load_calibration()
    coherence_self_test(calib=calib)

    print("\n[generate_realistic_data] génération des scénarios...")
    scenarios = generate_realistic_scenarios(n_scenarios=50, seed=42, calib=calib, min_throttled=18)

    n_throttled = sum(1 for s in scenarios if s["traj"][:, 1].max() > s["t_thresh"])
    print(f"[generate_realistic_data] {len(scenarios)} scénarios générés, "
          f"{n_throttled} avec throttling atteint")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump(scenarios, f)

    elapsed = time.perf_counter() - t0
    print(f"[generate_realistic_data] sauvegardé dans {OUTPUT_PATH} en {elapsed:.2f}s au total.")


if __name__ == "__main__":
    main()
