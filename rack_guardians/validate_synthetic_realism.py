"""
validate_synthetic_realism.py — Sanity-check simple : compare, sur la partie NON-throttling
(seule partie vérifiable -- ni MIT Supercloud ni gpu_log_perso ne calibrent le
franchissement de seuil lui-même, cf. generate_realistic_data.py/fetch_real_data.py),
deux métriques mesurées sur le log réel (gpu_log_stress2.csv) contre les mêmes métriques
mesurées sur des scénarios synthétiques fraîchement générés :

  1. Pente puissance/température, normalisée par l'amplitude de puissance de charge
     observée SUR LE MÊME segment que la pente (pas une normalisation par une plage
     globale différente -- cf. correction ci-dessous). Numérateur et dénominateur
     sont calculés sur la trajectoire COMPLÈTE (idle -> rampe -> charge/seuil) des
     deux côtés, réel et synthétique, pour rester apples-to-apples.
  2. Constante de temps tau de la montée en température (secondes) -- déjà comparable
     telle quelle, invariante à l'échelle de puissance.

CORRECTION (après investigation) : une version antérieure de ce script calculait la
pente réelle uniquement sur les échantillons "en charge stabilisée" (util>=95%, un
segment étroit et dominé par le bruit du capteur), mais la normalisait par l'amplitude
de puissance GLOBALE (idle->charge) -- un décalage de périmètre numérateur/dénominateur
qui gonflait artificiellement l'écart rapporté (+33%). Corrigé ici : les deux côtés
utilisent désormais la trajectoire complète, cohérente de bout en bout.

CAVEAT IMPORTANT (structurel, pas un bug) : une fois cette incohérence corrigée, l'écart
mesuré est en réalité beaucoup plus grand (~-65%), et il n'est PAS réductible en ajustant
R1/R2/C_die/C_hs sans casser la détection de franchissement de seuil (vérifié
empiriquement : réduire R1/R2 assez pour rapprocher la pente fait tomber le nombre de
scénarios franchissant leur seuil à 0/50, même avec le mécanisme de forçage). La raison
est structurelle, pas une erreur de calibration : gpu_log_perso est un GPU grand public
qui ne monte que d'environ 14°C entre idle et charge soutenue (45°C->58-61°C, plafonne
loin de tout throttling), alors que les scénarios synthétiques (échelle Crusoe/datacenter)
doivent nécessairement pouvoir approcher un seuil bien plus haut (75-85°C) depuis un
ambiant plus bas (18-27°C ASHRAE), donc opérer sur une excursion thermique ~4x plus large.
Comparer la pente P/T entre ces deux régimes physiques différents (laptop vs GPU
datacenter proche de sa limite thermique) est donc structurellement optimiste dans le
meilleur des cas -- on le rapporte quand même, honnêtement, mais tau (invariant
d'échelle) reste la métrique la plus significative de ce script.

Pas de test statistique : juste un écart en % affiché en console, pour juger si le
synthétique reste dans le bon ordre de grandeur sur la partie qu'on peut vérifier.

Exécutable seul : `python validate_synthetic_realism.py`
"""

import numpy as np
import pandas as pd

from generate_realistic_data import generate_realistic_scenarios, load_calibration

PERSO_LOG_PATH = "gpu_log_stress2.csv"


def _tau_from_rise(t, temp, t_onset, temp_idle, temp_asymp):
    """Constante de temps tau (s), en supposant une montée exponentielle
    temp(t) = temp_idle + (temp_asymp - temp_idle) * (1 - exp(-(t - t_onset)/tau)),
    estimée à partir de l'instant t90 où 90% de l'excursion est atteinte :
    tau = t90 / ln(10). Approximation simple (pas un fit non-linéaire complet),
    suffisante pour un écart en %, pas pour une calibration fine.
    """
    if temp_asymp <= temp_idle:
        return None
    target = temp_idle + 0.9 * (temp_asymp - temp_idle)
    after = t >= t_onset
    idx = np.where(after & (temp >= target))[0]
    if len(idx) == 0:
        return None
    t90 = t[idx[0]] - t_onset
    if t90 <= 0:
        return None
    return float(t90 / np.log(10))


def real_metrics(path=PERSO_LOG_PATH):
    """Pente P/T (normalisée) et tau de montée, mesurés sur le log nvidia-smi personnel.

    Ce log ne contient aucun throttling (cf. fetch_real_data.py) : il est déjà, dans son
    intégralité, la partie "saine" qu'on veut comparer au synthétique.

    Pente calculée sur la trajectoire COMPLÈTE (idle -> rampe -> charge), normalisée
    par l'amplitude de puissance de CE MÊME segment -- pas sur un sous-ensemble "charge
    stabilisée" étroit normalisé par une plage différente (cf. CORRECTION en tête de
    module), pour rester cohérent avec la méthode utilisée côté synthétique.
    """
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    df["util"] = pd.to_numeric(
        df["utilization.gpu [%]"].astype(str).str.replace("%", "", regex=False).str.strip(),
        errors="coerce")
    df["power"] = pd.to_numeric(
        df["power.draw [W]"].astype(str).str.replace("W", "", regex=False).str.strip(), errors="coerce")
    df["temp"] = pd.to_numeric(df["temperature.gpu"], errors="coerce")
    df["ts"] = pd.to_datetime(df["timestamp"].str.strip(), format="%Y/%m/%d %H:%M:%S.%f")
    df["t_s"] = (df["ts"] - df["ts"].iloc[0]).dt.total_seconds()

    first_load_idx = df.index[df["util"] > 10.0].min()
    pre_idle = df.loc[:first_load_idx - 1]
    pre_idle = pre_idle[pre_idle["util"] < 5.0]
    load_rows = df[df["util"] >= 95.0]

    p_idle, p_load_max = float(pre_idle["power"].mean()), float(load_rows["power"].max())
    temp_idle = float(pre_idle["temp"].mean())
    temp_asymp = float(load_rows["temp"].tail(20).mean())
    t_onset = float(df.loc[first_load_idx, "t_s"])

    valid = df[["util", "power", "temp"]].dropna()
    slope_raw, _ = np.polyfit(valid["temp"], valid["power"], 1)
    p_range_full = float(valid["power"].max() - valid["power"].min())
    slope_norm = float(slope_raw) / p_range_full  # 1/°C, même segment que le numérateur

    tau = _tau_from_rise(df["t_s"].to_numpy(), df["temp"].to_numpy(), t_onset, temp_idle, temp_asymp)
    return {"slope_norm_per_c": slope_norm, "tau_s": tau, "p_idle": p_idle, "p_load_max": p_load_max}


def synthetic_metrics(calib, n_scenarios=15, seed=777):
    """Mêmes métriques mesurées sur des scénarios ramp/sustained synthétiques fraîchement
    générés, restreintes à la partie T_die < t_thresh du scénario (seule partie
    vérifiable ; le comportement au-delà du seuil est hors-scope de cette validation,
    cf. docstring du module).
    """
    scenarios = generate_realistic_scenarios(n_scenarios=n_scenarios, seed=seed, calib=calib,
                                              min_throttled=0)
    slopes, taus = [], []
    for s in scenarios:
        if s["profile"] not in ("ramp", "sustained"):
            continue
        t, traj, t_thresh = s["t"], s["traj"], s["t_thresh"]
        p, t_die = traj[:, 0], traj[:, 1]
        healthy = t_die < t_thresh
        if healthy.sum() < 10:
            continue

        slope_raw, _ = np.polyfit(t_die[healthy], p[healthy], 1)
        p_idle_s, p_max_s = p[healthy].min(), p[healthy].max()
        if p_max_s - p_idle_s < 1e-6:
            continue
        slopes.append(float(slope_raw) / (p_max_s - p_idle_s))

        idx_healthy = np.where(healthy)[0]
        temp_idle_s = t_die[0]
        temp_asymp_s = t_die[idx_healthy[-20:]].mean() if len(idx_healthy) >= 20 else t_die[idx_healthy[-1]]
        tau = _tau_from_rise(t, t_die, t_onset=0.0, temp_idle=temp_idle_s, temp_asymp=temp_asymp_s)
        if tau is not None:
            taus.append(tau)

    return {
        "slope_norm_per_c": float(np.mean(slopes)) if slopes else None,
        "tau_s": float(np.mean(taus)) if taus else None,
        "n_used": len(slopes),
    }


def pct_diff(real, synth):
    """Écart en % de synth par rapport à real (référence = réel)."""
    if real is None or synth is None or real == 0:
        return None
    return 100.0 * (synth - real) / abs(real)


def main():
    calib = load_calibration()
    real = real_metrics()
    synth = synthetic_metrics(calib)

    print("\n=== validate_synthetic_realism : partie NON-throttling uniquement ===\n")
    print(f"réel   (gpu_log_perso, {real['p_idle']:.1f}W->{real['p_load_max']:.1f}W) : "
          f"pente P/T normalisée = {real['slope_norm_per_c']:.4f} /°C | tau montée = {real['tau_s']:.1f}s")
    print(f"synth. (scénarios ramp/sustained, partie T<t_thresh, n={synth['n_used']}) : "
          f"pente P/T normalisée = {synth['slope_norm_per_c']:.4f} /°C | tau montée = {synth['tau_s']:.1f}s")

    d_slope = pct_diff(real["slope_norm_per_c"], synth["slope_norm_per_c"])
    d_tau = pct_diff(real["tau_s"], synth["tau_s"])

    print(f"\nécart pente P/T (normalisée) : {d_slope:+.1f}%" if d_slope is not None
          else "\nécart pente P/T : non calculable")
    print(f"écart tau montée             : {d_tau:+.1f}%" if d_tau is not None
          else "écart tau montée : non calculable")
    print("\n(pas de test statistique -- repère simple pour juger si le synthétique reste dans "
          "le bon ordre de grandeur sur la partie NON-throttling, seule partie vérifiable sur "
          "données réelles.)")


if __name__ == "__main__":
    main()
