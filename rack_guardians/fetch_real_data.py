"""
fetch_real_data.py — Calibration à partir d'un échantillon réel du MIT Supercloud Dataset.

Source : MIT Lincoln Laboratory Supercomputing Center — "MIT Supercloud Dataset"
(Ali et al., télémétrie datacenter GPU/CPU : power draw, utilisation, jobs scheduler).
On utilise ici le mirroir Kaggle "skylarkphantom/mit-datacenter-challenge-data"
(sous-ensemble exploitable du dataset complet de ~2 To hébergé sur dcc.mit.edu),
via `kagglehub`, PAS le site source qui héberge les fichiers bruts complets.

Ce que ce mirroir Kaggle couvre RÉELLEMENT (vérifié en inspectant dcgm.csv) :
puissance GPU (powerusage_watts_avg/max/min), énergie consommée, utilisation SM et
mémoire (%), bande passante PCIe. Un second fichier (scheduler_data.csv) contient des
métadonnées de jobs Slurm (pas de télémétrie matérielle).
Ce que ce mirroir NE couvre PAS DU TOUT : aucune colonne de température (ni GPU die,
ni mémoire), aucune vitesse de ventilateur (RPM), aucune température ambiante
explicite. Les statistiques de température ci-dessous viennent donc TOUJOURS des
valeurs de secours documentées plus bas (jamais du téléchargement réel), et le
ventilateur/l'ambiant sont simulés séparément dans generate_realistic_data.py à partir
de courbes physiques documentées, pas mesurées ici.

DEUXIÈME SOURCE — log nvidia-smi personnel (gpu_log_stress2.csv, GPU grand public) :
capture idle -> charge soutenue (util 0->100%) sur ~330s, utilisée UNIQUEMENT pour la
FORME de la dynamique de montée en charge (durée relative de la rampe avant
stabilisation thermique) et pour un coefficient simple utilisation->puissance, PAS
pour les ordres de grandeur absolus (GPU laptop ~40W/61°C vs GPU datacenter
~150-250W/70°C — échelles non transposables telles quelles). Voir
`compute_perso_gpu_calibration()` ci-dessous.

  - fan.speed [%] est [N/A] sur TOUTE la capture (cette carte grand public ne
    remonte pas cette métrique au driver) : cette colonne est explicitement IGNORÉE
    et ne doit JAMAIS être traitée comme une vérité terrain. Le comportement du
    ventilateur (vitesse, hystérésis) reste une variable 100% synthétique du
    simulateur (cf. generate_realistic_data.py), pas une donnée observée.
  - Ce log ne contient AUCUN franchissement de seuil de throttling : la carte
    plafonne à ~61°C sous charge soutenue et les clocks.current.sm ne chutent
    jamais pour une raison thermique. t_thresh (température de déclenchement du
    throttling, utilisée par simulator.py/generate_realistic_data.py) reste donc
    une HYPOTHÈSE NON CALIBRÉE sur données réelles, quelle que soit la source
    (ni MIT Supercloud qui n'a pas de colonne température, ni ce log perso qui ne
    montre jamais l'événement) : la valeur de secours (70°C) est un ordre de
    grandeur plausible choisi à la main, pas une mesure.

Objectif : calibrer des ORDRES DE GRANDEUR pour le simulateur, PAS entraîner un modèle
dessus directement (d'où le plafond de téléchargement très strict ci-dessous).

Exécutable seul : `python fetch_real_data.py` (doit tourner en < 2 minutes, fallback
automatique si Kaggle n'est pas configuré ou si le réseau est indisponible).
"""

import json
import os
import sys
import time

import numpy as np
import pandas as pd

PERSO_GPU_LOG_PATH = "gpu_log_stress2.csv"

# ============================================================
# CONFIG — plafond explicite de téléchargement (calibration, pas entraînement)
# ============================================================
KAGGLE_DATASET = "skylarkphantom/mit-datacenter-challenge-data"
MAX_ROWS = 200_000        # quelques dizaines de Mo max, largement suffisant pour des stats
MAX_FILE_SIZE_MB = 50     # sécurité supplémentaire : on ignore les fichiers plus gros que ça
DOWNLOAD_TIMEOUT_S = 60   # si kagglehub traîne, on abandonne et on bascule sur le fallback

OUTPUT_PATH = "calibration.json"

# ============================================================
# TABLE TDP GPU CIBLES (Crusoe) — valeurs PUBLIQUES officielles constructeur (pas
# mesurées), vérifiées via recherche web (fiches techniques/annonces NVIDIA et AMD,
# juillet 2026) :
#   - H100 (SXM5)                         : 700 W  (NVIDIA)
#   - H200 (SXM5)                         : 700 W  (même profil de puissance que H100, NVIDIA)
#   - B200 (config datacenter/HGX typique) : 1000 W (le plein-spec B200 isolé atteint 1200W,
#     mais la config datacenter/HGX standard est limitée à ~90% de ce spec, cf. Tweaktown/
#     IntuitionLabs) -- valeur "config typique", pas le maximum théorique du die.
#   - GB200 (GPU B200 dans le Superchip GB200 NVL, pleine puissance) : 1200 W par GPU
#     (le Superchip complet = 2x1200W GPU + ~300W Grace CPU = 2700W)
#   - MI300X (AMD)                        : 750 W  (AMD datasheet officiel)
#   - MI355X (AMD)                        : 1400 W (TDP AMD officiel ; puissance runtime
#     typique observée 943-1256W, on garde le TDP nominal comme p_load_max)
# Pas de p_idle par modèle (aucune source fiable publique par GPU) : generate_realistic_data.py
# applique le ratio idle/max calibré sur MIT Supercloud (power_idle_mean_w/power_load_max_w)
# à ce nouveau p_load_max, cf. commentaire dans generate_realistic_data.py.
# ============================================================
GPU_TDP_TABLE_W = {
    "H100": 700.0,
    "H200": 700.0,
    "B200": 1000.0,
    "GB200": 1200.0,
    "MI300X": 750.0,
    "MI355X": 1400.0,
}
GPU_TDP_SOURCES = {
    "H100": "NVIDIA (fiche technique H100 SXM5, TDP 700W)",
    "H200": "NVIDIA (fiche technique H200, même enveloppe de puissance que H100 SXM5)",
    "B200": "NVIDIA (config datacenter/HGX typique ~1000W ; plein-spec isolé jusqu'à 1200W)",
    "GB200": "NVIDIA (GPU B200 pleine puissance dans le Superchip GB200 NVL, 1200W/GPU)",
    "MI300X": "AMD (data sheet officiel AMD Instinct MI300X, TDP 750W)",
    "MI355X": "AMD (TDP officiel 1400W ; puissance runtime typique 943-1256W)",
}
# Pondération de tirage par modèle dans generate_realistic_data.py : hypothèse de mix de
# parc raisonnable (H100/H200 = génération mature/déployée en volume, B200/GB200/MI355X =
# génération plus récente/moins répandue) -- PAS une donnée de flotte Crusoe réelle, à
# ajuster si Crusoe communique un mix réel.
GPU_TDP_WEIGHTS = {
    "H100": 0.30, "H200": 0.20, "B200": 0.15, "GB200": 0.10, "MI300X": 0.15, "MI355X": 0.10,
}

# ============================================================
# VALEURS DE SECOURS (fallback) — ordre de grandeur si le téléchargement échoue.
# Sourcées de la même calibration réelle que simulator.py (cf. commentaire en tête
# de ce fichier + simulator.py) : puissance idle GPU ~25W, pic ~100W/GPU,
# température idle ~25°C, cohérent avec les statistiques publiées sur le MIT
# Supercloud Dataset pour des GPU de génération Volta-class sous charge modérée.
# ============================================================
FALLBACK_CALIBRATION = {
    "source": "fallback codé en dur (téléchargement Kaggle indisponible)",
    "power_idle_mean_w": 25.0,
    "power_idle_std_w": 3.0,
    "power_load_mean_w": 85.0,
    "power_load_max_w": 100.0,
    "temp_gpu_idle_mean_c": 25.0,
    "temp_gpu_idle_p5_c": 22.0,
    "temp_gpu_idle_p95_c": 29.0,
    "temp_gpu_load_mean_c": 55.0,
    "temp_gpu_load_p5_c": 45.0,
    "temp_gpu_load_p95_c": 70.0,
    "temp_mem_idle_mean_c": 22.0,
    "temp_mem_idle_p5_c": 20.0,
    "temp_mem_idle_p95_c": 26.0,
    "temp_mem_load_mean_c": 45.0,
    "temp_mem_load_p5_c": 38.0,
    "temp_mem_load_p95_c": 58.0,
    "corr_util_power": 0.85,
    "fallback_used": True,
}

# Noms de colonnes candidats (tolérant aux variantes de schéma Kaggle) : on cherche
# une sous-chaîne insensible à la casse plutôt qu'un nom exact, pour rester robuste
# si le schéma diffère légèrement de nos hypothèses. Pas de hint température : ce
# mirroir n'en contient aucune (vérifié), inutile de chercher une colonne absente.
COLUMN_HINTS = {
    "power": ["powerusage_watts_avg", "power_draw", "gpu_power", "power"],
    "util": ["smutilization_pct_avg", "avgsmutilization_pct", "gpu_util", "utilization", "util"],
}


def find_column(df, hints):
    """Cherche la première colonne dont le nom contient un des indices (insensible à la casse)."""
    cols_lower = {c.lower(): c for c in df.columns}
    for hint in hints:
        for lower_name, real_name in cols_lower.items():
            if hint in lower_name:
                return real_name
    return None


def download_sample():
    """Télécharge et charge un échantillon plafonné du dataset Kaggle. Renvoie un DataFrame ou None.

    Le timeout est appliqué avec un vrai coupe-circuit (thread + future.result(timeout=...)),
    pas une vérification a posteriori : sur un réseau lent, on abandonne activement au bout
    de DOWNLOAD_TIMEOUT_S plutôt que d'attendre la fin d'un téléchargement qu'on jettera
    de toute façon (le run précédent avait payé 196s pour un fallback -> corrigé ici).
    """
    try:
        import kagglehub
    except ImportError:
        print("[fetch_real_data] kagglehub non installé -> fallback.")
        return None

    from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

    start = time.perf_counter()
    print(f"[fetch_real_data] téléchargement de {KAGGLE_DATASET} (peut nécessiter une clé API Kaggle, "
          f"timeout dur {DOWNLOAD_TIMEOUT_S}s)...")
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(kagglehub.dataset_download, KAGGLE_DATASET)
        try:
            path = future.result(timeout=DOWNLOAD_TIMEOUT_S)
        except FutureTimeoutError:
            print(f"[fetch_real_data] téléchargement > {DOWNLOAD_TIMEOUT_S}s (réseau lent) -> abandon, fallback.")
            return None
        except Exception as e:
            print(f"[fetch_real_data] échec du téléchargement Kaggle ({e}) -> fallback.")
            return None

    elapsed = time.perf_counter() - start
    print(f"[fetch_real_data] dataset récupéré en {elapsed:.1f}s -> {path}")

    import os
    csv_files = []
    for root, _, files in os.walk(path):
        for f in files:
            if f.endswith(".csv"):
                full = os.path.join(root, f)
                size_mb = os.path.getsize(full) / (1024 * 1024)
                if size_mb <= MAX_FILE_SIZE_MB:
                    csv_files.append(full)

    if not csv_files:
        print("[fetch_real_data] aucun CSV exploitable trouvé (ou tous > MAX_FILE_SIZE_MB) -> fallback.")
        return None

    try:
        print(f"[fetch_real_data] lecture de {csv_files[0]} (max {MAX_ROWS} lignes)...")
        df = pd.read_csv(csv_files[0], nrows=MAX_ROWS)
        print(f"[fetch_real_data] {len(df)} lignes chargées, colonnes : {list(df.columns)[:10]}...")
        return df
    except Exception as e:
        print(f"[fetch_real_data] échec de lecture du CSV ({e}) -> fallback.")
        return None


def compute_calibration(df):
    """Calcule les statistiques de calibration à partir du DataFrame réel. Renvoie un dict ou None.

    Ce mirroir Kaggle ne contient AUCUNE colonne de température (vérifié) : seules les
    stats de puissance/utilisation/corrélation sont calculées à partir des données
    réelles. Les stats de température restent TOUJOURS celles du fallback documenté
    (FALLBACK_CALIBRATION), fusionnées ici avec les stats de puissance réelles.
    """
    try:
        col_power = find_column(df, COLUMN_HINTS["power"])
        col_util = find_column(df, COLUMN_HINTS["util"])

        if col_power is None or col_util is None:
            print("[fetch_real_data] colonnes power/util introuvables dans le schéma réel -> fallback.")
            return None

        power = df[col_power].dropna().astype(float)
        util = df[col_util].dropna().astype(float)
        common = power.index.intersection(util.index)
        power, util = power.loc[common], util.loc[common]

        idle_mask = util < 10.0        # utilisation < 10% -> idle
        load_mask = util > 60.0        # utilisation > 60% -> charge soutenue

        if not idle_mask.any() or not load_mask.any():
            print("[fetch_real_data] pas assez de lignes idle/load distinctes dans l'échantillon -> fallback.")
            return None

        # on part du fallback pour la température (non disponible dans ce dataset),
        # et on écrase seulement les champs puissance/corrélation avec les valeurs réelles.
        calib = dict(FALLBACK_CALIBRATION)
        calib.update({
            "source": f"Kaggle:{KAGGLE_DATASET}/dcgm.csv (échantillon {len(df)} lignes, "
                       f"colonnes détectées: power={col_power}, util={col_util}). "
                       f"Température : pas de colonne dans ce dataset -> fallback documenté conservé.",
            "power_idle_mean_w": float(power[idle_mask].mean()),
            "power_idle_std_w": float(power[idle_mask].std()),
            "power_load_mean_w": float(power[load_mask].mean()),
            "power_load_max_w": float(power[load_mask].max()),
            "corr_util_power": float(np.corrcoef(util, power)[0, 1]),
            "fallback_used": "partiel (puissance réelle, température fallback — dataset sans colonne temp)",
        })
        return calib
    except Exception as e:
        print(f"[fetch_real_data] erreur pendant le calcul des stats ({e}) -> fallback.")
        return None


def compute_perso_gpu_calibration(path=PERSO_GPU_LOG_PATH):
    """Calibration de FORME (pas d'échelle) à partir d'un log nvidia-smi personnel.

    Extrait des coefficients simples (pas un modèle complet) :
      - power_idle_mean_w_perso / power_load_mean_w_perso / power_load_max_w_perso,
        temp_idle_mean_c_perso / temp_load_mean_c_perso / temp_load_p95_c_perso :
        stats descriptives, JAMAIS utilisées telles quelles à l'échelle datacenter.
      - util_power_slope_perso / util_power_intercept_perso / corr_util_power_perso :
        régression linéaire simple power ~ a*util + b.
      - ramp_frac_perso : fraction de la durée totale du log nécessaire pour atteindre
        90% de l'excursion thermique idle->charge après le début de la rampe. Sert de
        proxy de FORME (vitesse relative de la montée en régime) réutilisé par
        generate_realistic_data.py pour caler la durée de rampe à l'échelle datacenter
        (300s), au lieu d'une constante arbitraire.

    fan.speed n'est jamais lu (toujours [N/A] dans ce log, cf. commentaire en tête de
    fichier). Ne calcule PAS t_thresh : ce log ne contient aucun throttling.
    """
    if not os.path.exists(path):
        print(f"[fetch_real_data] {path} introuvable -> pas de calibration perso (shape uniquement).")
        return None
    try:
        df = pd.read_csv(path)
        df.columns = [c.strip() for c in df.columns]
        df["util"] = pd.to_numeric(
            df["utilization.gpu [%]"].astype(str).str.replace("%", "", regex=False).str.strip(),
            errors="coerce")
        df["power"] = pd.to_numeric(
            df["power.draw [W]"].astype(str).str.replace("W", "", regex=False).str.strip(),
            errors="coerce")
        df["temp"] = pd.to_numeric(df["temperature.gpu"], errors="coerce")
        df["ts"] = pd.to_datetime(df["timestamp"].str.strip(), format="%Y/%m/%d %H:%M:%S.%f")
        df["t_s"] = (df["ts"] - df["ts"].iloc[0]).dt.total_seconds()

        first_load_idx = df.index[df["util"] > 10.0].min()
        if pd.isna(first_load_idx):
            print("[fetch_real_data] aucune charge détectée dans le log perso -> pas de calibration perso.")
            return None

        pre_ramp_idle = df.loc[:first_load_idx - 1]
        pre_ramp_idle = pre_ramp_idle[pre_ramp_idle["util"] < 5.0]
        load_rows = df[df["util"] >= 95.0]

        power_idle = float(pre_ramp_idle["power"].mean())
        temp_idle = float(pre_ramp_idle["temp"].mean())
        power_load_mean = float(load_rows["power"].mean())
        power_load_max = float(load_rows["power"].max())
        temp_load_mean = float(load_rows["temp"].mean())
        temp_load_p95 = float(load_rows["temp"].quantile(0.95))

        valid = df[["util", "power"]].dropna()
        slope, intercept = np.polyfit(valid["util"], valid["power"], 1)
        corr = float(np.corrcoef(valid["util"], valid["power"])[0, 1])

        t_onset = float(df.loc[first_load_idx, "t_s"])
        temp_asymp = float(load_rows["temp"].tail(20).mean())
        target = temp_idle + 0.9 * (temp_asymp - temp_idle)
        after_onset = df[df["t_s"] >= t_onset]
        hit = after_onset[after_onset["temp"] >= target]
        total_duration = float(df["t_s"].iloc[-1])
        if len(hit) == 0:
            ramp_frac = 0.4  # rampe jamais stabilisée dans la fenêtre capturée -> repli neutre
        else:
            ramp_frac = float((hit["t_s"].iloc[0] - t_onset) / total_duration)
        ramp_frac = float(np.clip(ramp_frac, 0.2, 0.6))  # borne de sécurité, évite une forme dégénérée

        return {
            "source": f"nvidia-smi perso ({path}, {len(df)} lignes, ~{total_duration:.0f}s, "
                       f"GPU grand public — fan.speed ignoré car [N/A] sur toute la capture, "
                       f"aucun throttling observé donc pas de calibration de t_thresh).",
            "power_idle_mean_w_perso": power_idle,
            "power_load_mean_w_perso": power_load_mean,
            "power_load_max_w_perso": power_load_max,
            "temp_idle_mean_c_perso": temp_idle,
            "temp_load_mean_c_perso": temp_load_mean,
            "temp_load_p95_c_perso": temp_load_p95,
            "util_power_slope_perso": float(slope),
            "util_power_intercept_perso": float(intercept),
            "corr_util_power_perso": corr,
            "ramp_frac_perso": ramp_frac,
            "sustained_frac_perso": float(np.clip(ramp_frac * 0.375, 0.05, 0.3)),
        }
    except Exception as e:
        print(f"[fetch_real_data] erreur pendant le calcul de la calibration perso ({e}) -> ignorée.")
        return None


def main():
    t0 = time.perf_counter()
    print("[fetch_real_data] démarrage...")

    df = download_sample()
    calib = compute_calibration(df) if df is not None else None

    if calib is None:
        print("[fetch_real_data] téléchargement échoué, utilisation des valeurs de calibration "
              "de secours (sourcées dans le commentaire en tête de fichier).")
        calib = dict(FALLBACK_CALIBRATION)

    sources = {k: "mit_supercloud" for k in calib if k not in ("source", "fallback_used")}

    perso_calib = compute_perso_gpu_calibration()
    if perso_calib is not None:
        calib["gpu_log_perso"] = perso_calib
        sources.update({k: "gpu_log_perso" for k in perso_calib if k != "source"})
        print(f"[fetch_real_data] calibration perso (forme) intégrée : "
              f"ramp_frac={perso_calib['ramp_frac_perso']:.3f} "
              f"power={perso_calib['power_idle_mean_w_perso']:.1f}->{perso_calib['power_load_mean_w_perso']:.1f}W "
              f"(échelle laptop, non transposée telle quelle).")
    else:
        print("[fetch_real_data] pas de calibration perso disponible -> generate_realistic_data.py "
              "gardera ses fractions de rampe par défaut codées en dur.")

    calib["sources"] = sources

    calib["gpu_tdp_table_w"] = GPU_TDP_TABLE_W
    calib["gpu_tdp_weights"] = GPU_TDP_WEIGHTS
    calib["gpu_tdp_sources"] = GPU_TDP_SOURCES
    print(f"[fetch_real_data] table TDP GPU cibles (Crusoe) ajoutée : "
          f"{', '.join(f'{k}={v:.0f}W' for k, v in GPU_TDP_TABLE_W.items())}")

    with open(OUTPUT_PATH, "w") as f:
        json.dump(calib, f, indent=2)

    elapsed = time.perf_counter() - t0
    print(f"[fetch_real_data] calibration écrite dans {OUTPUT_PATH} en {elapsed:.1f}s au total.")
    if elapsed > 120:
        print("[fetch_real_data] ATTENTION : dépassement du budget de 2 minutes, "
              "réduire MAX_ROWS/MAX_FILE_SIZE_MB.", file=sys.stderr)


if __name__ == "__main__":
    main()
