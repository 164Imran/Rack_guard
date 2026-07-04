"""
fetch_real_data.py — Calibration à partir d'un échantillon réel du MIT Supercloud Dataset.

Source : MIT Lincoln Laboratory Supercomputing Center — "MIT Supercloud Dataset"
(Ali et al., télémétrie datacenter GPU/CPU : power draw, températures, utilisation).
On utilise ici le mirroir Kaggle "skylarkphantom/mit-datacenter-challenge-data"
(sous-ensemble exploitable du dataset complet de ~2 To hébergé sur dcc.mit.edu),
via `kagglehub`, PAS le site source qui héberge les fichiers bruts complets.

Ce que ce dataset couvre : puissance GPU, température die/mémoire GPU, utilisation.
Ce que ce dataset NE couvre PAS : vitesse de ventilateur (RPM), température ambiante
explicite du datacenter — ces deux grandeurs sont donc simulées séparément dans
generate_realistic_data.py à partir de courbes physiques documentées, pas mesurées ici.

Objectif : calibrer des ORDRES DE GRANDEUR pour le simulateur, PAS entraîner un modèle
dessus directement (d'où le plafond de téléchargement très strict ci-dessous).

Exécutable seul : `python fetch_real_data.py` (doit tourner en < 2 minutes, fallback
automatique si Kaggle n'est pas configuré ou si le réseau est indisponible).
"""

import json
import sys
import time

import numpy as np
import pandas as pd

# ============================================================
# CONFIG — plafond explicite de téléchargement (calibration, pas entraînement)
# ============================================================
KAGGLE_DATASET = "skylarkphantom/mit-datacenter-challenge-data"
MAX_ROWS = 200_000        # quelques dizaines de Mo max, largement suffisant pour des stats
MAX_FILE_SIZE_MB = 50     # sécurité supplémentaire : on ignore les fichiers plus gros que ça
DOWNLOAD_TIMEOUT_S = 60   # si kagglehub traîne, on abandonne et on bascule sur le fallback

OUTPUT_PATH = "calibration.json"

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
# si le schéma diffère légèrement de nos hypothèses.
COLUMN_HINTS = {
    "power": ["gpu_power", "power_draw", "power"],
    "gpu_temp": ["gpu_temp", "gputemperature", "temperature"],
    "mem_temp": ["mem_temp", "memory_temp", "hbm_temp"],
    "util": ["gpu_util", "utilization", "gpu_usage"],
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
    """Télécharge et charge un échantillon plafonné du dataset Kaggle. Renvoie un DataFrame ou None."""
    try:
        import kagglehub
    except ImportError:
        print("[fetch_real_data] kagglehub non installé -> fallback.")
        return None

    start = time.perf_counter()
    try:
        print(f"[fetch_real_data] téléchargement de {KAGGLE_DATASET} (peut nécessiter une clé API Kaggle)...")
        path = kagglehub.dataset_download(KAGGLE_DATASET)
        elapsed = time.perf_counter() - start
        print(f"[fetch_real_data] dataset récupéré en {elapsed:.1f}s -> {path}")
    except Exception as e:
        print(f"[fetch_real_data] échec du téléchargement Kaggle ({e}) -> fallback.")
        return None

    if time.perf_counter() - start > DOWNLOAD_TIMEOUT_S:
        print("[fetch_real_data] téléchargement trop long -> fallback.")
        return None

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
    """Calcule les statistiques de calibration à partir du DataFrame réel. Renvoie un dict ou None."""
    try:
        col_power = find_column(df, COLUMN_HINTS["power"])
        col_gpu_temp = find_column(df, COLUMN_HINTS["gpu_temp"])
        col_mem_temp = find_column(df, COLUMN_HINTS["mem_temp"])
        col_util = find_column(df, COLUMN_HINTS["util"])

        if col_power is None or col_gpu_temp is None or col_util is None:
            print("[fetch_real_data] colonnes attendues introuvables dans le schéma réel -> fallback.")
            return None

        power = df[col_power].dropna().astype(float)
        gpu_temp = df[col_gpu_temp].dropna().astype(float)
        util = df[col_util].dropna().astype(float)

        idle_mask = util < 10.0        # utilisation < 10% -> idle
        load_mask = util > 60.0        # utilisation > 60% -> charge soutenue

        calib = {
            "source": f"Kaggle:{KAGGLE_DATASET} (échantillon {len(df)} lignes, colonnes détectées: "
                       f"power={col_power}, gpu_temp={col_gpu_temp}, mem_temp={col_mem_temp}, util={col_util})",
            "power_idle_mean_w": float(power[idle_mask].mean()) if idle_mask.any() else None,
            "power_idle_std_w": float(power[idle_mask].std()) if idle_mask.any() else None,
            "power_load_mean_w": float(power[load_mask].mean()) if load_mask.any() else None,
            "power_load_max_w": float(power[load_mask].max()) if load_mask.any() else None,
            "temp_gpu_idle_mean_c": float(gpu_temp[idle_mask].mean()) if idle_mask.any() else None,
            "temp_gpu_idle_p5_c": float(gpu_temp[idle_mask].quantile(0.05)) if idle_mask.any() else None,
            "temp_gpu_idle_p95_c": float(gpu_temp[idle_mask].quantile(0.95)) if idle_mask.any() else None,
            "temp_gpu_load_mean_c": float(gpu_temp[load_mask].mean()) if load_mask.any() else None,
            "temp_gpu_load_p5_c": float(gpu_temp[load_mask].quantile(0.05)) if load_mask.any() else None,
            "temp_gpu_load_p95_c": float(gpu_temp[load_mask].quantile(0.95)) if load_mask.any() else None,
            "corr_util_power": float(np.corrcoef(util, power)[0, 1]),
            "fallback_used": False,
        }

        if col_mem_temp is not None:
            mem_temp = df[col_mem_temp].dropna().astype(float)
            calib.update({
                "temp_mem_idle_mean_c": float(mem_temp[idle_mask].mean()) if idle_mask.any() else None,
                "temp_mem_idle_p5_c": float(mem_temp[idle_mask].quantile(0.05)) if idle_mask.any() else None,
                "temp_mem_idle_p95_c": float(mem_temp[idle_mask].quantile(0.95)) if idle_mask.any() else None,
                "temp_mem_load_mean_c": float(mem_temp[load_mask].mean()) if load_mask.any() else None,
                "temp_mem_load_p5_c": float(mem_temp[load_mask].quantile(0.05)) if load_mask.any() else None,
                "temp_mem_load_p95_c": float(mem_temp[load_mask].quantile(0.95)) if load_mask.any() else None,
            })
        else:
            print("[fetch_real_data] pas de colonne mémoire trouvée -> valeurs mem_temp non calculées (None).")
            for k in ["temp_mem_idle_mean_c", "temp_mem_idle_p5_c", "temp_mem_idle_p95_c",
                      "temp_mem_load_mean_c", "temp_mem_load_p5_c", "temp_mem_load_p95_c"]:
                calib[k] = None

        # si une statistique clé n'a pas pu être calculée (pas assez de lignes idle/load), on ne
        # fait pas semblant : on retombe proprement sur le fallback plutôt que publier du None.
        required = ["power_idle_mean_w", "power_load_mean_w", "temp_gpu_idle_mean_c", "temp_gpu_load_mean_c"]
        if any(calib[k] is None for k in required):
            print("[fetch_real_data] statistiques clés manquantes (pas assez de données idle/load) -> fallback.")
            return None

        return calib
    except Exception as e:
        print(f"[fetch_real_data] erreur pendant le calcul des stats ({e}) -> fallback.")
        return None


def main():
    t0 = time.perf_counter()
    print("[fetch_real_data] démarrage...")

    df = download_sample()
    calib = compute_calibration(df) if df is not None else None

    if calib is None:
        print("[fetch_real_data] téléchargement échoué, utilisation des valeurs de calibration "
              "de secours (sourcées dans le commentaire en tête de fichier).")
        calib = FALLBACK_CALIBRATION

    with open(OUTPUT_PATH, "w") as f:
        json.dump(calib, f, indent=2)

    elapsed = time.perf_counter() - t0
    print(f"[fetch_real_data] calibration écrite dans {OUTPUT_PATH} en {elapsed:.1f}s au total.")
    if elapsed > 120:
        print("[fetch_real_data] ATTENTION : dépassement du budget de 2 minutes, "
              "réduire MAX_ROWS/MAX_FILE_SIZE_MB.", file=sys.stderr)


if __name__ == "__main__":
    main()
