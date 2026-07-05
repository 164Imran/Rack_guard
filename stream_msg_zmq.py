# producer_csv.py
import zmq
import json
import time
import pandas as pd
import glob

context = zmq.Context()
socket = context.socket(zmq.PUB)
socket.bind("tcp://*:5555")

time.sleep(1)  # laisse le temps aux abonnés de se connecter avant le premier envoi

def replay_csv(filepath, topic="gpu_channel", speed=1.0):
    df = pd.read_csv(filepath)
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    prev_time = None
    for _, row in df.iterrows():
        if prev_time is not None:
            delta = (row['timestamp'] - prev_time).total_seconds() / speed
            time.sleep(max(0, delta))
        prev_time = row['timestamp']

        data = row.to_dict()
        data['timestamp'] = str(data['timestamp'])
        data['source_file'] = filepath

        message = f"{topic} {json.dumps(data)}"
        socket.send_string(message)
        print("Envoyé :", data)

# Rejoue tous les fichiers stress trouvés, dans l'ordre
files = sorted(glob.glob("gpu_log_stress*.csv"))

if not files:
    print("Aucun fichier gpu_log_stress*.csv trouvé dans le dossier courant.")
else:
    for f in files:
        print(f"\n--- Rejeu de {f} ---")
        replay_csv(f, topic="gpu_channel", speed=1.0)  # speed=10.0 pour rejouer 10x plus vite

print("\nFin du rejeu de tous les fichiers.")