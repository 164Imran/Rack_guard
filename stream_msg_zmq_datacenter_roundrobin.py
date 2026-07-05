"""Round-robin ZMQ producer for GPU CSVs under datacenter_recorded/.

It walks every rack folder, then every gpu_log_stress*.csv file inside it,
and publishes one row at a time while rotating through all GPUs.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import zmq


DEFAULT_ROOT = Path("datacenter_recorded")
DEFAULT_PORT = 5555
DEFAULT_TOPIC_PREFIX = "datacenter"


@dataclass
class StreamState:
    rack: str
    csv_path: Path
    rows: list[dict]
    index: int = 0


def iter_gpu_files(root: Path):
    for rack_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for csv_path in sorted(rack_dir.glob("gpu_log_stress*.csv")):
            yield rack_dir.name, csv_path


def load_streams(root: Path) -> list[StreamState]:
    streams: list[StreamState] = []
    for rack, csv_path in iter_gpu_files(root):
        df = pd.read_csv(csv_path, skipinitialspace=True)
        rows = df.to_dict(orient="records")
        streams.append(StreamState(rack=rack, csv_path=csv_path, rows=rows))
    return streams


def publish_round_robin(socket: zmq.Socket, streams: list[StreamState], topic_prefix: str, delay: float) -> int:
    sent = 0
    active = True

    while active:
        active = False
        for stream in streams:
            if stream.index >= len(stream.rows):
                continue

            active = True
            row = dict(stream.rows[stream.index])
            row["rack"] = stream.rack
            row["gpu_id"] = stream.csv_path.stem
            row["sample_index"] = stream.index
            row["source_file"] = str(stream.csv_path)

            topic = f"{topic_prefix}.{stream.rack}.{stream.csv_path.stem}"
            socket.send_string(f"{topic} {json.dumps(row, default=str, ensure_ascii=False)}")
            print(f"Sent {topic}: sample {stream.index}")

            stream.index += 1
            sent += 1

            if delay > 0:
                time.sleep(delay)

    return sent


def main() -> None:
    parser = argparse.ArgumentParser(description="Round-robin GPU datacenter ZMQ publisher.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="Root folder containing rack folders.")
    parser.add_argument("--host", default="*", help="Bind host for the publisher.")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Bind port for the publisher.")
    parser.add_argument("--topic-prefix", default=DEFAULT_TOPIC_PREFIX, help="Topic prefix to publish.")
    parser.add_argument("--delay", type=float, default=0.05, help="Delay between messages, in seconds.")
    parser.add_argument("--loop", action="store_true", help="Replay the whole folder forever.")
    args = parser.parse_args()

    if not args.root.exists():
        raise FileNotFoundError(f"Folder not found: {args.root}")

    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.bind(f"tcp://{args.host}:{args.port}")

    print(f"Publishing from {args.root} on tcp://{args.host}:{args.port}")
    time.sleep(1)

    while True:
        streams = load_streams(args.root)
        if not streams:
            print(f"No gpu_log_stress*.csv files found under {args.root}")
            break

        total = publish_round_robin(socket, streams, args.topic_prefix, args.delay)
        print(f"Replay complete: {total} messages sent")

        if not args.loop:
            break


if __name__ == "__main__":
    main()
