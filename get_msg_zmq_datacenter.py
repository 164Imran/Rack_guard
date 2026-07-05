"""Receive rack-based GPU ZMQ PUB messages and print a readable summary."""

from __future__ import annotations

import argparse
import json
from typing import Any

import zmq


DEFAULT_PORT = 5555
DEFAULT_TOPIC_PREFIX = "datacenter"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Subscribe to GPU datacenter messages over ZMQ.")
    parser.add_argument("--host", default="localhost", help="Publisher host to connect to.")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Publisher port to connect to.")
    parser.add_argument(
        "--topic-prefix",
        default=DEFAULT_TOPIC_PREFIX,
        help="Topic prefix to subscribe to, for example datacenter.",
    )
    return parser


def _format_summary(data: dict[str, Any]) -> str:
    interesting_keys = [
        "rack",
        "gpu_id",
        "sample_index",
        "timestamp",
        "temperature.gpu",
        "temperature_gpu",
        "utilization.gpu [%]",
        "utilization_gpu_pct",
        "memory.used [MiB]",
        "memory_used_MiB",
        "power.draw [W]",
        "power_draw_W",
    ]

    parts = []
    for key in interesting_keys:
        if key in data:
            parts.append(f"{key}={data[key]}")

    if parts:
        return ", ".join(parts)
    return json.dumps(data, ensure_ascii=False, sort_keys=True)


def main() -> None:
    args = build_parser().parse_args()

    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.connect(f"tcp://{args.host}:{args.port}")
    socket.setsockopt_string(zmq.SUBSCRIBE, args.topic_prefix)

    print(f"Listening on tcp://{args.host}:{args.port} for topics starting with '{args.topic_prefix}'")
    while True:
        message = socket.recv_string()
        topic, payload = message.split(" ", 1)
        data = json.loads(payload)
        print(f"[{topic}] {_format_summary(data)}")


if __name__ == "__main__":
    main()
