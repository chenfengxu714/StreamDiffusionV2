from typing import NamedTuple
import argparse
import os


class Args(NamedTuple):
    host: str
    port: int
    max_queue_size: int
    timeout: float
    ssl_certfile: str
    ssl_keyfile: str
    config_path: str
    checkpoint_folder: str
    step: int
    noise_scale: float
    debug: bool
    num_gpus: int
    gpu_ids: str
    max_outstanding: int
    schedule_block: bool
    model_type: str
    enable_metrics: bool
    target_latency: float

    def pretty_print(self):
        print("\n")
        for field, value in self._asdict().items():
            print(f"{field}: {value}")
        print("\n")


MAX_QUEUE_SIZE = int(os.environ.get("MAX_QUEUE_SIZE", 0))
TIMEOUT = float(os.environ.get("TIMEOUT", 0))

default_host = os.getenv("HOST", "0.0.0.0")
default_port = int(os.getenv("PORT", "7860"))

parser = argparse.ArgumentParser(description="Run the app")
parser.add_argument("--host", type=str, default=default_host, help="Host address")
parser.add_argument("--port", type=int, default=default_port, help="Port number")
parser.add_argument(
    "--max-queue-size",
    dest="max_queue_size",
    type=int,
    default=MAX_QUEUE_SIZE,
    help="Max Queue Size",
)
parser.add_argument(
    "--ssl-certfile",
    dest="ssl_certfile",
    type=str,
    default=None,
    help="SSL certfile",
)
parser.add_argument(
    "--ssl-keyfile",
    dest="ssl_keyfile",
    type=str,
    default=None,
    help="SSL keyfile",
)
parser.add_argument("--timeout", type=float, default=TIMEOUT, help="Timeout")

# This is the default config for the pipeline, it can be overridden by the command line arguments
parser.add_argument("--config_path", type=str, default="../configs/wan_causal_dmd_v2v.yaml")
parser.add_argument("--checkpoint_folder", type=str, default="../ckpts/wan_causal_dmd_v2v")
parser.add_argument("--step", type=int, default=2)
parser.add_argument("--noise_scale", type=float, default=0.8)
parser.add_argument("--debug", type=bool, default=True)
parser.add_argument("--num_gpus", type=int, default=2)
parser.add_argument("--gpu_ids", type=str, default="0,1") # id separated by comma, size should match num_gpus

# These are only used when num_gpus > 1
parser.add_argument("--max_outstanding", type=int, default=2, help="max number of outstanding sends/recv to keep")
parser.add_argument("--schedule_block", action="store_true", default=False)
parser.add_argument("--model_type", type=str, default="T2V-1.3B", help="Model type (e.g., T2V-1.3B)")

# Metrics collection
parser.add_argument("--enable-metrics", dest="enable_metrics", action="store_true", default=False, help="Enable SLO metrics collection")
parser.add_argument("--target-latency", dest="target_latency", type=float, default=1.0, help="Target latency in seconds for deadline miss rate calculation (default: 0.5s)")

config = Args(**vars(parser.parse_args()))
