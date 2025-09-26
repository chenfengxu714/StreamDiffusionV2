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
    noise_scale: float
    overlap: int
    num_kv_cache: int
    debug: bool
    world_size: int
    max_outstanding: int
    schedule_block: bool

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
parser.add_argument("--config_path", type=str, default="../configs/wan_causal_dmd_v2v.yaml")
parser.add_argument("--checkpoint_folder", type=str, default="../ckpts/wan_causal_dmd_v2v")
parser.add_argument("--noise_scale", type=float, default=0.8)
parser.add_argument("--overlap", type=int, default=0)
parser.add_argument("--num_kv_cache", type=int, default=6)
parser.add_argument("--debug", type=bool, default=True)
parser.add_argument("--world_size", type=int, default=2)
parser.add_argument("--max_outstanding", type=int, default=2, help="max number of outstanding sends/recv to keep")
parser.add_argument("--schedule_block", action="store_true", default=False)

config = Args(**vars(parser.parse_args()))
config.pretty_print()
