from typing import NamedTuple
import argparse
import os


class Args(NamedTuple):
    host: str
    port: int
    reload: bool
    mode: str
    max_queue_size: int
    timeout: float
    safety_checker: bool
    taesd: bool
    ssl_certfile: str
    ssl_keyfile: str
    debug: bool
    acceleration: str
    num_frames: int
    overlap: int
    num_inference_steps: int
    t_start: int
    model_path: str
    guidance_scale: float
    embedded_cfg_scale: float
    flow_shift: int
    flow_reverse: bool
    neg_prompt: str
    seed: int
    lora_checkpoint_dir: str
    cpu_offload: bool

    def pretty_print(self):
        print("\n")
        for field, value in self._asdict().items():
            print(f"{field}: {value}")
        print("\n")


MAX_QUEUE_SIZE = int(os.environ.get("MAX_QUEUE_SIZE", 0))
TIMEOUT = float(os.environ.get("TIMEOUT", 0))
SAFETY_CHECKER = os.environ.get("SAFETY_CHECKER", None) == "True"
USE_TAESD = os.environ.get("USE_TAESD", "True") == "True"
ACCELERATION = os.environ.get("ACCELERATION", "xformers")

default_host = os.getenv("HOST", "0.0.0.0")
default_port = int(os.getenv("PORT", "7860"))
default_mode = os.getenv("MODE", "default")

parser = argparse.ArgumentParser(description="Run the app")
parser.add_argument("--host", type=str, default=default_host, help="Host address")
parser.add_argument("--port", type=int, default=default_port, help="Port number")
parser.add_argument("--reload", action="store_true", help="Reload code on change")
parser.add_argument(
    "--mode", type=str, default=default_mode, help="App Inferece Mode: txt2img, img2img"
)
parser.add_argument(
    "--max-queue-size",
    dest="max_queue_size",
    type=int,
    default=MAX_QUEUE_SIZE,
    help="Max Queue Size",
)
parser.add_argument("--timeout", type=float, default=TIMEOUT, help="Timeout")
parser.add_argument(
    "--safety-checker",
    dest="safety_checker",
    action="store_true",
    default=SAFETY_CHECKER,
    help="Safety Checker",
)
parser.add_argument(
    "--taesd",
    dest="taesd",
    action="store_true",
    help="Use Tiny Autoencoder",
)
parser.add_argument(
    "--no-taesd",
    dest="taesd",
    action="store_false",
    help="Use Tiny Autoencoder",
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
parser.add_argument(
    "--debug",
    action="store_true",
    default=False,
    help="Debug",
)
parser.add_argument(
    "--acceleration",
    type=str,
    default=ACCELERATION,
    choices=["none", "xformers", "sfast", "tensorrt"],
    help="Acceleration",
)
parser.add_argument("--num_frames", type=int, default=17)
parser.add_argument("--overlap", type=int, default=1)
parser.add_argument("--num_inference_steps", type=int, default=4)
parser.add_argument("--t_start", type=int, default=1)
parser.add_argument("--model_path", type=str, default="../ckpt/DCM_WAN/")
parser.add_argument(
        "--guidance_scale",
        type=float,
        default=1.0,
        help="Classifier free guidance scale.",
    )
parser.add_argument(
    "--embedded_cfg_scale",
    type=float,
    default=1.0,
    help="Embedded classifier free guidance scale.",
)
parser.add_argument(
    "--flow_shift",
    type=int,
    default=3,
    help="Flow shift parameter."
)
parser.add_argument(
    "--flow-reverse",
    action="store_true",
    default=True,
    help="If reverse, learning/sampling from t=1 -> t=0.",
)
parser.add_argument(
    "--neg_prompt",
    type=str,
    default=None,
    help="Negative prompt for sampling."
)
parser.add_argument(
    "--seed",
    type=int,
    default=0,
    help="Seed for evaluation."
)
parser.add_argument(
    "--lora_checkpoint_dir",
    type=str,
    default=None,
    help="Path to the directory containing LoRA checkpoints",
)
parser.add_argument("--cpu_offload", action="store_true")

parser.set_defaults(taesd=USE_TAESD)
config = Args(**vars(parser.parse_args()))
config.pretty_print()
