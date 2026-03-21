"""Small helpers shared by the demo pipelines."""

from PIL import Image
import io
import logging
import os
import time
import numpy as np
import torch

LOGGER = logging.getLogger(__name__)

BF16_BYTES = 2
KV_HEAD_DIM = 128
STREAM_BATCH_HEADROOM_BYTES = 1024**3
STREAM_BATCH_SAFETY_FACTOR = 1.15
MODEL_LAYOUTS = {
    "T2V-1.3B": {"num_transformer_blocks": 30, "num_heads": 12},
    "T2V-14B": {"num_transformer_blocks": 40, "num_heads": 40},
}


def bytes_to_pil(image_bytes: bytes) -> Image.Image:
    image = Image.open(io.BytesIO(image_bytes))
    return image


def pil_to_frame(image: Image.Image) -> bytes:
    frame_data = io.BytesIO()
    image.save(frame_data, format="JPEG")
    frame_data = frame_data.getvalue()
    return (
        b"--frame\r\n"
        + b"Content-Type: image/jpeg\r\n"
        + f"Content-Length: {len(frame_data)}\r\n\r\n".encode()
        + frame_data
        + b"\r\n"
    )


def is_firefox(user_agent: str) -> bool:
    return "Firefox" in user_agent


def read_images_from_queue(queue, num_frames_needed, device, stop_event=None):
    # Wait until we have enough frames
    while queue.qsize() < num_frames_needed:
        if stop_event and stop_event.is_set():
            return None
        time.sleep(0.01)

    # Read exactly num_frames_needed frames in order (FIFO), don't discard any frames.
    images = []
    for _ in range(num_frames_needed):
        images.append(queue.get())

    # Stack images in order (FIFO)
    images = np.stack(images, axis=0)
    images = torch.from_numpy(images).unsqueeze(0)
    images = images.permute(0, 4, 1, 2, 3).to(dtype=torch.bfloat16).to(device=device)
    return images


def clear_queue(queue):
    while queue.qsize() > 0:
        queue.get()


def _config_value(config, key, default=None):
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


def dump_pydantic_model(model) -> dict:
    """Serialize a Pydantic model across v1/v2 without deprecation warnings."""
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def parse_gpu_ids(gpu_ids: str) -> list[int]:
    return [int(gpu_id.strip()) for gpu_id in gpu_ids.split(",") if gpu_id.strip()]


def resolve_worker_device(gpu_ids: str, rank: int) -> torch.device:
    """
    Resolve the correct CUDA device for a worker rank.

    When `CUDA_VISIBLE_DEVICES` is set, torch renumbers the visible devices to
    `cuda:0..N-1`. The demo still accepts physical GPU IDs, so this helper maps
    them back to the correct local index.
    """
    requested_ids = parse_gpu_ids(gpu_ids)
    target_id = requested_ids[rank]
    visible_env = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()

    if not visible_env:
        return torch.device(f"cuda:{target_id}")

    visible_ids = parse_gpu_ids(visible_env)
    if target_id in visible_ids:
        return torch.device(f"cuda:{visible_ids.index(target_id)}")

    if 0 <= target_id < len(visible_ids):
        return torch.device(f"cuda:{target_id}")

    LOGGER.warning(
        "GPU id %s is not present in CUDA_VISIBLE_DEVICES=%s; falling back to local rank %s",
        target_id,
        visible_env,
        rank,
    )
    return torch.device(f"cuda:{rank}")


def compute_stream_token_shapes(width: int, height: int) -> dict[str, int]:
    """Compute the VAE-latent and DiT-token grids for the current video size."""
    latent_width = width // 8
    latent_height = height // 8
    token_width = width // 16
    token_height = height // 16
    return {
        "latent_width": latent_width,
        "latent_height": latent_height,
        "token_width": token_width,
        "token_height": token_height,
        "token_count": token_width * token_height,
    }


def get_model_layout(config) -> dict[str, int]:
    model_type = _config_value(config, "model_type", "T2V-1.3B")
    return MODEL_LAYOUTS.get(model_type, MODEL_LAYOUTS["T2V-1.3B"])


def get_num_transformer_blocks(config) -> int:
    return int(get_model_layout(config)["num_transformer_blocks"])


def infer_stream_dimensions(config) -> tuple[int, int]:
    """Infer pixel-space width/height from explicit config or latent-shape metadata."""
    width = _config_value(config, "width")
    height = _config_value(config, "height")
    if width is not None and height is not None:
        return int(width), int(height)

    image_or_video_shape = _config_value(config, "image_or_video_shape")
    if image_or_video_shape and len(image_or_video_shape) >= 5:
        latent_height = int(image_or_video_shape[-2])
        latent_width = int(image_or_video_shape[-1])
        return latent_width * 8, latent_height * 8

    raise ValueError("Unable to infer demo stream width/height from config")


def estimate_stream_batch_extra_memory_bytes(config, width: int, height: int) -> int:
    """
    Estimate the extra CUDA memory required by stream-batch over no-batch mode.

    The main delta comes from repeating the KV cache across denoising steps in
    `prepare(..., batch_denoise=True)`, plus the batched hidden-state buffer.
    """
    non_terminal_steps = [
        int(step)
        for step in _config_value(config, "denoising_step_list", [])
        if int(step) != 0
    ]
    num_steps = len(non_terminal_steps)
    if num_steps <= 1:
        return 0

    model_layout = get_model_layout(config)
    shapes = compute_stream_token_shapes(width, height)

    num_frame_per_block = int(_config_value(config, "num_frame_per_block", 1))
    num_kv_cache = int(_config_value(config, "num_kv_cache", 6))
    kv_cache_length = shapes["token_count"] * num_kv_cache

    kv_bytes_per_step = (
        model_layout["num_transformer_blocks"]
        * kv_cache_length
        * model_layout["num_heads"]
        * KV_HEAD_DIM
        * 2  # K and V
        * BF16_BYTES
    )
    hidden_state_bytes = (
        num_steps
        * num_frame_per_block
        * 16
        * shapes["latent_height"]
        * shapes["latent_width"]
        * BF16_BYTES
    )
    return (num_steps - 1) * kv_bytes_per_step + hidden_state_bytes


def select_stream_execution_mode(config, device: torch.device) -> dict[str, object]:
    """
    Choose between batched and no-batch online inference based on free CUDA memory.
    """
    width, height = infer_stream_dimensions(config)
    shapes = compute_stream_token_shapes(width=width, height=height)
    estimate_bytes = estimate_stream_batch_extra_memory_bytes(
        config,
        width=width,
        height=height,
    )
    required_bytes = int(estimate_bytes * STREAM_BATCH_SAFETY_FACTOR) + STREAM_BATCH_HEADROOM_BYTES

    if device.type != "cuda":
        return {
            "mode": "stream_batch",
            "free_bytes": None,
            "required_bytes": required_bytes,
            "estimated_extra_bytes": estimate_bytes,
            "shapes": shapes,
        }

    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    use_stream_batch = free_bytes >= required_bytes
    mode = "stream_batch" if use_stream_batch else "stream_wo_batch"

    LOGGER.info(
        "Online inference mode=%s, free=%.2f GiB, required=%.2f GiB, estimated_extra=%.2f GiB, latent=%sx%s, tokens=%sx%s",
        mode,
        free_bytes / 1024**3,
        required_bytes / 1024**3,
        estimate_bytes / 1024**3,
        shapes["latent_width"],
        shapes["latent_height"],
        shapes["token_width"],
        shapes["token_height"],
    )
    return {
        "mode": mode,
        "free_bytes": free_bytes,
        "total_bytes": total_bytes,
        "required_bytes": required_bytes,
        "estimated_extra_bytes": estimate_bytes,
        "shapes": shapes,
    }


def image_to_array(
        image: Image.Image,
        width: int,
        height: int,
        normalize: bool = True
    ) -> np.ndarray:
        image = image.convert("RGB").resize((width, height))
        image_array = np.array(image)
        if normalize:
            image_array = image_array / 127.5 - 1.0
        return image_array


def array_to_image(image_array: np.ndarray, normalize: bool = True) -> Image.Image:
    if normalize:
        image_array = image_array * 255.0
    image_array = image_array.astype(np.uint8)
    image = Image.fromarray(image_array)
    return image
