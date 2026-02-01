from importlib import import_module
from types import ModuleType
from PIL import Image
import io
import time
import numpy as np
import torch


def get_pipeline_class(pipeline_name: str) -> ModuleType:
    try:
        module = import_module(f"pipelines.{pipeline_name}")
    except ModuleNotFoundError:
        raise ValueError(f"Pipeline {pipeline_name} module not found")

    pipeline_class = getattr(module, "Pipeline", None)

    if pipeline_class is None:
        raise ValueError(f"'Pipeline' class not found in module '{pipeline_name}'.")

    return pipeline_class


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


def read_images_from_queue(queue, num_frames_needed, device, stop_event=None, dynamic_batch=False):
    # Wait until we have enough frames
    while queue.qsize() < num_frames_needed:
        if stop_event and stop_event.is_set():
            return None
        time.sleep(0.05)

    # Read exactly num_frames_needed frames in order (FIFO), don't discard any frames
    if dynamic_batch:
        num_frames_needed = queue.qsize()//num_frames_needed * num_frames_needed
    # print(f"Reading {num_frames_needed} frames from queue of size {queue.qsize()}")
    images = []
    for _ in range(num_frames_needed):
        images.append(queue.get())

    # Stack images in order (FIFO)
    images = np.stack(images, axis=0)
    images = torch.from_numpy(images).unsqueeze(0)
    images = images.permute(0, 4, 1, 2, 3).to(dtype=torch.bfloat16).to(device=device)
    return images


def select_images(images, num_images: int):
    if len(images) < num_images:
        return []
    step = len(images) / (num_images - 1)
    indices = [int(i * step) for i in range(num_images - 1)] + [-1]
    selected_images = np.stack([images[i] for i in indices], axis=0)
    return selected_images


def clear_queue(queue):
    while queue.qsize() > 0:
        queue.get()


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
