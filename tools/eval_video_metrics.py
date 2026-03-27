#!/usr/bin/env python3
"""Evaluate frame fidelity and temporal consistency between two videos."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from skimage.metrics import structural_similarity as structural_similarity
from torchvision.io import read_video


def load_video(path: Path) -> tuple[torch.Tensor, dict]:
    frames, _, info = read_video(str(path), pts_unit="sec", output_format="TCHW")
    return frames.float() / 255.0, info


def compute_psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = torch.mean((a - b) ** 2).item()
    return 99.0 if mse == 0.0 else 10.0 * math.log10(1.0 / mse)


def compute_ssim(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(
        structural_similarity(
            a.permute(1, 2, 0).numpy(),
            b.permute(1, 2, 0).numpy(),
            channel_axis=2,
            data_range=1.0,
        )
    )


def summarize(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True, help="Reference video path")
    parser.add_argument("--candidate", type=Path, required=True, help="Candidate video path")
    args = parser.parse_args()

    reference, reference_info = load_video(args.reference)
    candidate, candidate_info = load_video(args.candidate)

    frame_count = min(reference.shape[0], candidate.shape[0])
    reference = reference[:frame_count]
    candidate = candidate[:frame_count]

    frame_psnr = [compute_psnr(reference[i], candidate[i]) for i in range(frame_count)]
    frame_ssim = [compute_ssim(reference[i], candidate[i]) for i in range(frame_count)]

    reference_delta = reference[1:] - reference[:-1]
    candidate_delta = candidate[1:] - candidate[:-1]
    temporal_mae = torch.mean(torch.abs(reference_delta - candidate_delta)).item()
    temporal_energy_err = torch.mean(
        torch.abs(
            reference_delta.abs().mean(dim=(1, 2, 3))
            - candidate_delta.abs().mean(dim=(1, 2, 3))
        )
    ).item()

    result = {
        "reference_path": str(args.reference),
        "candidate_path": str(args.candidate),
        "reference_shape": list(reference.shape),
        "candidate_shape": list(candidate.shape),
        "reference_fps": reference_info.get("video_fps"),
        "candidate_fps": candidate_info.get("video_fps"),
        "frame_psnr": summarize(frame_psnr),
        "frame_ssim": summarize(frame_ssim),
        "temporal_psnr": compute_psnr(reference_delta, candidate_delta),
        "temporal_mae": temporal_mae,
        "temporal_energy_err": temporal_energy_err,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
