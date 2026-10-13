"""Measure temporal changes within and between three-frame latent chunks."""

import json
from pathlib import Path

import torch


def _validate_latent(latent, min_frames=1):
    if not isinstance(latent, torch.Tensor):
        raise TypeError("Latent must be a torch.Tensor.")
    if latent.ndim != 5 or latent.shape[0] != 1:
        raise ValueError(
            f"Latent must have shape (1, T, C, H, W); got {tuple(latent.shape)}."
        )
    if latent.shape[1] < min_frames:
        raise ValueError(f"At least {min_frames} latent frames are required.")
    if any(size == 0 for size in latent.shape[2:]):
        raise ValueError("Latent channel and spatial dimensions must be nonempty.")
    if latent.is_complex():
        raise TypeError("Latent must be real-valued.")


def load(path) -> torch.Tensor:
    """Load a saved (1, T, C, H, W) latent tensor onto the CPU."""
    latent = torch.load(Path(path), map_location="cpu", weights_only=True)
    _validate_latent(latent)
    return latent


@torch.inference_mode()
def d1(latent) -> tuple[float, float]:
    """Return (d1_inter_chunk, d1_intra_chunk) as Python floats.

    ``y[i]`` is the mean of ``abs(X[:, i+1] - X[:, i])`` over all
    channels and pixels. Inter-chunk errors use ``y[2], y[5], ...``;
    intra-chunk errors use the remaining elements of ``y[1:]``.
    Index zero is excluded from both averages. At least four frames are
    required so both groups contain an element.
    """
    _validate_latent(latent, min_frames=4)
    latent = latent if latent.dtype == torch.float64 else latent.float()
    y = (latent[:, 1:] - latent[:, :-1]).abs().mean(dim=(0, 2, 3, 4))
    indices = torch.arange(1, y.numel(), device=y.device)
    inter_chunk = y[2::3].mean().item()
    intra_chunk = y[1:][indices % 3 != 2].mean().item()
    return inter_chunk, intra_chunk


@torch.inference_mode()
def d2(latent) -> tuple[float, float]:
    """Return (d2_intra_chunk, d2_inter_chunk) as Python floats.

    ``y[i]`` is the mean of ``abs(X[:, i] + X[:, i+2] - 2*X[:, i+1])``
    over all channels and pixels. Intra-chunk errors use
    ``y[3], y[6], ...``; inter-chunk errors use the remaining elements of
    ``y[1:]``. Index zero is excluded from both averages. At least six
    frames are required so both groups contain an element.
    """
    _validate_latent(latent, min_frames=6)
    latent = latent if latent.dtype == torch.float64 else latent.float()
    y = (
        latent[:, :-2] + latent[:, 2:] - 2 * latent[:, 1:-1]
    ).abs().mean(dim=(0, 2, 3, 4))
    indices = torch.arange(1, y.numel(), device=y.device)
    intra_chunk = y[3::3].mean().item()
    inter_chunk = y[1:][indices % 3 != 0].mean().item()
    return intra_chunk, inter_chunk


def batch(dir_path, target_path) -> dict:
    """Save and return a JSON report containing averages and per-file metrics.

    Immediate ``.pt`` files (case insensitive) are processed in filename order.
    The ``average`` node contains all four chunk metrics plus ``d1_metric``
    and ``d2_metric``, the averages of each file's inter/intra chunk ratios.
    The ``files`` node maps each input filename to its four chunk metrics.
    Each file contributes equally to the averages. Output parent directories
    are created as needed.

    Raises ValueError for a directory without PT files or a video with fewer
    than six latent frames, since the requested averages would be undefined.
    A zero intra-chunk value also raises ValueError because its ratio is
    undefined.
    """
    dir_path = Path(dir_path)
    target_path = Path(target_path)
    if not dir_path.exists():
        raise FileNotFoundError(f"Latent directory not found: {dir_path}")
    if not dir_path.is_dir():
        raise NotADirectoryError(f"Latent path is not a directory: {dir_path}")

    paths = sorted(
        path for path in dir_path.iterdir()
        if path.is_file() and path.suffix.lower() == ".pt"
    )
    if not paths:
        raise ValueError(f"No .pt files found in: {dir_path}")

    files = {}
    for path in paths:
        latent = load(path)
        d1_inter_chunk, d1_intra_chunk = d1(latent)
        d2_intra_chunk, d2_inter_chunk = d2(latent)
        metrics = {
            "d1_intra_chunk": d1_intra_chunk,
            "d1_inter_chunk": d1_inter_chunk,
            "d2_intra_chunk": d2_intra_chunk,
            "d2_inter_chunk": d2_inter_chunk,
        }
        for order in ("d1", "d2"):
            if metrics[f"{order}_intra_chunk"] == 0:
                raise ValueError(
                    f"Cannot compute {order}_metric for {path.name}: "
                    f"{order}_intra_chunk is zero."
                )
        files[path.name] = metrics

    metric_names = (
        "d1_intra_chunk", "d1_inter_chunk", "d2_intra_chunk", "d2_inter_chunk"
    )
    report = {
        "average": {
            name: sum(metrics[name] for metrics in files.values()) / len(files)
            for name in metric_names
        },
        "files": files,
    }
    for order in ("d1", "d2"):
        report["average"][f"{order}_metric"] = sum(
            metrics[f"{order}_inter_chunk"] / metrics[f"{order}_intra_chunk"]
            for metrics in files.values()
        ) / len(files)
    # Reject nonfinite metrics before opening or replacing the report file.
    text = json.dumps(report, indent=2, allow_nan=False) + "\n"
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(text, encoding="utf-8")
    return report


if __name__=='__main__':
    batch('temp/1001_bid', 'temp/1001_bid_metric.json')
    batch('temp/1001_dc', 'temp/1001_dc_metric.json')
    print("Voila.")