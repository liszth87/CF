"""Encode video files into the normalized Wan2.1 VAE latent space."""

from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F

from tqdm import tqdm


DEFAULT_VAE_PATH = "/mnt/ddn/zhenhao_team/models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth"


def _load_vae(vae_path):
    vae_path = Path(vae_path)
    if vae_path.is_dir():
        vae_path = vae_path / "Wan2.1_VAE.pth"
    if not vae_path.is_file():
        raise FileNotFoundError(f"VAE checkpoint not found: {vae_path}")

    from wan.modules.vae import WanVAE

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae = WanVAE(vae_pth=str(vae_path), dtype=torch.float32, device=device)
    vae.model.to(dtype=torch.float32)
    return vae


def _encode_video_with_vae(video_path, vae) -> torch.Tensor:
    """Encode one video using an already loaded VAE."""
    with imageio.get_reader(str(video_path), format="ffmpeg") as reader:
        frames = [frame for frame in reader]
    if not frames:
        raise ValueError(f"Video contains no decodable frames: {video_path}")

    # FFmpeg returns uint8 RGB frames in [T, H, W, C] order.
    video = torch.from_numpy(np.stack(frames)).permute(3, 0, 1, 2).unsqueeze(0)
    del frames
    video = video.to(device=vae.device, dtype=torch.float32).div_(127.5).sub_(1.0)

    num_frames, height, width = video.shape[2:]
    padding = (0, (-width) % 8, 0, (-height) % 8, 0, (1 - num_frames) % 4)
    if any(padding):
        video = F.pad(video, padding, mode="replicate")

    # Use the model directly so CPU inference does not enter CUDA autocast.
    latent = vae.model.encode(video, vae.scale)
    return latent.permute(0, 2, 1, 3, 4).float().cpu().contiguous()


@torch.inference_mode()
def encode_video(
    video_path,
    vae_path=DEFAULT_VAE_PATH,
) -> torch.Tensor:
    """Load a video and return its normalized, deterministic VAE latents.

    Args:
        video_path: Path to a video file supported by ImageIO's FFmpeg reader.
        vae_path: Path to ``Wan2.1_VAE.pth``, or a directory containing it.
            Defaults to the checkpoint used by ``inference.py`` through
            ``utils.wan_wrapper.WanVAEWrapper``.

    Returns:
        A contiguous float32 CPU tensor with shape
        ``[1, latent_frames, 16, latent_height, latent_width]``, matching
        ``WanVAEWrapper.encode_to_latent``. Encoding uses CUDA when available
        and otherwise runs on the CPU.

    All frames are read at their original resolution and frame rate. Pixels
    are converted to RGB and normalized to [-1, 1]. The final frame is repeated
    to reach a frame count of 4*n+1; the bottom and right edges are replicated
    to make the height and width divisible by 8. Thus ``latent_frames`` is
    ``1 + ceil((frames - 1) / 4)``, and the spatial latent dimensions are
    ``ceil(height / 8)`` and ``ceil(width / 8)``. No frames are discarded.

    Example:
        latents = encode_video("input.mp4")
        torch.save(latents, "input_latents.pt")
    """
    video_path = Path(video_path)
    if not video_path.is_file():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    vae = _load_vae(vae_path)
    return _encode_video_with_vae(video_path, vae)


@torch.inference_mode()
def batch_process_dir(dir_source, dir_target, vae_path=DEFAULT_VAE_PATH) -> list[Path]:
    """Encode MP4s whose filenames contain ``bustling``, loading the VAE once.

    Args:
        dir_source: Directory containing the input videos. Only immediate
            files with a ``.mp4`` extension (case insensitive) and ``bustling``
            in their filename (case sensitive) are processed.
        dir_target: Output directory, created automatically if needed.
            ``clip.mp4`` is saved as ``clip.pt``; existing outputs are replaced.
        vae_path: VAE checkpoint or checkpoint directory, with the same default
            as ``encode_video``.

    Returns:
        A list of saved ``.pt`` paths in input filename order. Each file contains
        the same float32 CPU latent tensor returned by ``encode_video``.
        An empty source directory returns an empty list without loading a VAE.

    Example:
        output_paths = batch_process_dir("videos", "video_latents")
    """
    dir_source = Path(dir_source)
    dir_target = Path(dir_target)
    if not dir_source.exists():
        raise FileNotFoundError(f"Source directory not found: {dir_source}")
    if not dir_source.is_dir():
        raise NotADirectoryError(f"Source path is not a directory: {dir_source}")

    video_paths = sorted(
        path for path in dir_source.iterdir()
        if path.is_file() and path.suffix.lower() == ".mp4"
    )
    dir_target.mkdir(parents=True, exist_ok=True)
    if not video_paths:
        return []

    vae = _load_vae(vae_path)
    output_paths = []
    for video_path in tqdm(video_paths):
        if "bustling" not in video_path.name:
            continue
        latent = _encode_video_with_vae(video_path, vae)
        output_path = dir_target / video_path.with_suffix(".pt").name
        torch.save(latent, output_path)
        output_paths.append(output_path)
    return output_paths


if __name__  == '__main__':
    # dir_source = '/mnt/ddn/george/diffusion/john-AR/vbench_results/261001_dawood-double_cache_14b_baseline_iter12k_bidirectional/videos_vbench'
    # dir_target = 'temp/1001_bid'

    dir_source = '/mnt/ddn/george/diffusion/john-AR/vbench_results/261001_dawood-double_cache_14b_baseline_iter12k/videos_vbench'
    dir_target = 'temp/1001_dc'

    batch_process_dir(dir_source, dir_target)
    print("Voila.")
