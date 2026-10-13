"""Render the triangular grid in each saved attention-map file as PNG images.

Usage:
    python visualize_attention_maps.py ATTENTION_FOLDER OUTPUT_FOLDER
"""

import argparse
from pathlib import Path

import imageio.v3 as iio
import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt


BLOCK_SIZE = 26


def normalize_to_uint8(image: torch.Tensor) -> torch.Tensor:
    """Scale a nonnegative image so its maximum becomes 255."""
    maximum = image.max()
    if maximum <= 0:
        return torch.zeros_like(image, dtype=torch.uint8)
    return (image * (255.0 / maximum)).round().clamp(0, 255).to(torch.uint8)


def load_grid(path: Path) -> tuple[list[list[torch.Tensor]], dict]:
    data = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(data, dict) or "maps" not in data:
        raise ValueError(f"{path}: expected a dictionary with a 'maps' entry")
    maps = data["maps"]
    if not isinstance(maps, list) or not maps or not isinstance(maps[0], list) or not maps[0]:
        raise ValueError(f"{path}: maps must be a nonempty [chunk][step] grid")

    steps = len(maps[0])
    for chunk_index, chunk in enumerate(maps):
        if not isinstance(chunk, list) or len(chunk) != steps:
            raise ValueError(f"{path}: all chunks must have the same number of steps")
        for step_index, cell in enumerate(chunk):
            expected_shape = (BLOCK_SIZE, BLOCK_SIZE * (chunk_index + 1))
            if not isinstance(cell, torch.Tensor) or tuple(cell.shape) != expected_shape:
                raise ValueError(
                    f"{path}: maps[{chunk_index}][{step_index}] must have shape "
                    f"{expected_shape}; got {getattr(cell, 'shape', None)}"
                )
            if not torch.isfinite(cell).all() or (cell < 0).any():
                raise ValueError(f"{path}: maps[{chunk_index}][{step_index}] has invalid values")
    return maps, data


def step_image(maps: list[list[torch.Tensor]], step_index: int) -> torch.Tensor:
    """Place each query chunk's visible key blocks in the square image."""
    side = BLOCK_SIZE * len(maps)
    grid = torch.zeros((side, side), dtype=torch.float32)
    for chunk_index, chunk in enumerate(maps):
        width = BLOCK_SIZE * (chunk_index + 1)
        grid[BLOCK_SIZE * chunk_index:BLOCK_SIZE * (chunk_index + 1), :width] = (
            chunk[step_index].float()
        )
    return normalize_to_uint8(grid)


def render_file(path: Path, output_folder: Path) -> list[Path]:
    maps, _ = load_grid(path)
    num_steps = len(maps[0])
    images = []
    paths = []

    for step_index in range(num_steps):
        pixels = step_image(maps, step_index)
        output_path = output_folder / f"{path.stem}_step_{step_index + 1:03d}.png"
        iio.imwrite(output_path, pixels.numpy())
        images.append(pixels)
        paths.append(output_path)

    # Average the displayed step images, then rescale the average so its
    # brightest pixel is also 255. The empty upper-right triangle stays zero.
    average = torch.stack(images).float().mean(dim=0)
    average_path = output_folder / f"{path.stem}_average.png"
    iio.imwrite(average_path, normalize_to_uint8(average*5).numpy())
    paths.append(average_path)
    return paths


def block_weights(pixels: np.ndarray) -> np.ndarray:
    """Average each 26x26 block and normalize each query-chunk row."""
    if pixels.ndim != 2 or pixels.shape[0] != pixels.shape[1] or pixels.shape[0] % BLOCK_SIZE:
        raise ValueError("Expected a square grayscale PNG with sides divisible by 26")
    chunks = pixels.shape[0] // BLOCK_SIZE
    means = pixels.astype(np.float64).reshape(
        chunks, BLOCK_SIZE, chunks, BLOCK_SIZE
    ).mean(axis=(1, 3))
    return normalize_weight_rows(means)


def normalize_weight_rows(weights: np.ndarray) -> np.ndarray:
    row_sums = weights.sum(axis=1, keepdims=True)
    # An entirely black row has no attention mass and remains zero.
    return np.divide(weights, row_sums, out=np.zeros_like(weights), where=row_sums > 0)


def render_weight_grid(weights: np.ndarray, output_path: Path) -> Path:
    """Save an N x N, row-normalized chunk-weight heatmap."""
    chunks = weights.shape[0]
    figure, axes = plt.subplots(figsize=(max(4, 0.65 * chunks + 2), max(4, 0.65 * chunks + 2)))
    maximum = float(weights.max())
    plot = axes.imshow(weights, cmap="coolwarm", vmin=0, vmax=maximum if maximum > 0 else 1)
    axes.set_xticks(np.arange(chunks))
    axes.set_xticklabels(np.arange(1, chunks + 1))
    axes.set_yticks(np.arange(chunks))
    axes.set_yticklabels(np.arange(1, chunks + 1))
    axes.set_xlabel("Key chunk")
    axes.set_ylabel("Query chunk")
    if chunks <= 12:
        for row in range(chunks):
            for column in range(chunks):
                axes.text(column, row, f"{weights[row, column]:.2f}",
                          ha="center", va="center", fontsize=8)
    figure.colorbar(plot, ax=axes, label="Row-normalized block weight")
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    return output_path


def render_weight_image(path: Path) -> Path:
    """Save a cold-to-warm heatmap beside one grayscale attention PNG."""
    weights = block_weights(iio.imread(path))
    return render_weight_grid(weights, path.with_name(f"{path.stem}_weight.png"))


def render_per_head_file(path: Path, output_folder: Path) -> tuple[int, list[np.ndarray]]:
    """Render a per-file chunk-weight average over all timesteps."""
    maps, data = load_grid(path)
    head_index = data.get("head_index")
    if isinstance(head_index, bool) or not isinstance(head_index, int) or head_index < 0:
        raise ValueError(f"{path}: expected a nonnegative integer 'head_index'")

    step_weights = []
    for step_index in range(len(maps[0])):
        pixels = step_image(maps, step_index)
        weights = block_weights(pixels.numpy())
        step_weights.append(weights)

    # Every timestep contributes equally to this file's N x N weight image.
    average_weights = normalize_weight_rows(np.stack(step_weights).mean(axis=0))
    output_path = output_folder / f"{path.stem}_average_weight.png"
    print(render_weight_grid(average_weights, output_path))
    return head_index, step_weights


def render_per_head_folder(attention_folder: Path, output_folder: Path) -> None:
    source = attention_folder / "per_head"
    if not source.is_dir():
        return
    destination = output_folder / "per_head"
    destination.mkdir(parents=True, exist_ok=True)
    files = sorted(
        path for path in source.iterdir()
        if path.is_file() and path.suffix.lower() in {".pth", ".pt"}
    )
    by_head: dict[int, list[np.ndarray]] = {}
    for path in files:
        head_index, step_weights = render_per_head_file(path, destination)
        by_head.setdefault(head_index, []).extend(step_weights)

    for head_index, grids in sorted(by_head.items()):
        if any(grid.shape != grids[0].shape for grid in grids[1:]):
            raise ValueError(f"Head {head_index}: prompt maps have different chunk counts")
        # Average every timestep from every prompt for this head.
        average_weights = normalize_weight_rows(np.stack(grids).mean(axis=0))
        output_path = destination / f"head_{head_index:02d}_all_prompts_weight.png"
        print(render_weight_grid(average_weights, output_path))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("attention_folder", type=Path, help="Folder containing .pth or .pt attention maps")
    parser.add_argument("output_folder", type=Path, help="Folder for PNG images")
    args = parser.parse_args(argv)

    if not args.attention_folder.is_dir():
        parser.error(f"Not a folder: {args.attention_folder}")
    files = sorted(
        path for path in args.attention_folder.iterdir()
        if path.is_file() and path.suffix.lower() in {".pth", ".pt"}
    )
    args.output_folder.mkdir(parents=True, exist_ok=True)
    overall_sum = None
    step_sums = None
    generated_pngs = []
    for path in files:
        output_paths = render_file(path, args.output_folder)
        for output_path in output_paths:
            print(output_path)
        generated_pngs.extend(output_paths)

        # Give each source file equal weight by averaging its saved
        # step-average PNG. All source files must have the same chunk count.
        file_average = torch.from_numpy(iio.imread(output_paths[-1])).float()
        if overall_sum is None:
            overall_sum = torch.zeros_like(file_average)
        elif file_average.shape != overall_sum.shape:
            raise ValueError(
                f"{path}: average image shape {tuple(file_average.shape)} differs "
                f"from the first file's {tuple(overall_sum.shape)}"
            )
        overall_sum += file_average

        # Average the displayed map at each denoising step across prompts.
        # Require all files to provide the same steps so every prompt has
        # equal weight in every step image.
        step_paths = output_paths[:-1]
        if step_sums is None:
            step_sums = [torch.zeros_like(file_average) for _ in step_paths]
        elif len(step_paths) != len(step_sums):
            raise ValueError(
                f"{path}: has {len(step_paths)} steps; expected {len(step_sums)}"
            )
        for step_index, step_path in enumerate(step_paths):
            step_pixels = torch.from_numpy(iio.imread(step_path)).float()
            if step_pixels.shape != step_sums[step_index].shape:
                raise ValueError(f"{path}: step {step_index} image size differs from the first file")
            step_sums[step_index] += step_pixels

    if files:
        for step_index, step_sum in enumerate(step_sums):
            step_path = args.output_folder / f"all_files_step_{step_index:03d}.png"
            iio.imwrite(step_path, normalize_to_uint8(step_sum / len(files)).numpy())
            print(step_path)
            generated_pngs.append(step_path)

        overall_path = args.output_folder / "all_files_average.png"
        overall_pixels = normalize_to_uint8(overall_sum / len(files))
        iio.imwrite(overall_path, overall_pixels.numpy())
        print(overall_path)
        generated_pngs.append(overall_path)
        for png_path in generated_pngs:
            print(render_weight_image(png_path))
    else:
        print(f"No .pth or .pt files found in {args.attention_folder}")

    render_per_head_folder(args.attention_folder, args.output_folder)


if __name__ == "__main__":
    main()
