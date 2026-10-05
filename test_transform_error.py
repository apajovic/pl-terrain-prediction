"""
test_transform_error.py

Loads one image from data/v3/buildingMap, applies radial_unwrap (preprocess)
followed by radial_wrap (postprocess), then measures reconstruction error.
Sweeps over num_angles and num_radii to show how they affect quality.
"""

import sys
import os
import csv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src", "python"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from skimage.io import imread
from skimage.transform import resize

from preprocess import radial_unwrap
from postprocess import radial_wrap

# ── configuration ────────────────────────────────────────────────────────────

IMG_PATH = "data/v3/gainMap/614_76.png"
IMG_SIZE = (256, 256)

ANGLES_LIST = [16, 32, 64, 128, 196, 256, 384, 512, 768, 896, 1024, 2048]
RADII_LIST  = [16, 32, 64, 128, 196, 256, 384, 512, 768, 896, 1024, 2048]
METHODS = ["nearest", "linear"]

OUTPUT_DIR = "output/transform_error_interpolation_comparison"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── helpers ──────────────────────────────────────────────────────────────────

def circular_mask(img_size, center, radius):
    H, W = img_size
    yc, xc = center
    YY, XX = np.ogrid[:H, :W]
    return (XX - xc) ** 2 + (YY - yc) ** 2 <= radius ** 2


def compute_errors(original, reconstructed, mask):
    """Return MSE, MAE, RMSE, PSNR inside the circular mask."""
    o = original[mask].astype(float)
    r = reconstructed[mask].astype(float)
    mse  = np.mean((o - r) ** 2)
    mae  = np.mean(np.abs(o - r))
    rmse = np.sqrt(mse)
    # PSNR with max value 255
    psnr = 10 * np.log10(255 ** 2 / mse) if mse > 0 else float("inf")
    return dict(mse=mse, mae=mae, rmse=rmse, psnr=psnr)


# ── load image ───────────────────────────────────────────────────────────────

img = imread(IMG_PATH)
if img.ndim == 3:
    img = img[..., 0]           # take first channel if RGB
img = resize(img, IMG_SIZE, order=1, preserve_range=True).astype(np.uint8)

H, W = img.shape
center = (H / 2, W / 2)
max_r  = min(center[0], center[1], H - center[0], W - center[1])
mask   = circular_mask(IMG_SIZE, center, max_r)

print(f"Image: {IMG_PATH}  shape={img.shape}  dtype={img.dtype}")
print(f"Circular mask: center={center}  radius={max_r:.1f}  "
      f"pixels={mask.sum()}\n")

# ── sweep ────────────────────────────────────────────────────────────────────

results = {}            # (method, num_angles, num_radii) -> error dict

header = f"{'method':>8} {'num_angles':>12} {'num_radii':>10} {'MSE':>10} {'MAE':>10} {'RMSE':>10} {'PSNR(dB)':>10}"
print(header)
print("-" * len(header))

for method in METHODS:
    for na in ANGLES_LIST:
        for nr in RADII_LIST:
            # preprocess: wrap -> unwrap (polar rectangular form)
            unwrapped = radial_unwrap(img.astype(float), na, nr, center, method=method)

            # postprocess: unwrap -> wrap (back to Cartesian)
            reconstructed = radial_wrap(unwrapped, IMG_SIZE, center, method=method)

            # clip and cast to uint8 for fair comparison
            reconstructed = np.clip(reconstructed, 0, 255).astype(np.uint8)

            errs = compute_errors(img, reconstructed, mask)
            results[(method, na, nr)] = errs

            print(f"{method:>8} {na:>12} {nr:>10} {errs['mse']:>10.2f} "
                  f"{errs['mae']:>10.2f} {errs['rmse']:>10.2f} {errs['psnr']:>10.2f}")

with open(os.path.join(OUTPUT_DIR, "interpolation_comparison.csv"), "w", newline="") as csv_file:
    writer = csv.DictWriter(
        csv_file,
        fieldnames=["method", "num_angles", "num_radii", "mse", "mae", "rmse", "psnr"],
    )
    writer.writeheader()
    for method in METHODS:
        for na in ANGLES_LIST:
            for nr in RADII_LIST:
                writer.writerow({
                    "method": method,
                    "num_angles": na,
                    "num_radii": nr,
                    **results[(method, na, nr)],
                })

# ── visual comparison at default (256, 256) ──────────────────────────────────

na_def, nr_def = 896, 896
fig, axes = plt.subplots(len(METHODS), 4, figsize=(18, 9))
for row, method in enumerate(METHODS):
    unwrapped_def = radial_unwrap(img.astype(float), na_def, nr_def, center, method=method)
    recon_def = radial_wrap(unwrapped_def, IMG_SIZE, center, method=method)
    recon_def = np.clip(recon_def, 0, 255).astype(np.uint8)
    diff_def = np.abs(img.astype(int) - recon_def.astype(int)).astype(np.uint8)
    axes[row, 0].imshow(img, cmap="gray", vmin=0, vmax=255)
    axes[row, 0].set_title(f"Original ({method})")
    axes[row, 1].imshow(unwrapped_def, cmap="gray")
    axes[row, 1].set_title(f"Unwrapped {na_def}x{nr_def}")
    axes[row, 2].imshow(recon_def, cmap="gray", vmin=0, vmax=255)
    axes[row, 2].set_title("Reconstructed")
    axes[row, 3].imshow(diff_def, cmap="hot", vmin=0)
    axes[row, 3].set_title(
        f"Abs diff: RMSE={results[(method, na_def, nr_def)]['rmse']:.2f}\n"
        f"PSNR={results[(method, na_def, nr_def)]['psnr']:.2f} dB"
    )
for ax in axes.flat:
    ax.axis("off")
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "comparison_896x896.png"), dpi=120)
plt.close(fig)

# ── heatmaps of MSE and PSNR ─────────────────────────────────────────────────

metric_grids = {}
for method in METHODS:
    metric_grids[(method, "mse")] = np.array([
        [results[(method, na, nr)]["mse"] for nr in RADII_LIST]
        for na in ANGLES_LIST
    ])
    metric_grids[(method, "psnr")] = np.array([
        [results[(method, na, nr)]["psnr"] for nr in RADII_LIST]
        for na in ANGLES_LIST
    ])

metric_ranges = {
    metric: (
        min(metric_grids[(method, metric)].min() for method in METHODS),
        max(metric_grids[(method, metric)].max() for method in METHODS),
    )
    for metric in ("mse", "psnr")
}

fig, axes = plt.subplots(len(METHODS), 2, figsize=(14, 10))
for row, method in enumerate(METHODS):
    for col, (metric, cmap, title) in enumerate((
        ("mse", "Reds", "MSE (lower is better)"),
        ("psnr", "Greens", "PSNR dB (higher is better)"),
    )):
        grid = metric_grids[(method, metric)]
        vmin, vmax = metric_ranges[metric]
        ax = axes[row, col]
        image = ax.imshow(
            grid, cmap=cmap, aspect="auto", norm=LogNorm(vmin=vmin, vmax=vmax)
        )
        ax.set_xticks(range(len(RADII_LIST))); ax.set_xticklabels(RADII_LIST, rotation=45)
        ax.set_yticks(range(len(ANGLES_LIST))); ax.set_yticklabels(ANGLES_LIST)
        ax.set_xlabel("num_radii"); ax.set_ylabel("num_angles")
        ax.set_title(f"{method}: {title} - log color scale")
        plt.colorbar(image, ax=ax)
        for grid_row in range(grid.shape[0]):
            for grid_col in range(grid.shape[1]):
                value = grid[grid_row, grid_col]
                text_color = "white" if image.norm(value) > 0.55 else "black"
                ax.text(
                    grid_col, grid_row, f"{value:.1f}",
                    ha="center", va="center", fontsize=5.5, color=text_color,
                )

plt.suptitle(f"Reconstruction quality vs. num_angles / num_radii\n"
             f"Image: {os.path.basename(IMG_PATH)}", y=1.02)
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "error_heatmap.png"), dpi=120, bbox_inches="tight")
plt.close(fig)

# ── line plots: fix one axis, vary the other ──────────────────────────────────

fig, axes = plt.subplots(2, 2, figsize=(13, 9))

mid_nr = RADII_LIST[len(RADII_LIST) // 2]   # fix num_radii = middle value
mid_na = ANGLES_LIST[len(ANGLES_LIST) // 2]  # fix num_angles = middle value

# vary num_angles, fix num_radii
for method in METHODS:
    mse_vs_na = [results[(method, na, mid_nr)]["mse"] for na in ANGLES_LIST]
    psnr_vs_na = [results[(method, na, mid_nr)]["psnr"] for na in ANGLES_LIST]
    axes[0, 0].plot(ANGLES_LIST, mse_vs_na, "o-", label=method)
    axes[0, 1].plot(ANGLES_LIST, psnr_vs_na, "o-", label=method)
axes[0, 0].set_title(f"MSE vs num_angles  (num_radii={mid_nr})")
axes[0, 0].set_xlabel("num_angles"); axes[0, 0].set_ylabel("MSE")
axes[0, 0].grid(True); axes[0, 0].legend()
axes[0, 1].set_title(f"PSNR vs num_angles  (num_radii={mid_nr})")
axes[0, 1].set_xlabel("num_angles"); axes[0, 1].set_ylabel("PSNR (dB)")
axes[0, 1].grid(True); axes[0, 1].legend()

# vary num_radii, fix num_angles
for method in METHODS:
    mse_vs_nr = [results[(method, mid_na, nr)]["mse"] for nr in RADII_LIST]
    psnr_vs_nr = [results[(method, mid_na, nr)]["psnr"] for nr in RADII_LIST]
    axes[1, 0].plot(RADII_LIST, mse_vs_nr, "o-", label=method)
    axes[1, 1].plot(RADII_LIST, psnr_vs_nr, "o-", label=method)
axes[1, 0].set_title(f"MSE vs num_radii  (num_angles={mid_na})")
axes[1, 0].set_xlabel("num_radii"); axes[1, 0].set_ylabel("MSE")
axes[1, 0].grid(True); axes[1, 0].legend()
axes[1, 1].set_title(f"PSNR vs num_radii  (num_angles={mid_na})")
axes[1, 1].set_xlabel("num_radii"); axes[1, 1].set_ylabel("PSNR (dB)")
axes[1, 1].grid(True); axes[1, 1].legend()

plt.suptitle(f"Effect of sampling resolution on reconstruction quality\n"
             f"Image: {os.path.basename(IMG_PATH)}")
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "error_line_plots.png"), dpi=120)
plt.close(fig)

print(f"\nPlots saved to: {OUTPUT_DIR}/")
