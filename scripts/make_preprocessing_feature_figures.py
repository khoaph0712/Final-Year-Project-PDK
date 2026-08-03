#!/usr/bin/env python
"""Build the image-preprocessing and feature-extraction figures for the FYP site.

Lecturer feedback asked for two things the site was missing: an image-preprocessing
chapter that shows *images* rather than only charts, split explicitly into spatial
and frequency domain steps, and histograms for the feature-extraction stage.

Everything here runs the project's real code paths on real crops from
data/hard_case_classifier_v1_clean, so nothing on the page is an illustration:

  scripts/custom_feature_extractor.py ...... deployed 637-D vector (server.py imports this)
  scripts/feature_ml_analysis.py ........... analysis 637-D vector (spatial + FFT + color + HOG)
  scripts/archive/ml_balanced_training.py .. augment_crop, the real training augmentation
  web/server.py ............................ CROP_PAD_PX=10, 24x24 reject, 224x224 model input

Outputs land in web/assets/figures/.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
# archive/ goes last: it holds a same-named shim for feature_ml_analysis that must not shadow.
sys.path.append(str(ROOT / "scripts" / "archive"))

from custom_feature_extractor import (  # noqa: E402
    compute_glcm_features,
    compute_lbp,
    compute_shape_features,
)
from feature_ml_analysis import (  # noqa: E402
    extract_frequency_features,
    extract_spatial_features,
)
from ml_balanced_training import augment_crop  # noqa: E402

WEB = ROOT / "web" / "assets" / "figures"
# detector_crops_v1 is the production distribution: crops cut from the deployed detector's
# own boxes, so they keep the real aspect ratios and the real small-object tail. Using the
# studio-heavy classifier benchmark instead would make every figure look easier than reality.
CROPS = Path("C:/FYP/data/detector_crops_v1/val")
FREQ_ANALYSIS = Path("C:/FYP/data/frequency_analysis")

# Dataset folder names -> display names. Background is the veto class, not a material.
CLASS_DIRS = ["plastic", "glass", "metal", "paper", "cardboard", "organic", "Background"]
MATERIALS = ["plastic", "glass", "metal", "paper", "cardboard", "organic"]
NICE = {c: c.capitalize() for c in CLASS_DIRS}
NICE["Background"] = "Background"

# Site accent ramp (--accent #047857 / --accent-bright #34d399) plus a warm contrast.
GREEN = "#047857"
BRIGHT = "#34d399"
INK = "#1b1b18"
WARN = "#b45309"
CLASS_COLORS = {
    "plastic": "#047857",
    "glass": "#0891b2",
    "metal": "#6b7280",
    "paper": "#b45309",
    "cardboard": "#7c3aed",
    "organic": "#65a30d",
    "Background": "#9ca3af",
}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.titlesize": 10.5,
    "axes.titleweight": "bold",
    "axes.labelsize": 9,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
})


def save(fig, name, dpi=165):
    """Photo grids ship at a lower dpi than charts - upscaled 64x64 crops gain
    nothing from 165 dpi and the page has to download every figure."""
    WEB.mkdir(parents=True, exist_ok=True)
    out = WEB / name
    fig.savefig(out, dpi=dpi, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
    print(f"[fig] {name}  ({out.stat().st_size / 1024:.0f} KB)")


def crop_paths(class_dir: str, n: int, seed: int = 7) -> list[Path]:
    """Deterministic sample of crop paths so figures are reproducible."""
    files = sorted((CROPS / class_dir).glob("*.jpg")) + sorted((CROPS / class_dir).glob("*.png"))
    if not files:
        raise SystemExit(f"no crops found under {CROPS / class_dir}")
    rng = random.Random(seed)
    return rng.sample(files, min(n, len(files)))


def load_bgr(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"cannot read {path}")
    return img


def show_bgr(ax, bgr, title):
    ax.imshow(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    ax.set_title(title, fontsize=8.5, pad=4)
    ax.set_xticks([])
    ax.set_yticks([])


def show_gray(ax, gray, title, cmap="gray"):
    ax.imshow(gray, cmap=cmap)
    ax.set_title(title, fontsize=8.5, pad=4)
    ax.set_xticks([])
    ax.set_yticks([])


def bar_axes(ax, ylabel=""):
    ax.set_ylabel(ylabel)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#e5e7eb", linewidth=0.8)
    ax.set_axisbelow(True)


# --------------------------------------------------------- 1. spatial-domain steps

def fig_spatial_domain():
    """The spatial-domain half of preprocessing, run on three real crops."""
    picks = [("plastic", 0), ("metal", 0), ("cardboard", 0)]
    rows = [(cls, load_bgr(crop_paths(cls, 3, seed=11)[i])) for cls, i in picks]

    titles = [
        "1. Detector crop\n+10 px pad, native aspect",
        "2. Model input\n224x224, INTER_CUBIC",
        "3. Feature input\n64x64, INTER_AREA",
        "4. Grayscale\ncv2.BGR2GRAY",
        "5. Sobel gradient\nmagnitude (ksize 3)",
        "6. Otsu mask\nshape descriptors",
    ]

    fig, axes = plt.subplots(len(rows), 6, figsize=(13.2, 2.35 * len(rows)))
    for r, (cls, bgr) in enumerate(rows):
        model_in = cv2.resize(bgr, (224, 224), interpolation=cv2.INTER_CUBIC)
        feat_in = cv2.resize(bgr, (64, 64), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(feat_in, cv2.COLOR_BGR2GRAY)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        grad = np.sqrt(gx * gx + gy * gy)
        _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        h, w = bgr.shape[:2]
        show_bgr(axes[r, 0], bgr, titles[0] if r == 0 else "")
        axes[r, 0].set_ylabel(f"{NICE[cls]}\n{w}x{h} px", fontsize=8.5, fontweight="bold")
        show_bgr(axes[r, 1], model_in, titles[1] if r == 0 else "")
        show_bgr(axes[r, 2], feat_in, titles[2] if r == 0 else "")
        show_gray(axes[r, 3], gray, titles[3] if r == 0 else "")
        show_gray(axes[r, 4], grad, titles[4] if r == 0 else "", cmap="magma")
        show_gray(axes[r, 5], otsu, titles[5] if r == 0 else "")

    fig.suptitle(
        "Spatial-domain preprocessing: every step below runs on the pixel grid itself",
        fontsize=11.5, fontweight="bold", y=1.0,
    )
    fig.tight_layout()
    save(fig, "prep_spatial_domain.png", dpi=110)


# ------------------------------------------------------------ 2. augmentation

def fig_augmentation():
    """The four real augment_crop operations, applied one at a time and combined."""
    bgr = load_bgr(crop_paths("glass", 3, seed=23)[0])
    h, w = bgr.shape[:2]

    def rotate(img, angle):
        m = cv2.getRotationMatrix2D((img.shape[1] // 2, img.shape[0] // 2), angle, 1.0)
        return cv2.warpAffine(img, m, (img.shape[1], img.shape[0]), borderMode=cv2.BORDER_REPLICATE)

    panels = [
        (bgr, "Original crop"),
        (cv2.flip(bgr, 1), "Horizontal flip\np = 0.5"),
        (cv2.flip(bgr, 0), "Vertical flip\np = 0.5"),
        (rotate(bgr, 12.0), "Rotate +12deg\nBORDER_REPLICATE"),
        (np.clip(bgr * 0.85, 0, 255).astype(np.uint8), "Brightness x0.85\nlow end of range"),
        (np.clip(bgr * 1.15, 0, 255).astype(np.uint8), "Brightness x1.15\nhigh end of range"),
    ]
    # Two sampled compositions, from the real function with fixed seeds.
    combos = [augment_crop(bgr, random.Random(s)) for s in (4, 17)]

    fig, axes = plt.subplots(2, 4, figsize=(9.6, 5.4))
    flat = axes.flatten()
    for ax, (img, title) in zip(flat[:6], panels):
        show_bgr(ax, img, title)
    for ax, img, i in zip(flat[6:], combos, (1, 2)):
        show_bgr(ax, img, f"Sampled composition {i}\naugment_crop(seed)")
        for side in ax.spines.values():
            side.set_edgecolor(GREEN)
            side.set_linewidth(2.0)

    fig.suptitle(
        f"Training-time augmentation on a real {w}x{h} px glass crop\n"
        "minority classes are oversampled through these four operations, never duplicated",
        fontsize=11, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    save(fig, "prep_augmentation.png", dpi=110)


# ----------------------------------------------------------- 3. min-size filter

def fig_size_filter():
    """The 24x24 px reject rule, shown on real crops sorted by area."""
    pool: list[tuple[Path, int, int]] = []
    for cls in MATERIALS:
        for p in crop_paths(cls, 60, seed=5):
            img = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if img is None:
                continue
            pool.append((p, img.shape[1], img.shape[0]))
    pool.sort(key=lambda t: t[1] * t[2])

    # The four smallest crops in the sample next to four typical ones: the gate is
    # min(w, h) < 24 px, so whether the small end actually trips it is a real result.
    picks = pool[:4] + pool[len(pool) // 2 + 6:][:4]
    n_rejected = sum(1 for _, w, h in picks if min(w, h) < 24)

    fig, axes = plt.subplots(1, len(picks), figsize=(2.0 * len(picks), 2.9))
    for ax, (path, w, h) in zip(np.atleast_1d(axes), picks):
        img = load_bgr(path)
        rejected = min(w, h) < 24
        # Upscale for display only, nearest-neighbour so the real pixel grid stays visible.
        disp = cv2.resize(img, (160, 160), interpolation=cv2.INTER_NEAREST)
        show_bgr(ax, disp, f"{w}x{h} px\n{'REJECTED' if rejected else 'classified'}")
        ax.title.set_color(WARN if rejected else GREEN)
        for side in ax.spines.values():
            side.set_edgecolor(WARN if rejected else GREEN)
            side.set_linewidth(2.2)

    fig.suptitle(
        "The 24x24 px gate: the four smallest crops in the sample, then four typical ones\n"
        f"displayed at nearest-neighbour upscale so the real pixel grid stays visible - "
        f"{n_rejected} of these 8 trip the gate",
        fontsize=10.5, fontweight="bold", y=1.04,
    )
    fig.tight_layout()
    save(fig, "prep_size_filter.png", dpi=110)


# ------------------------------------------------------ 4. frequency-domain steps

def fig_frequency_domain():
    """The frequency-domain half of preprocessing: 2D FFT, then band reconstructions."""
    picks = [("organic", 0), ("paper", 0), ("metal", 1)]
    rows = [(cls, load_bgr(crop_paths(cls, 3, seed=31)[i])) for cls, i in picks]

    titles = [
        "1. Grayscale 64x64\nmean-centred",
        "2. log |FFT2|\nfftshift, DC at centre",
        "3. Low-pass\ninner 25% radius",
        "4. High-pass\nouter 50% radius",
        "5. Radial band energy\n8 bins -> fft_bin_1..8",
    ]

    fig, axes = plt.subplots(len(rows), 5, figsize=(13.0, 2.5 * len(rows)))
    for r, (cls, bgr) in enumerate(rows):
        gray = cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), (64, 64),
                          interpolation=cv2.INTER_AREA).astype(np.float32)
        centred = gray - gray.mean()
        fft = np.fft.fftshift(np.fft.fft2(centred))
        spectrum = np.log1p(np.abs(fft))

        h, w = centred.shape
        yy, xx = np.indices((h, w))
        radius = np.sqrt((yy - h // 2) ** 2 + (xx - w // 2) ** 2)
        rmax = radius.max()

        def reconstruct(mask):
            return np.real(np.fft.ifft2(np.fft.ifftshift(fft * mask)))

        low = reconstruct(radius <= 0.25 * rmax)
        high = reconstruct(radius > 0.50 * rmax)

        freq = extract_frequency_features(bgr)

        show_gray(axes[r, 0], gray, titles[0] if r == 0 else "")
        axes[r, 0].set_ylabel(NICE[cls], fontsize=9, fontweight="bold")
        show_gray(axes[r, 1], spectrum, titles[1] if r == 0 else "", cmap="magma")
        show_gray(axes[r, 2], low, titles[2] if r == 0 else "")
        show_gray(axes[r, 3], high, titles[3] if r == 0 else "")

        ax = axes[r, 4]
        ax.bar(range(1, 9), freq[:8], color=CLASS_COLORS[cls], width=0.72)
        bar_axes(ax, "share of power")
        ax.set_yscale("log")
        ax.set_xticks(range(1, 9))
        ax.set_xlabel("radial band", fontsize=8)
        ax.tick_params(labelsize=7.5)
        if r == 0:
            ax.set_title(titles[4], fontsize=8.5, pad=4)
        ax.text(0.97, 0.9, f"high-freq {freq[8]:.3f}", transform=ax.transAxes,
                ha="right", fontsize=7.5, color=INK)

    fig.suptitle(
        "Frequency-domain preprocessing: the same crop read as spatial frequencies\n"
        "columns 3 and 4 are inverse transforms of one band only, so you can see what each band carries",
        fontsize=11.5, fontweight="bold", y=1.01,
    )
    fig.tight_layout()
    save(fig, "prep_frequency_domain.png", dpi=125)


# -------------------------------------------------- 5. colour-histogram features

def fig_color_histograms():
    """The 256 colour features are literally histograms - so plot them as histograms."""
    classes = ["plastic", "metal", "cardboard", "organic"]
    fig, axes = plt.subplots(len(classes), 3, figsize=(12.6, 2.4 * len(classes)),
                             gridspec_kw={"width_ratios": [0.55, 1.35, 1.35]})

    for r, cls in enumerate(classes):
        bgr = load_bgr(crop_paths(cls, 3, seed=41)[0])
        resized = cv2.resize(bgr, (64, 64), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)

        show_bgr(axes[r, 0], resized, "crop, 64x64" if r == 0 else "")
        axes[r, 0].set_ylabel(NICE[cls], fontsize=9, fontweight="bold")

        ax = axes[r, 1]
        for chan, colour, label in ((2, "#dc2626", "R"), (1, "#16a34a", "G"), (0, "#2563eb", "B")):
            h = cv2.calcHist([resized], [chan], None, [48], [0, 256]).flatten()
            h = h / (h.sum() + 1e-9)
            ax.plot(h, color=colour, linewidth=1.4, label=label)
            ax.fill_between(range(48), h, color=colour, alpha=0.12)
        bar_axes(ax, "normalized count")
        ax.set_xlim(0, 47)
        ax.tick_params(labelsize=7.5)
        ax.legend(fontsize=7.5, frameon=False, ncols=3, loc="upper right")
        if r == 0:
            ax.set_title("RGB histogram - 3 channels x 48 bins = 144 features", fontsize=9)
        if r == len(classes) - 1:
            ax.set_xlabel("intensity bin")

        ax = axes[r, 2]
        offset = 0
        for chan, bins, rng, colour, label in (
            (0, 48, 180, "#7c3aed", "Hue (48)"),
            (1, 32, 256, "#0891b2", "Sat (32)"),
            (2, 32, 256, "#ca8a04", "Val (32)"),
        ):
            h = cv2.calcHist([hsv], [chan], None, [bins], [0, rng]).flatten()
            h = h / (h.sum() + 1e-9)
            ax.bar(np.arange(bins) + offset, h, width=1.0, color=colour, label=label)
            offset += bins
            if offset < 112:
                ax.axvline(offset - 0.5, color="#9ca3af", linewidth=0.8, linestyle=":")
        bar_axes(ax, "")
        ax.set_xlim(-0.5, 111.5)
        ax.tick_params(labelsize=7.5)
        ax.legend(fontsize=7.5, frameon=False, ncols=3, loc="upper right")
        if r == 0:
            ax.set_title("HSV histogram - 48 + 32 + 32 bins = 112 features", fontsize=9)
        if r == len(classes) - 1:
            ax.set_xlabel("concatenated HSV bin index")

    fig.suptitle(
        "Colour block of the deployed 637-D vector: 144 RGB + 112 HSV = 256 features\n"
        "these are not summary statistics - the histogram bins are the features",
        fontsize=11.5, fontweight="bold", y=1.005,
    )
    fig.tight_layout()
    save(fig, "feat_color_histograms.png")


# ------------------------------------------------- 6. texture-histogram features

def fig_texture_histograms():
    """LBP histogram + GLCM descriptors: the 47 texture features, per class."""
    fig, axes = plt.subplots(1, 3, figsize=(15.4, 4.1),
                             gridspec_kw={"width_ratios": [1.05, 1.15, 1.15], "wspace": 0.34})

    lbp_by_class, glcm_by_class = {}, {}
    for cls in MATERIALS:
        lbps, glcms = [], []
        for p in crop_paths(cls, 60, seed=53):
            gray = cv2.cvtColor(cv2.resize(load_bgr(p), (64, 64), interpolation=cv2.INTER_AREA),
                                cv2.COLOR_BGR2GRAY)
            lbps.append(compute_lbp(gray))
            glcms.append(compute_glcm_features(gray))
        lbp_by_class[cls] = np.mean(lbps, axis=0)
        glcm_by_class[cls] = np.mean(glcms, axis=0)

    ax = axes[0]
    width = 0.14
    for i, cls in enumerate(MATERIALS):
        ax.bar(np.arange(10) + (i - 2.5) * width, lbp_by_class[cls], width=width,
               color=CLASS_COLORS[cls], label=NICE[cls])
    bar_axes(ax, "mean normalized count (log)")
    # Log scale because bin 9 absorbs all 198 non-uniform patterns (see compute_lbp's
    # docstring) and swamps bins 0-8 on a linear axis.
    ax.set_yscale("log")
    ax.set_xticks(range(10))
    ax.set_xlabel("LBP bin (0-8 uniform patterns, 9 = catch-all)")
    ax.set_title("LBP histogram - 10 features\nmean over 60 crops per class", fontsize=9.5)
    ax.set_ylim(top=ax.get_ylim()[1] * 4)  # headroom so the legend clears the bars
    ax.legend(fontsize=7, frameon=False, ncols=3, loc="upper center")

    # 9 descriptors averaged across the 4 GLCM directions, for readability.
    names = ["contrast", "homogen.", "energy", "dissim.", "max prob", "entropy", "mean i", "std i", "corr"]
    ax = axes[1]
    grid = np.array([[np.mean([glcm_by_class[cls][d * 9 + k] for d in range(4)])
                      for k in range(9)] for cls in MATERIALS])
    norm = (grid - grid.min(axis=0)) / (np.ptp(grid, axis=0) + 1e-9)
    ax.imshow(norm, cmap="YlGn", vmin=0, vmax=1, aspect="auto")
    for i in range(len(MATERIALS)):
        for j in range(9):
            ax.text(j, i, f"{grid[i, j]:.2f}", ha="center", va="center", fontsize=6.6,
                    color=INK if norm[i, j] < 0.6 else "#ffffff")
    ax.set_xticks(range(9), names, rotation=40, ha="right", fontsize=7.5)
    ax.set_yticks(range(len(MATERIALS)), [NICE[c] for c in MATERIALS], fontsize=8)
    ax.set_title("GLCM - 37 features\n9 descriptors x 4 directions + avg entropy", fontsize=9.5)
    for side in ax.spines.values():
        side.set_visible(False)

    ax = axes[2]
    bgr = load_bgr(crop_paths("metal", 3, seed=53)[0])
    gray = cv2.cvtColor(cv2.resize(bgr, (64, 64), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    shape = compute_shape_features(gray)
    labels = [f"Hu {i + 1}" for i in range(7)] + ["area", "perimeter", "circularity"]
    colours = [GREEN] * 7 + [BRIGHT] * 3
    ax.barh(range(10), shape, color=colours)
    ax.set_yticks(range(10), labels, fontsize=8)
    ax.invert_yaxis()
    ax.axvline(0, color="#9ca3af", linewidth=0.8)
    bar_axes(ax, "")
    ax.set_xlabel("value (Hu moments are -sign*log10|h|)")
    ax.set_title("Shape - 10 features\n7 Hu moments + area, perimeter, circularity", fontsize=9.5)

    fig.suptitle(
        "Texture and shape blocks of the deployed 637-D vector: 47 texture + 10 shape features",
        fontsize=11.5, fontweight="bold", y=1.03,
    )
    fig.tight_layout(w_pad=3.0)
    save(fig, "feat_texture_histograms.png")


# ------------------------------------------------------------ 7. HOG histogram

def fig_hog_histogram():
    """HOG is a histogram of oriented gradients - show the orientation bins directly."""
    bgr = load_bgr(crop_paths("cardboard", 3, seed=67)[0])
    gray = cv2.cvtColor(cv2.resize(bgr, (64, 64), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)

    hog = cv2.HOGDescriptor(_winSize=(64, 64), _blockSize=(32, 32), _blockStride=(16, 16),
                            _cellSize=(16, 16), _nbins=9)
    desc = hog.compute(gray).flatten()

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=1)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=1)
    mag = np.sqrt(gx * gx + gy * gy)
    ang = (np.rad2deg(np.arctan2(gy, gx)) % 180.0)
    orient_hist, edges = np.histogram(ang, bins=9, range=(0, 180), weights=mag)
    orient_hist = orient_hist / (orient_hist.sum() + 1e-9)

    fig, axes = plt.subplots(1, 4, figsize=(14.0, 3.4),
                             gridspec_kw={"width_ratios": [0.62, 0.62, 1.0, 1.5]})

    show_gray(axes[0], gray, "crop, 64x64 grayscale")
    show_gray(axes[1], mag, "gradient magnitude", cmap="magma")

    ax = axes[2]
    centres = (edges[:-1] + edges[1:]) / 2
    ax.bar(centres, orient_hist, width=17, color=GREEN)
    bar_axes(ax, "magnitude-weighted share")
    ax.set_xticks(centres.astype(int))
    ax.tick_params(labelsize=7.5)
    ax.set_xlabel("gradient orientation (degrees, unsigned)")
    ax.set_title("The 9 orientation bins\nHOG's underlying histogram", fontsize=9.5)

    ax = axes[3]
    ax.bar(range(len(desc)), desc, width=1.0, color=GREEN)
    for b in range(1, 9):
        ax.axvline(b * 36 - 0.5, color="#9ca3af", linewidth=0.7, linestyle=":")
    bar_axes(ax, "L2-Hys normalized")
    ax.set_xlim(-1, len(desc))
    ax.tick_params(labelsize=7.5)
    ax.set_xlabel("descriptor index (9 blocks x 4 cells x 9 bins = 324)")
    ax.set_title(f"Full HOG descriptor - {len(desc)} features\n"
                 "dotted lines separate the 9 blocks", fontsize=9.5)

    fig.suptitle(
        "Edge block of the deployed 637-D vector: HOG, 324 features\n"
        "64x64 window, 32x32 block, 16x16 stride, 16x16 cell, 9 bins",
        fontsize=11.5, fontweight="bold", y=1.06,
    )
    fig.tight_layout()
    save(fig, "feat_hog_histogram.png")


# ----------------------------------------- 8. per-class scalar feature histograms

def fig_scalar_distributions():
    """Real distributions of the interpretable scalar features, one histogram per feature."""
    per_class: dict[str, np.ndarray] = {}
    for cls in MATERIALS:
        rows = []
        for p in crop_paths(cls, 120, seed=71):
            bgr = load_bgr(p)
            rows.append(np.concatenate([extract_spatial_features(bgr),
                                        extract_frequency_features(bgr)]))
        per_class[cls] = np.array(rows)

    # 8 spatial + high_freq_energy: the features that have a readable physical meaning.
    picks = [
        (0, "mean_intensity", "spatial"),
        (1, "std_intensity", "spatial"),
        (5, "grad_mean", "spatial"),
        (6, "grad_std", "spatial"),
        (7, "edge_density", "spatial"),
        (16, "high_freq_energy", "frequency"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(13.2, 6.4))
    for ax, (idx, name, domain) in zip(axes.flatten(), picks):
        allv = np.concatenate([per_class[c][:, idx] for c in MATERIALS])
        # Clip the top 1%: high_freq_energy has a long tail that would squash every class
        # into the first bin and hide the between-class differences the figure is about.
        bins = np.linspace(allv.min(), float(np.percentile(allv, 99)), 26)
        for cls in MATERIALS:
            ax.hist(per_class[cls][:, idx], bins=bins, histtype="step", linewidth=1.6,
                    color=CLASS_COLORS[cls], label=NICE[cls])
        bar_axes(ax, "crops")
        ax.set_title(f"{name}  ({domain} domain)", fontsize=9.5)
        ax.tick_params(labelsize=7.5)
    axes[0, 0].legend(fontsize=7.5, frameon=False, ncols=2)

    fig.suptitle(
        "Per-class distribution of the interpretable scalar features, 120 real crops per class\n"
        "overlap here is exactly why the classical branch tops out around 74% - "
        "no single descriptor separates the materials",
        fontsize=11.5, fontweight="bold", y=1.0,
    )
    fig.tight_layout()
    save(fig, "feat_scalar_distributions.png")


# ------------------------------------------- 9. per-class radial frequency profile

def fig_frequency_bins_by_class():
    """fft_bin_1..8 per class, measured fresh on real crops."""
    means, stds = {}, {}
    for cls in MATERIALS:
        rows = [extract_frequency_features(load_bgr(p)) for p in crop_paths(cls, 120, seed=83)]
        arr = np.array(rows)
        means[cls] = arr.mean(axis=0)
        stds[cls] = arr.std(axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.2), gridspec_kw={"width_ratios": [1.35, 1.0]})

    ax = axes[0]
    for cls in MATERIALS:
        ax.errorbar(range(1, 9), means[cls][:8], yerr=stds[cls][:8], marker="o", markersize=4,
                    linewidth=1.6, capsize=2.5, color=CLASS_COLORS[cls], label=NICE[cls])
    ax.set_yscale("log")
    bar_axes(ax, "share of total power (log)")
    ax.set_xticks(range(1, 9))
    ax.set_xlabel("radial band (1 = lowest frequency, 8 = highest)")
    ax.set_title("Radial band energy per class - features fft_bin_1..8\n"
                 "mean +/- 1 sd over 120 crops per class", fontsize=9.5)
    ax.legend(fontsize=7.5, frameon=False, ncols=2)

    ax = axes[1]
    vals = [means[c][8] for c in MATERIALS]
    order = np.argsort(vals)[::-1]
    ax.bar(range(len(MATERIALS)), [vals[i] for i in order],
           color=[CLASS_COLORS[MATERIALS[i]] for i in order])
    bar_axes(ax, "high_freq_energy")
    ax.set_xticks(range(len(MATERIALS)), [NICE[MATERIALS[i]] for i in order], rotation=25, ha="right")
    ax.set_title("high_freq_energy, ranked\npaper and cardboard carry the most fine texture", fontsize=9.5)
    for i, v in enumerate([vals[i] for i in order]):
        ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=7.5)

    fig.suptitle(
        "Frequency block, measured per class: the 9 FFT features of the analysis vector",
        fontsize=11.5, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    save(fig, "feat_frequency_bins.png")


# ---------------------------------------- 10. reuse the committed spectra figures

def copy_frequency_analysis_figures():
    """Bring the existing frequency-analysis renders into the site's figure folder.

    Only the per-class average spectra are reused; the radial-profile and band-energy
    renders in that folder are superseded by fig_frequency_bins_by_class() above, which
    measures the same thing with error bars and the site's palette.
    """
    jobs = [
        ("average_spectra_grid.png", "prep_average_spectra_grid.png", 1500),
    ]
    for src_name, dst_name, max_width in jobs:
        src = FREQ_ANALYSIS / src_name
        if not src.exists():
            print(f"[skip] {src} missing")
            continue
        img = cv2.imread(str(src), cv2.IMREAD_UNCHANGED)
        if max_width and img.shape[1] > max_width:
            scale = max_width / img.shape[1]
            img = cv2.resize(img, (max_width, int(img.shape[0] * scale)),
                             interpolation=cv2.INTER_AREA)
        out = WEB / dst_name
        cv2.imwrite(str(out), img, [cv2.IMWRITE_PNG_COMPRESSION, 9])
        print(f"[copy] {dst_name}  ({out.stat().st_size / 1024:.0f} KB)")


def main():
    WEB.mkdir(parents=True, exist_ok=True)
    fig_spatial_domain()
    fig_augmentation()
    fig_size_filter()
    fig_frequency_domain()
    fig_color_histograms()
    fig_texture_histograms()
    fig_hog_histogram()
    fig_scalar_distributions()
    fig_frequency_bins_by_class()
    copy_frequency_analysis_figures()


if __name__ == "__main__":
    main()
