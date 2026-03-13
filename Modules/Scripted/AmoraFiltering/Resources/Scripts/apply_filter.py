#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
apply_filter.py - Apply image filters to the AMORA tensor cache.

Loads _tensor_cache.npy (or _tensor_filtered.npy if it exists),
applies the requested filter, and saves to _tensor_filtered.npy.

Filters:
  gaussian  - 3D Gaussian blur          (sigma)
  median    - 3D median filter           (size)
  clahe     - Slice-by-slice CLAHE       (clip_limit, kernel_size)
  unsharp   - 3D unsharp mask            (sigma, amount)
  nlm       - Non-local means denoising  (h, patch_size, patch_distance)

Usage:
  python apply_filter.py --filter gaussian --sigma 1.5
  python apply_filter.py --filter median --size 3
  python apply_filter.py --filter clahe --clip-limit 0.03 --kernel-size 8
  python apply_filter.py --filter unsharp --sigma 2.0 --amount 1.0
  python apply_filter.py --filter nlm --h 0.1 --patch-size 5 --patch-distance 6
  python apply_filter.py --reset   # discard filtered, revert to raw
"""
import sys
import argparse
import time
import tempfile
import traceback
from pathlib import Path
import numpy as np


def cache_path(name: str) -> Path:
    return Path(tempfile.gettempdir()) / name


def load_input():
    """Load the current working tensor: filtered if exists, else raw."""
    filt = cache_path("_tensor_filtered.npy")
    raw = cache_path("_tensor_cache.npy")

    if filt.exists():
        print(f"[filter] Loading filtered tensor: {filt}")
        return np.load(filt, mmap_mode=None).astype(np.float32, copy=False)
    elif raw.exists():
        print(f"[filter] Loading raw tensor: {raw}")
        return np.load(raw, mmap_mode=None).astype(np.float32, copy=False)
    else:
        raise FileNotFoundError(
            "No tensor in cache. Load data first.\n"
            f"  Looked for: {filt}\n"
            f"  And:        {raw}"
        )


def save_output(tensor: np.ndarray):
    """Save filtered tensor to cache."""
    out = cache_path("_tensor_filtered.npy")
    np.save(out, tensor)
    mark = cache_path("_tensor_filtered.npy.ok")
    mark.write_text("ok")
    print(f"[filter] Saved filtered tensor: {out}  shape={tensor.shape}")


# ═══════════════════════════════════════════════════════════════════════════
# Filters
# ═══════════════════════════════════════════════════════════════════════════

def apply_gaussian(tensor, sigma=1.0):
    """3D Gaussian blur using scipy."""
    from scipy.ndimage import gaussian_filter
    print(f"[filter] Gaussian blur: sigma={sigma}")
    return gaussian_filter(tensor, sigma=sigma).astype(np.float32)


def apply_median(tensor, size=3):
    """3D median filter using scipy."""
    from scipy.ndimage import median_filter
    size = int(size)
    if size % 2 == 0:
        size += 1  # ensure odd
    print(f"[filter] Median filter: size={size}")
    return median_filter(tensor, size=size).astype(np.float32)


def apply_clahe(tensor, clip_limit=0.03, kernel_size=8):
    """Slice-by-slice CLAHE (Contrast Limited Adaptive Histogram Equalization).
    Uses skimage.exposure.equalize_adapthist on each 2D slice along axis 0.
    """
    try:
        from skimage.exposure import equalize_adapthist
    except ImportError:
        raise ImportError(
            "scikit-image not installed. Install with: pip install scikit-image"
        )

    kernel_size = int(kernel_size)
    n_slices = tensor.shape[0]
    print(f"[filter] CLAHE: clip_limit={clip_limit}, kernel_size={kernel_size}, "
          f"slices={n_slices}")

    # Normalize full volume to [0, 1] for CLAHE
    vmin, vmax = float(tensor.min()), float(tensor.max())
    if vmax > vmin:
        norm = (tensor - vmin) / (vmax - vmin)
    else:
        norm = np.zeros_like(tensor)

    result = np.empty_like(norm, dtype=np.float32)
    for i in range(n_slices):
        if i % max(1, n_slices // 10) == 0:
            print(f"[filter]   CLAHE slice {i}/{n_slices}")
        slc = norm[i, :, :]
        enhanced = equalize_adapthist(
            slc,
            kernel_size=kernel_size,
            clip_limit=clip_limit,
        )
        result[i, :, :] = enhanced.astype(np.float32)

    # Scale back to original range
    result = result * (vmax - vmin) + vmin
    return result


def apply_unsharp(tensor, sigma=2.0, amount=1.0):
    """3D unsharp mask: output = original + amount * (original - blurred)."""
    from scipy.ndimage import gaussian_filter
    print(f"[filter] Unsharp mask: sigma={sigma}, amount={amount}")
    blurred = gaussian_filter(tensor, sigma=sigma)
    sharpened = tensor + amount * (tensor - blurred)
    return sharpened.astype(np.float32)


def apply_nlm(tensor, h=0.1, patch_size=5, patch_distance=6):
    """Non-local means denoising, slice-by-slice.
    Uses skimage.restoration.denoise_nl_means.
    """
    try:
        from skimage.restoration import denoise_nl_means, estimate_sigma
    except ImportError:
        raise ImportError(
            "scikit-image not installed. Install with: pip install scikit-image"
        )

    patch_size = int(patch_size)
    patch_distance = int(patch_distance)
    n_slices = tensor.shape[0]
    print(f"[filter] Non-local means: h={h}, patch_size={patch_size}, "
          f"patch_distance={patch_distance}, slices={n_slices}")

    # Normalize to [0, 1]
    vmin, vmax = float(tensor.min()), float(tensor.max())
    if vmax > vmin:
        norm = (tensor - vmin) / (vmax - vmin)
    else:
        norm = np.zeros_like(tensor)

    result = np.empty_like(norm, dtype=np.float32)
    for i in range(n_slices):
        if i % max(1, n_slices // 10) == 0:
            print(f"[filter]   NLM slice {i}/{n_slices}")
        slc = norm[i, :, :]
        sigma_est = estimate_sigma(slc)
        denoised = denoise_nl_means(
            slc,
            h=h * sigma_est if sigma_est > 0 else h,
            patch_size=patch_size,
            patch_distance=patch_distance,
            fast_mode=True,
        )
        result[i, :, :] = denoised.astype(np.float32)

    result = result * (vmax - vmin) + vmin
    return result


def apply_ring_removal(tensor, max_radius=300, ring_width=21, center_x=0, center_y=0):
    """Remove ring artifacts from CT/microCT data.

    For each axial slice:
      1. Convert Cartesian -> polar coordinates (centered on rotation axis)
      2. In polar space, ring artifacts become vertical stripes
      3. Compute column-wise (radial) median and subtract it to remove stripes
      4. Convert back to Cartesian

    Parameters based on Avizo Ring Artifact Removal tool.
    """
    from scipy.ndimage import median_filter as scipy_median

    ring_width = int(ring_width)
    if ring_width % 2 == 0:
        ring_width += 1
    max_radius = int(max_radius)

    n_slices, ny, nx = tensor.shape

    # Auto-detect center if not specified
    cx = center_x if center_x > 0 else nx // 2
    cy = center_y if center_y > 0 else ny // 2

    print(f"[filter] Ring Artifact Removal: center=({cx},{cy}), "
          f"max_radius={max_radius}, ring_width={ring_width}, slices={n_slices}")

    # Build polar coordinate maps (shared across slices)
    n_angles = max(360, int(2 * np.pi * max_radius))
    n_radii = max_radius

    theta = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
    radii = np.arange(n_radii).astype(np.float32)

    # Polar -> Cartesian mapping (for each (angle, radius) -> (x, y))
    map_x = cx + np.outer(np.cos(theta), radii)  # (n_angles, n_radii)
    map_y = cy + np.outer(np.sin(theta), radii)

    # Clip to image bounds
    map_xi = np.clip(map_x, 0, nx - 1).astype(np.float32)
    map_yi = np.clip(map_y, 0, ny - 1).astype(np.float32)

    # Integer indices for nearest-neighbor interpolation
    ix0 = np.floor(map_xi).astype(np.intp)
    iy0 = np.floor(map_yi).astype(np.intp)
    ix1 = np.minimum(ix0 + 1, nx - 1)
    iy1 = np.minimum(iy0 + 1, ny - 1)
    fx = map_xi - ix0
    fy = map_yi - iy0

    result = tensor.copy()

    for i in range(n_slices):
        if i % max(1, n_slices // 10) == 0:
            print(f"[filter]   Ring removal slice {i}/{n_slices}")

        slc = tensor[i]

        # Bilinear interpolation to polar
        polar = (slc[iy0, ix0] * (1 - fx) * (1 - fy) +
                 slc[iy0, ix1] * fx * (1 - fy) +
                 slc[iy1, ix0] * (1 - fx) * fy +
                 slc[iy1, ix1] * fx * fy)

        # Ring artifacts are constant along angle (columns in polar image)
        # Compute radial profile: median along angular axis for each radius
        radial_profile = np.median(polar, axis=0)  # shape (n_radii,)

        # Smooth the radial profile to get the "artifact" component
        smooth_profile = scipy_median(radial_profile, size=ring_width)

        # The ring artifact is the difference
        artifact = radial_profile - smooth_profile  # shape (n_radii,)

        # Subtract artifact from polar image (broadcast along angles)
        polar_corrected = polar - artifact[np.newaxis, :]

        # Map corrected polar back to Cartesian using scatter
        # For each polar pixel, write back to Cartesian
        corrected_slc = slc.copy()
        weight = np.zeros((ny, nx), dtype=np.float32)
        accum = np.zeros((ny, nx), dtype=np.float32)

        # Use integer coordinates for accumulation
        yi_flat = np.round(map_yi).astype(np.intp).ravel()
        xi_flat = np.round(map_xi).astype(np.intp).ravel()
        val_flat = polar_corrected.ravel()

        # Valid mask
        valid = (yi_flat >= 0) & (yi_flat < ny) & (xi_flat >= 0) & (xi_flat < nx)
        yi_v = yi_flat[valid]
        xi_v = xi_flat[valid]
        val_v = val_flat[valid]

        np.add.at(accum, (yi_v, xi_v), val_v)
        np.add.at(weight, (yi_v, xi_v), 1.0)

        # Where we have data, use the corrected values
        mask = weight > 0
        corrected_slc[mask] = accum[mask] / weight[mask]

        result[i] = corrected_slc

    return result.astype(np.float32)


def reset_filter():
    """Remove the filtered tensor cache (revert to raw data)."""
    filt = cache_path("_tensor_filtered.npy")
    mark = cache_path("_tensor_filtered.npy.ok")
    removed = False
    if filt.exists():
        filt.unlink()
        removed = True
    if mark.exists():
        mark.unlink()
    if removed:
        print("[filter] Filtered tensor removed. Reverted to raw data.")
    else:
        print("[filter] No filtered tensor to remove.")


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

FILTERS = {
    "gaussian": apply_gaussian,
    "median": apply_median,
    "clahe": apply_clahe,
    "unsharp": apply_unsharp,
    "nlm": apply_nlm,
    "ring_removal": apply_ring_removal,
}


def parse_args():
    ap = argparse.ArgumentParser(
        prog="apply_filter",
        description="Apply image filter to AMORA tensor cache.",
    )
    ap.add_argument("--filter", choices=list(FILTERS.keys()),
                    help="Filter to apply")
    ap.add_argument("--reset", action="store_true",
                    help="Remove filtered tensor (revert to raw)")

    # Gaussian
    ap.add_argument("--sigma", type=float, default=1.0,
                    help="Gaussian sigma / Unsharp sigma (default: 1.0)")
    # Median
    ap.add_argument("--size", type=int, default=3,
                    help="Median kernel size (default: 3)")
    # CLAHE
    ap.add_argument("--clip-limit", type=float, default=0.03,
                    help="CLAHE clip limit (default: 0.03)")
    ap.add_argument("--kernel-size", type=int, default=8,
                    help="CLAHE kernel size (default: 8)")
    # Unsharp
    ap.add_argument("--amount", type=float, default=1.0,
                    help="Unsharp mask amount (default: 1.0)")
    # NLM
    ap.add_argument("--h", type=float, default=0.1,
                    help="NLM denoising strength h (default: 0.1)")
    ap.add_argument("--patch-size", type=int, default=5,
                    help="NLM patch size (default: 5)")
    ap.add_argument("--patch-distance", type=int, default=6,
                    help="NLM patch distance (default: 6)")

    # Ring Artifact Removal
    ap.add_argument("--max-radius", type=int, default=300,
                    help="Ring removal max radius in pixels (default: 300)")
    ap.add_argument("--ring-width", type=int, default=21,
                    help="Ring removal median kernel width (default: 21)")
    ap.add_argument("--center-x", type=int, default=0,
                    help="Ring removal center X (0=auto, default: 0)")
    ap.add_argument("--center-y", type=int, default=0,
                    help="Ring removal center Y (0=auto, default: 0)")

    # Source override (use raw instead of filtered)
    ap.add_argument("--from-raw", action="store_true",
                    help="Force load from raw cache, ignoring filtered")

    return ap.parse_args()


def main():
    args = parse_args()

    if args.reset:
        reset_filter()
        return 0

    if not args.filter:
        print("[filter] ERROR: --filter required (or --reset). See --help.")
        return 1

    try:
        t0 = time.time()

        # Load input
        if args.from_raw:
            raw = cache_path("_tensor_cache.npy")
            if not raw.exists():
                print("[filter] ERROR: No raw tensor cache found.")
                return 1
            tensor = np.load(raw, mmap_mode=None).astype(np.float32, copy=False)
            print(f"[filter] Loaded RAW tensor: shape={tensor.shape}")
        else:
            tensor = load_input()

        print(f"[filter] Input: shape={tensor.shape}, "
              f"dtype={tensor.dtype}, "
              f"range=[{tensor.min():.2f}, {tensor.max():.2f}]")

        # Apply filter
        filter_fn = FILTERS[args.filter]

        if args.filter == "gaussian":
            result = filter_fn(tensor, sigma=args.sigma)
        elif args.filter == "median":
            result = filter_fn(tensor, size=args.size)
        elif args.filter == "clahe":
            result = filter_fn(tensor, clip_limit=args.clip_limit,
                              kernel_size=args.kernel_size)
        elif args.filter == "unsharp":
            result = filter_fn(tensor, sigma=args.sigma, amount=args.amount)
        elif args.filter == "nlm":
            result = filter_fn(tensor, h=args.h, patch_size=args.patch_size,
                              patch_distance=args.patch_distance)
        elif args.filter == "ring_removal":
            result = filter_fn(tensor, max_radius=args.max_radius,
                              ring_width=args.ring_width,
                              center_x=args.center_x,
                              center_y=args.center_y)

        elapsed = time.time() - t0
        print(f"[filter] Output: shape={result.shape}, "
              f"range=[{result.min():.2f}, {result.max():.2f}], "
              f"time={elapsed:.2f}s")

        save_output(result)
        print("[filter] OK")
        return 0

    except Exception as e:
        print(f"[filter] ERROR: {e}", file=sys.stderr)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
