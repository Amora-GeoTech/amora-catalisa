#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Segmentation script for AMORA Processing.
Supports: Otsu, Multi-Otsu, Hysteresis thresholding.
Outputs JSON result to stdout for GUI integration.
"""
import argparse
import json
import os
import sys
import tempfile

import numpy as np


def otsu_threshold(arr, nbins=256):
    """Single Otsu threshold."""
    a_min, a_max = float(np.nanmin(arr)), float(np.nanmax(arr))
    if not np.isfinite(a_min) or not np.isfinite(a_max) or a_max <= a_min:
        return [a_min]
    hist, edges = np.histogram(arr, bins=nbins, range=(a_min, a_max))
    hist = hist.astype(np.float64)
    total = hist.sum()
    if total <= 0:
        return [a_min]
    centers = 0.5 * (edges[:-1] + edges[1:])
    p = hist / total
    w0 = np.cumsum(p)
    w1 = 1.0 - w0
    mu = np.cumsum(p * centers)
    mu_t = mu[-1]
    denom = w0 * w1
    denom[denom == 0] = np.nan
    sigma_b2 = (mu_t * w0 - mu) ** 2 / denom
    idx = int(np.nanargmax(sigma_b2))
    return [float(edges[idx])]


def multi_otsu_thresholds(arr, n_classes=3, nbins=256):
    """Multi-Otsu: find (n_classes - 1) thresholds that maximize between-class variance."""
    a_min, a_max = float(np.nanmin(arr)), float(np.nanmax(arr))
    if a_max <= a_min:
        return [a_min]

    hist, edges = np.histogram(arr, bins=nbins, range=(a_min, a_max))
    centers = 0.5 * (edges[:-1] + edges[1:])
    hist = hist.astype(np.float64)
    total = hist.sum()
    if total <= 0:
        return [a_min]

    p = hist / total
    n_thresholds = n_classes - 1

    if n_thresholds == 1:
        return otsu_threshold(arr, nbins)

    # For 3 classes (2 thresholds), exhaustive search
    if n_thresholds == 2:
        best_sigma = -1
        best_t1, best_t2 = 0, 1
        cum_p = np.cumsum(p)
        cum_mu = np.cumsum(p * centers)
        mu_t = cum_mu[-1]

        for t1 in range(1, nbins - 2):
            for t2 in range(t1 + 1, nbins - 1):
                w0 = cum_p[t1]
                w1 = cum_p[t2] - cum_p[t1]
                w2 = 1.0 - cum_p[t2]
                if w0 <= 0 or w1 <= 0 or w2 <= 0:
                    continue
                mu0 = cum_mu[t1] / w0
                mu1 = (cum_mu[t2] - cum_mu[t1]) / w1
                mu2 = (mu_t - cum_mu[t2]) / w2
                sigma = w0 * (mu0 - mu_t)**2 + w1 * (mu1 - mu_t)**2 + w2 * (mu2 - mu_t)**2
                if sigma > best_sigma:
                    best_sigma = sigma
                    best_t1, best_t2 = t1, t2
        return [float(edges[best_t1]), float(edges[best_t2])]

    # Fallback: quantile-based for n_classes > 3
    quantiles = [i / n_classes for i in range(1, n_classes)]
    thresholds = [float(np.percentile(arr, q * 100)) for q in quantiles]
    return thresholds


def hysteresis_threshold(arr, low_frac=0.3, high_frac=0.7):
    """
    Hysteresis thresholding: keeps connected regions where any voxel
    exceeds high_threshold, as long as they're above low_threshold.
    Good for noisy data where single threshold misses thin features.
    """
    from scipy import ndimage

    a_min, a_max = float(np.nanmin(arr)), float(np.nanmax(arr))
    low_t = a_min + low_frac * (a_max - a_min)
    high_t = a_min + high_frac * (a_max - a_min)

    strong = arr >= high_t
    weak = arr >= low_t

    # Label connected components in weak mask
    labeled, n_features = ndimage.label(weak)
    # Keep only components that contain at least one strong voxel
    strong_labels = set(np.unique(labeled[strong])) - {0}
    mask = np.isin(labeled, list(strong_labels))
    return mask.astype(np.uint8), low_t, high_t


def main():
    ap = argparse.ArgumentParser(description="AMORA segmentation")
    ap.add_argument("--method", default="otsu",
                    choices=["otsu", "multi_otsu", "hysteresis"],
                    help="Segmentation method")
    ap.add_argument("--n-classes", type=int, default=3,
                    help="Number of classes for multi-otsu")
    ap.add_argument("--low-frac", type=float, default=0.3,
                    help="Low threshold fraction for hysteresis (0-1)")
    ap.add_argument("--high-frac", type=float, default=0.7,
                    help="High threshold fraction for hysteresis (0-1)")
    ap.add_argument("--nbins", type=int, default=256)
    ap.add_argument("--output", default="",
                    help="Output path (empty = temp dir)")
    args = ap.parse_args()

    base_dir = tempfile.gettempdir()
    src_path = os.path.join(base_dir, "_tensor_cache.npy")

    if not os.path.exists(src_path):
        print(json.dumps({"error": f"Source not found: {src_path}"}))
        sys.exit(2)

    print(f"[SEG] Loading: {src_path}", flush=True)
    arr = np.load(src_path)
    print(f"[SEG] Shape={arr.shape}, dtype={arr.dtype}", flush=True)

    if args.method == "otsu":
        thresholds = otsu_threshold(arr, nbins=args.nbins)
        mask = (arr > thresholds[0]).astype(np.uint8)
        method_info = f"Otsu (threshold={thresholds[0]:.4f})"

    elif args.method == "multi_otsu":
        thresholds = multi_otsu_thresholds(arr, n_classes=args.n_classes, nbins=args.nbins)
        # Create labeled image: 0, 1, ..., n_classes-1
        mask = np.zeros_like(arr, dtype=np.uint8)
        for i, t in enumerate(thresholds):
            mask[arr > t] = i + 1
        method_info = f"Multi-Otsu ({args.n_classes} classes, thresholds={[f'{t:.4f}' for t in thresholds]})"

    elif args.method == "hysteresis":
        mask, low_t, high_t = hysteresis_threshold(arr, args.low_frac, args.high_frac)
        thresholds = [low_t, high_t]
        method_info = f"Hysteresis (low={low_t:.4f}, high={high_t:.4f})"

    # Determine output path
    if args.output:
        dst_path = args.output
    else:
        dst_path = os.path.join(base_dir, "_tensor_bin.npy")

    # Save
    np.save(dst_path, mask)
    ok_path = os.path.join(base_dir, "_tensor_bin.ok")
    with open(ok_path, "w", encoding="utf-8") as f:
        f.write("ok\n")

    # Print result as JSON for GUI
    result = {
        "ok": True,
        "method": args.method,
        "thresholds": thresholds,
        "output": dst_path,
        "shape": list(mask.shape),
        "info": method_info,
        "unique_labels": len(np.unique(mask)),
    }
    print(json.dumps(result), flush=True)
    print(f"[SEG] {method_info}", flush=True)
    print(f"[SEG] Saved: {dst_path}", flush=True)


if __name__ == "__main__":
    main()
