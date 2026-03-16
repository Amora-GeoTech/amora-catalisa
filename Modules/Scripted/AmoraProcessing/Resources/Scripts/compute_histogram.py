#!/usr/bin/env python3
# compute_histogram.py
import os, sys, argparse, json, tempfile
import numpy as np

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--basename", default="_tensor_cache.npy",
                   help="Name of the .npy file in temp dir")
    p.add_argument("--bins", type=int, default=256)
    p.add_argument("--logy", action="store_true")
    p.add_argument("--normalize", action="store_true")
    p.add_argument("--out", default="_histogram.png",
                   help="Output PNG filename in temp dir")
    args = p.parse_args()

    base_dir = tempfile.gettempdir()
    src = os.path.join(base_dir, args.basename)

    if not os.path.exists(src):
        print(json.dumps({"error": f"Source not found: {src}"}))
        return 2

    # Load tensor, flatten for histogram
    arr = np.load(src, mmap_mode="r")
    data = np.asarray(arr).ravel()

    # Appropriate bins for integer dtypes
    bins = args.bins
    if np.issubdtype(data.dtype, np.integer):
        bins = min(args.bins, 256)

    counts, edges = np.histogram(data, bins=bins)
    if args.normalize:
        total = counts.sum()
        if total > 0:
            counts = counts / total

    out_png = os.path.join(base_dir, args.out)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.figure()
        center = 0.5 * (edges[:-1] + edges[1:])
        plt.plot(center, counts)
        plt.xlabel("Voxel Intensity")
        plt.ylabel("Frequency" + (" (normalized)" if args.normalize else ""))
        if args.logy:
            plt.yscale("log")
        plt.title(f"Histogram: {args.basename}")
        plt.tight_layout()
        plt.savefig(out_png, dpi=120)
        print(json.dumps({"ok": True, "png": out_png}), flush=True)
    except Exception as e:
        # fallback CSV
        out_csv = out_png.replace(".png", ".csv")
        np.savetxt(out_csv, np.vstack([edges[:-1], edges[1:], counts]).T,
                   delimiter=",", header="bin_left,bin_right,count", comments="")
        print(json.dumps({"ok": True, "csv": out_csv}), flush=True)
    return 0

if __name__ == "__main__":
    sys.exit(main())
