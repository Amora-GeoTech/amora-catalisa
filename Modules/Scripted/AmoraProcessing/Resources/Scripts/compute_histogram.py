#!/usr/bin/env python3
# bridge_histogram.py
import os, sys, argparse, json
import numpy as np

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--basename", default="_tensor_cache.npy",
                   help="nome do arquivo .npy no /dev/shm (ex.: _tensor_cache.npy, _tensor_bin.npy)")
    p.add_argument("--bins", type=int, default=256)
    p.add_argument("--logy", action="store_true")
    p.add_argument("--normalize", action="store_true")
    p.add_argument("--out", default="_histogram.png",
                   help="nome do PNG de saída em /dev/shm")
    args = p.parse_args()

    import tempfile
    base_dir = tempfile.gettempdir() if args.cache_mode == "ram" else "."

    # carrega tensor (qualquer shape), flutua dtype p/ segurança
    arr = np.load(src, mmap_mode="r")
    data = np.asarray(arr).ravel()

    # bins adequados ao dtype inteiro (ex.: uint8 => 256)
    bins = args.bins
    if np.issubdtype(data.dtype, np.integer):
        info = np.iinfo(data.dtype)
        # evita milhões de bins em dtypes largos:
        bins = min(args.bins, max(32, 256))

    counts, edges = np.histogram(data, bins=bins)
    if args.normalize:
        total = counts.sum()
        if total > 0:
            counts = counts / total

    # tenta gerar PNG com matplotlib; se não houver, salva CSV como fallback
    out_png = os.path.join("/dev/shm", args.out)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.figure()
        center = 0.5 * (edges[:-1] + edges[1:])
        plt.plot(center, counts)
        plt.xlabel("Intensidade / valor do voxel")
        plt.ylabel("Frequência" + (" (normalizada)" if args.normalize else ""))
        if args.logy:
            plt.yscale("log")
        plt.title(f"Histograma: {args.basename}")
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
