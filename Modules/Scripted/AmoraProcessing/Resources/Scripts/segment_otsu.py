#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, os, sys
import numpy as np

def otsu_threshold(arr: np.ndarray, nbins: int = 256) -> float:
    a_min = float(np.nanmin(arr)); a_max = float(np.nanmax(arr))
    if not np.isfinite(a_min) or not np.isfinite(a_max) or a_max <= a_min:
        return a_min
    hist, edges = np.histogram(arr, bins=nbins, range=(a_min, a_max))
    hist = hist.astype(np.float64); total = hist.sum()
    if total <= 0: return a_min
    centers = 0.5*(edges[:-1] + edges[1:])
    p = hist / total
    w0 = np.cumsum(p)
    w1 = 1.0 - w0
    mu = np.cumsum(p * centers)
    mu_t = mu[-1]
    denom = (w0 * w1); denom[denom == 0] = np.nan
    sigma_b2 = (mu_t * w0 - mu) ** 2 / denom
    idx = int(np.nanargmax(sigma_b2))
    return float(edges[idx])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="_tensor_cache.npy",
                    help="basename do tensor fonte (.npy)")
    ap.add_argument("--dst", default="_tensor_bin.npy",
                    help="basename do binarizado (.npy)")
    ap.add_argument("--ok",  default="_tensor_bin.ok",
                    help="arquivo marcador de sucesso")
    ap.add_argument("--nbins", type=int, default=256)
    ap.add_argument("--cache-mode", choices=["disk","ram"], default="ram",
                    help="ram = /dev/shm ; disk = diretório atual")
    ap.add_argument("--mmap", action="store_true")
    ap.add_argument("--auto", action="store_true",
                    help="ignora --src e tenta detectar fonte em RAM")
    args = ap.parse_args()

    import tempfile
    base_dir = tempfile.gettempdir() if args.cache_mode == "ram" else "."
    src_path = os.path.join(base_dir, args.src)
    dst_path = os.path.join(base_dir, args.dst)
    ok_path  = os.path.join(base_dir, args.ok)

    if args.auto:
        # tenta achar algum cache conhecido na RAM
        candidatos = [
            os.path.join(base_dir, "_tensor_cache.npy"),
            os.path.join(base_dir, "_tensor_npy.npy"),
            os.path.join(base_dir, "_tensor_tiff.npy"),
        ]
        for c in candidatos:
            if os.path.exists(c):
                src_path = c
                break

    if not os.path.exists(src_path):
        print(f"[bridge_segment_otsu] ERRO: fonte não encontrada: {src_path}", file=sys.stderr)
        sys.exit(2)

    print(f"[bridge_segment_otsu] Lendo: {src_path}")
    arr = np.load(src_path, mmap_mode=("r" if args.mmap else None))
    arr = np.asarray(arr)  # garante indexável

    print(f"[bridge_segment_otsu] shape={arr.shape}, dtype={arr.dtype}")
    thr = otsu_threshold(arr, nbins=args.nbins)
    print(f"[bridge_segment_otsu] Otsu = {thr:.6g}")

    mask = (arr > thr).astype(np.uint8)

    os.makedirs(base_dir, exist_ok=True)
    np.save(dst_path, mask)
    with open(ok_path, "w", encoding="utf-8") as f:
        f.write("ok\n")

    print(f"[bridge_segment_otsu] Salvo: {dst_path}")
    print(f"[bridge_segment_otsu] OK:    {ok_path}")

if __name__ == "__main__":
    main()
