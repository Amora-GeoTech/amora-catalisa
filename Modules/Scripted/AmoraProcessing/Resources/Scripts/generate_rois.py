# -*- coding: utf-8 -*-
# python/generate_rois.py
import argparse, json, os, sys
from pathlib import Path
import numpy as np
rng = np.random.default_rng()

def ram_path(name: str) -> Path:
    import tempfile
    base = Path(tempfile.gettempdir())
    return base / name

def must_exist(p: Path, what: str):
    if not p.exists():
        print(f"[ERROR] Missing {what}: {p}", file=sys.stderr)
        sys.exit(2)

def load_npy(p: Path) -> np.ndarray:
    return np.load(p, mmap_mode=None, allow_pickle=False)

def save_ok(data_path: Path):
    data_path.with_suffix(".ok").write_text("ok", encoding="utf-8")

def main():
    ap = argparse.ArgumentParser(
        description="Generate random ROI/Patches from Otsu binarized mask, saving crops from raw volume."
    )
    ap.add_argument("--bin-basename", default="_tensor_bin.npy",
                    help="Binarized mask basename in /dev/shm (default: _tensor_bin.npy)")
    ap.add_argument("--raw-basename", default="_tensor_cache.npy",
                    help="Raw tensor basename in /dev/shm (default: _tensor_cache.npy)")
    ap.add_argument("--roi-size-z", type=int, required=True, help="ROI size along Z (N)")
    ap.add_argument("--roi-size-y", type=int, required=True, help="ROI size along Y")
    ap.add_argument("--roi-size-x", type=int, required=True, help="ROI size along X")
    ap.add_argument("--num-rois", type=int, required=True, help="How many ROIs to generate")
    ap.add_argument("--save-mask", action="store_true", help="Also save mask crop for each ROI")
    ap.add_argument("--max-tries", type=int, default=20000, help="Safety cap for random sampling")
    args = ap.parse_args()

    p_bin = ram_path(args.bin_basename)
    p_raw = ram_path(args.raw_basename)
    must_exist(p_bin, "binarized mask")
    must_exist(p_bin.with_suffix(".ok"), "binarized .ok")
    must_exist(p_raw, "raw tensor")
    must_exist(p_raw.with_suffix(".ok"), "raw .ok")

    mask = load_npy(p_bin)
    vol  = load_npy(p_raw)

    if mask.shape != vol.shape:
        print(f"[ERROR] Shape mismatch: mask {mask.shape} vs raw {vol.shape}", file=sys.stderr)
        sys.exit(3)

    if mask.dtype != np.uint8:
        # tolerante: considera >0 como 1
        mask = (mask > 0).astype(np.uint8)

    Z, Y, X = mask.shape  # (N,Y,X)
    sz, sy, sx = int(args.roi_size_z), int(args.roi_size_y), int(args.roi_size_x)

    # margens p/ bbox caber inteiro
    mz, my, mx = sz // 2, sy // 2, sx // 2
    if (sz <= 0 or sy <= 0 or sx <= 0 or
        sz > Z or sy > Y or sx > X):
        print("[ERROR] Invalid ROI sizes for volume shape.", file=sys.stderr)
        sys.exit(4)

    # candidatos de centro: voxels 1 dentro das margens
    valid = np.zeros_like(mask, dtype=bool)
    valid[mz:Z-mz, my:Y-my, mx:X-mx] = True
    centers_ok = np.where((mask.astype(bool)) & valid)
    num_centers = centers_ok[0].size
    if num_centers == 0:
        print("[ERROR] No valid centers inside mask with margins. Adjust ROI size or mask.", file=sys.stderr)
        sys.exit(5)

    # se há menos centros que num_rois, amostra com reposição
    need = args.num_rois
    out_meta = []
    seen = set()

    # estratégia: sorteio até montar 'need' ROIs válidos, evitando centros repetidos
    tries = 0
    while len(out_meta) < need and tries < args.max_tries:
        tries += 1
        idx = rng.integers(0, num_centers)
        cz, cy, cx = int(centers_ok[0][idx]), int(centers_ok[1][idx]), int(centers_ok[2][idx])
        key = (cz, cy, cx)
        if key in seen:
            continue
        seen.add(key)

        z0, z1 = cz - mz, cz - mz + sz
        y0, y1 = cy - my, cy - my + sy
        x0, x1 = cx - mx, cx - mx + sx
        if z0 < 0 or y0 < 0 or x0 < 0 or z1 > Z or y1 > Y or x1 > X:
            continue  # segurança extra

        crop = vol[z0:z1, y0:y1, x0:x1]
        roi_id = len(out_meta) + 1
        p_crop = ram_path(f"_roi_{roi_id:03d}.npy")
        np.save(p_crop, crop)
        save_ok(p_crop)

        p_mask_crop = None
        if args.save_mask:
            mcr = mask[z0:z1, y0:y1, x0:x1]
            p_mask_crop = ram_path(f"_roi_{roi_id:03d}_mask.npy")
            np.save(p_mask_crop, mcr)
            save_ok(p_mask_crop)

        meta = {
            "id": roi_id,
            "center_zyx": [cz, cy, cx],
            "size_zyx": [sz, sy, sx],
            "bbox_zyx": [ [z0, z1], [y0, y1], [x0, x1] ],
            "roi_path": str(p_crop),
            "mask_path": str(p_mask_crop) if p_mask_crop else None
        }
        out_meta.append(meta)

    if len(out_meta) < need:
        print(f"[WARN] Generated {len(out_meta)}/{need} ROIs after {tries} tries. Consider smaller sizes or fewer ROIs.")

    p_index = ram_path("_rois.json")
    with open(p_index, "w", encoding="utf-8") as f:
        json.dump({"volume_shape_zyx": [Z, Y, X], "items": out_meta}, f, indent=2)
    save_ok(p_index)

    print(f"[OK] ROIs generated: {len(out_meta)}")
    print(f"[INFO] Index: {p_index}")
    for m in out_meta[:5]:
        print(f"  - id={m['id']:03d} center={m['center_zyx']} size={m['size_zyx']} -> {m['roi_path']}")
    if len(out_meta) > 5:
        print(f"  ... (+{len(out_meta)-5} more)")
    sys.exit(0)

if __name__ == "__main__":
    main()
