# -*- coding: utf-8 -*-
# python/plot_roi_vtk.py
import argparse, json, os, sys, inspect
from pathlib import Path
import numpy as np
from vtk_viewer import show_volume  # já existe no projeto

def ram_path(name: str) -> Path:
    import tempfile
    base = Path(tempfile.gettempdir())
    return base / name

def must_exist(p: Path, what: str):
    if not p.exists():
        print(f"[ERROR] Missing {what}: {p}", file=sys.stderr)
        sys.stderr.flush()
        sys.exit(2)

def main():
    ap = argparse.ArgumentParser(description="Plot a single ROI (VTK) by its id.")
    ap.add_argument("--roi-id", type=int, required=True)
    ap.add_argument("--offscreen", action="store_true")
    ap.add_argument("--blend", type=str, default="linear",
                    help="Transfer function preset (linear, bone, etc.)")
    ap.add_argument("--title", type=str, default=None,
                    help="Window title (if supported)")
    args = ap.parse_args()

    p_index = ram_path("_rois.json")
    must_exist(p_index, "_rois.json")
    must_exist(p_index.with_suffix(".ok"), "_rois.json.ok")

    data = json.loads(p_index.read_text(encoding="utf-8"))
    items = data.get("items", [])
    roi = next((it for it in items if int(it["id"]) == args.roi_id), None)
    if roi is None:
        print(f"[ERROR] ROI id {args.roi_id} not found.", file=sys.stderr)
        sys.exit(3)

    p_roi = Path(roi["roi_path"])
    must_exist(p_roi, f"roi {args.roi_id} npy")
    vol = np.load(p_roi, mmap_mode=None, allow_pickle=False)

    print(f"[INFO] Plotting ROI {args.roi_id}: shape={vol.shape}")

    # --- compat: só passa kwargs que a função realmente aceita
    sig = inspect.signature(show_volume)
    accepted = set(sig.parameters.keys())

    # mapeia opções comuns
    call_kwargs = {}
    if "offscreen" in accepted:
        call_kwargs["offscreen"] = bool(args.offscreen)
    # alguns viewers usam "preset", outros "blend"
    if "preset" in accepted:
        call_kwargs["preset"] = args.blend
    elif "blend" in accepted:
        call_kwargs["blend"] = args.blend
    # poucos aceitam "title"
    if args.title and "title" in accepted:
        call_kwargs["title"] = args.title

    try:
        show_volume(vol, **call_kwargs)
    except TypeError:
        # último recurso: chama sem kwargs
        show_volume(vol)

    print("[OK] Done.")
    sys.exit(0)

if __name__ == "__main__":
    main()
