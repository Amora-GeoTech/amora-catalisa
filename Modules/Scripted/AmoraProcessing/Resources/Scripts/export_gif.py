# -*- coding: utf-8 -*-
"""
export_gif.py
====================

Generate a 360-degree rotating GIF of a volume rendered with VTK.

Usage examples (from Qt):

    # 8 s por volta, 120 frames (~15 fps efetivo)
    python export_gif.py --basename _tensor_cache.npy --axis y --rev-sec 8 --frames 120

    # compat: 6 s por volta a 60 fps (360 frames)
    python export_gif.py --basename _tensor_cache.npy --axis y --rev-sec 6 --fps 60
"""

import argparse
import os
import sys
import shutil
import traceback
from pathlib import Path

import numpy as np

try:
    import imageio
except Exception:
    imageio = None

try:
    import vtk
    from vtk.util import numpy_support as vtknp
except Exception as exc:
    print("[ERROR] VTK is required for GIF export: {}".format(exc), file=sys.stderr)
    sys.stderr.flush()
    sys.exit(1)


def ram_paths(basename: str):
    """Return the path and ok marker in temp dir for a given basename."""
    import tempfile
    shm = Path(tempfile.gettempdir())
    p = shm / basename
    return p, p.with_suffix(".ok")


def disk_paths(basename: str):
    """Return the path and ok marker next to this script for a given basename."""
    p = Path(__file__).with_name(basename)
    return p, p.with_suffix(".ok")


def parse_args():
    ap = argparse.ArgumentParser(description="Export a rotating GIF of a 3D volume rendered via VTK")
    ap.add_argument("--basename", default="_tensor_cache.npy",
                    help="Name of the .npy cache file (default: _tensor_cache.npy)")
    ap.add_argument("--axis", choices=["x", "y", "z"], default="y",
                    help="Axis around which to rotate (default: y)")

    # Controle robusto de velocidade:
    ap.add_argument("--rev-sec", type=float, default=6.0,
                    help="Seconds per full revolution (default: 6.0)")
    ap.add_argument("--frames", type=int, default=120,
                    help="Number of frames per revolution (default: 120)")
    # Compatibilidade: se fps fornecido, sobrescreve --frames
    ap.add_argument("--fps", type=int, default=None,
                    help="Optional FPS; if given, frames = round(rev-sec * fps)")

    ap.add_argument("--offscreen", action="store_true",
                    help="Render offscreen (no window). Default: on-screen with window")

    # Loop contínuo (remove frame duplicado no fechamento)
    ap.add_argument("--seamless", action="store_true", default=True,
                    help="Drop last frame so loop is seamless (default: on)")
    ap.add_argument("--no-seamless", dest="seamless", action="store_false",
                    help="Keep all frames (may stutter at loop)")

    # Diretório de saída
    ap.add_argument("--outdir", type=str, default=None,
                    help="Diretório para salvar o arquivo final (opcional). "
                         "Se não for definido, usa /dev/shm. "
                         "Também pode usar a env AMORA_EXPORT_DIR.")
    # Perguntar/selecionar diretório se não houver um definido
    ap.add_argument("--ask-outdir", action="store_true",
                    help="Perguntar/selecionar um diretório de saída se nenhum for informado.")

    return ap.parse_args()


def _ask_outdir_interactively() -> str | None:
    # 1) tentar Tkinter (GUI) se disponível
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        chosen = filedialog.askdirectory(title="Escolha a pasta para salvar o export")
        root.destroy()
        if chosen:
            return chosen
    except Exception:
        pass
    # 2) fallback: prompt no terminal
    try:
        print("[INFO] Nenhum --outdir/AMORA_EXPORT_DIR. Digite um diretório (ENTER p/ pular): ", end="", flush=True)
        resp = input().strip()
        return resp or None
    except Exception:
        return None


def _tensor_to_vtk_image(tensor: np.ndarray, spacing=(1.0, 1.0, 1.0)):
    """
    Convert a 3D numpy array (Z,X,Y) into a vtkImageData with axes transposed to (X,Y,Z).
    """
    if tensor.ndim != 3:
        raise ValueError(f"Expected 3D tensor, got shape={tensor.shape}")
    Z, X, Y = tensor.shape
    # transpose to (X,Y,Z) and cast to float32 for volume rendering
    vol_xyz = tensor.transpose(1, 2, 0).astype(np.float32, copy=False)
    vtk_array = vtknp.numpy_to_vtk(vol_xyz.ravel(order="F"), deep=True, array_type=vtk.VTK_FLOAT)
    image = vtk.vtkImageData()
    image.SetDimensions(int(X), int(Y), int(Z))
    image.SetSpacing(*[float(s) for s in spacing])
    image.SetOrigin(0.0, 0.0, 0.0)
    image.GetPointData().SetScalars(vtk_array)
    return image


def build_volume_renderer(tensor: np.ndarray):
    # handle boolean tensors (masks)
    if tensor.dtype == bool:
        tensor = tensor.astype(np.uint8) * 255

    image = _tensor_to_vtk_image(tensor)
    arr = tensor.transpose(1, 2, 0)
    vmin = float(np.percentile(arr, 1))
    vmax = float(np.percentile(arr, 99))
    if vmin == vmax:
        vmin, vmax = float(arr.min()), float(arr.max())

    opacity = vtk.vtkPiecewiseFunction()
    opacity.AddPoint(vmin, 0.00)
    opacity.AddPoint((vmin + vmax) * 0.5, 0.08)
    opacity.AddPoint(vmax, 0.20)

    color = vtk.vtkColorTransferFunction()
    color.AddRGBPoint(vmin, 0.0, 0.0, 0.0)
    color.AddRGBPoint(vmax, 1.0, 1.0, 1.0)

    prop = vtk.vtkVolumeProperty()
    prop.SetIndependentComponents(True)
    prop.SetScalarOpacity(opacity)
    prop.SetColor(color)
    prop.SetInterpolationTypeToLinear()
    prop.ShadeOn()
    prop.SetAmbient(0.2)
    prop.SetDiffuse(0.7)
    prop.SetSpecular(0.1)
    prop.SetSpecularPower(10.0)

    mapper = vtk.vtkSmartVolumeMapper()
    mapper.SetInputData(image)
    mapper.SetBlendModeToComposite()

    volume = vtk.vtkVolume()
    volume.SetMapper(mapper)
    volume.SetProperty(prop)

    renderer = vtk.vtkRenderer()
    renderer.AddVolume(volume)
    renderer.SetBackground(0.05, 0.06, 0.08)

    renwin = vtk.vtkRenderWindow()
    renwin.AddRenderer(renderer)
    renwin.SetSize(900, 700)

    # **importante**: reset depois de adicionar à janela
    renderer.ResetCamera()
    # centro e tamanho do volume
    b = volume.GetBounds()  # (xmin,xmax, ymin,ymax, zmin,zmax)
    cx = 0.5*(b[0]+b[1]); cy = 0.5*(b[2]+b[3]); cz = 0.5*(b[4]+b[5])
    # raio para orbitar a uma distância “segura”
    diag = ((b[1]-b[0])**2 + (b[3]-b[2])**2 + (b[5]-b[4])**2)**0.5
    radius = 2.0*diag  # ajuste seu "zoom" aqui (ex.: 0.75*diag, 1.0*diag, etc.)

    return renderer, renwin, volume, (cx, cy, cz), radius


def capture_rotating_gif(
    tensor: np.ndarray,
    out_path: Path,
    axis: str,
    n_frames: int,
    frame_duration: float,
    show_window: bool,
    seamless: bool
) -> None:
    if imageio is None:
        print("[ERROR] imageio is not available; cannot write GIF.", file=sys.stderr)
        sys.exit(5)

    renderer, renwin, volume, center, radius = build_volume_renderer(tensor)

    # Offscreen rendering se solicitado
    if not show_window:
        renwin.OffScreenRenderingOn()

    # Interactor só se on-screen
    iren = None
    if show_window:
        iren = vtk.vtkRenderWindowInteractor()
        iren.SetRenderWindow(renwin)
        iren.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
        iren.AddObserver("KeyPressEvent", lambda o, e: (o.TerminateApp() if o.GetKeySym().lower() in ("escape","q") else None))

    if iren:
        iren.Initialize()
    renwin.Render()

    camera = renderer.GetActiveCamera()
    cx, cy, cz = center

    # defina ViewUp coerente por eixo e posição inicial
    if axis == "y":
        view_up = (0, 1, 0)
        # começa “na frente” e orbita no plano XZ
        camera.SetPosition(cx + radius, cy, cz)
    elif axis == "x":
        view_up = (0, 0, 1)  # manter Z “pra cima” ao orbitar no plano YZ
        camera.SetPosition(cx, cy + radius, cz)
    elif axis == "z":
        view_up = (0, 1, 0)
        camera.SetPosition(cx + radius, cy, cz)  # orbitar no plano XY
    else:
        raise ValueError("axis must be one of x,y,z")

    camera.SetFocalPoint(cx, cy, cz)
    camera.SetViewUp(*view_up)
    camera.ParallelProjectionOff()
    renderer.ResetCameraClippingRange()
    renwin.Render()

    # capturador da janela -> imagem
    w2i = vtk.vtkWindowToImageFilter()
    w2i.SetInput(renwin)
    w2i.SetScale(1)
    w2i.SetInputBufferTypeToRGB()
    w2i.ReadFrontBufferOff()  # funciona melhor em offscreen
    w2i.Update()

    frames = []
    for i in range(n_frames):
        # ângulo de 0..2π
        theta = 2.0 * np.pi * (i / float(n_frames))

        if axis == "y":
            x = cx + radius * np.cos(theta)
            y = cy
            z = cz + radius * np.sin(theta)
        elif axis == "x":
            x = cx
            y = cy + radius * np.cos(theta)
            z = cz + radius * np.sin(theta)
        else:  # "z"
            x = cx + radius * np.cos(theta)
            y = cy + radius * np.sin(theta)
            z = cz

        camera.SetPosition(x, y, z)
        camera.SetFocalPoint(cx, cy, cz)
        camera.SetViewUp(*view_up)
        renderer.ResetCameraClippingRange()

        renwin.Render()
        w2i.Modified()
        w2i.Update()

        vtk_image = w2i.GetOutput()
        width, height, _ = vtk_image.GetDimensions()
        vtk_array = vtk_image.GetPointData().GetScalars()
        arr = vtknp.vtk_to_numpy(vtk_array).reshape(height, width, -1)
        frames.append(arr[:, :, :3].copy())

        if iren:
            iren.ProcessEvents()

    # GIF (loop infinito). frame_duration em segundos por frame.
    imageio.mimsave(str(out_path), frames, duration=max(float(frame_duration), 0.01), loop=0)

    if iren:
        try:
            iren.TerminateApp()
        except Exception:
            pass
    renwin.Finalize()


def main() -> int:
    args = parse_args()
    basename = args.basename
    axis = args.axis
    rev_sec = max(float(args.rev_sec), 0.1)

    # frames efetivos: se --fps foi dado, sobrescreve --frames
    if args.fps is not None:
        n_frames = max(int(round(rev_sec * int(args.fps))), 1)
    else:
        n_frames = max(int(args.frames), 1)

    frame_duration = rev_sec / float(n_frames)
    show_window = not args.offscreen

    # 1) localizar cache como antes
    npy_path, ok_path = ram_paths(basename)
    if not (npy_path.exists() and ok_path.exists()):
        npy_path, ok_path = disk_paths(basename)
    if not (npy_path.exists() and ok_path.exists()):
        print("[ERROR] Cache not found for {}".format(basename), file=sys.stderr)
        sys.exit(2)

    # 2) ler tensor
    try:
        tensor = np.load(npy_path, mmap_mode=None)
    except Exception as exc:
        print(f"[ERROR] Failed to load {npy_path}: {exc}", file=sys.stderr)
        sys.exit(3)

    # 3) caminhos de saída
    gif_name = Path(basename).with_suffix('.gif').name

    # saída padrão (mantém pipeline atual em /dev/shm com .ok)
    shm_out_path, shm_ok = ram_paths(gif_name)

    # saída opcional do usuário (flag --outdir, env ou pergunta)
    user_outdir = args.outdir or os.getenv("AMORA_EXPORT_DIR")
    if args.ask_outdir and not user_outdir:
        user_outdir = _ask_outdir_interactively()

    user_out_path = None
    if user_outdir:
        user_outdir = Path(user_outdir).expanduser().resolve()
        try:
            user_outdir.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            print(f"[WARN] Não consegui criar diretório '{user_outdir}': {exc}. "
                  f"Vou prosseguir só com /dev/shm.", file=sys.stderr)
            user_outdir = None
        if user_outdir:
            user_out_path = user_outdir / gif_name

    # 4) gerar GIF em /dev/shm (comportamento original preservado)
    try:
        capture_rotating_gif(
            tensor, shm_out_path, axis=axis,
            n_frames=n_frames,
            frame_duration=frame_duration,
            show_window=show_window,
            seamless=bool(args.seamless)
        )
    except Exception:
        traceback.print_exc()
        sys.exit(4)

    # 5) copiar para outdir do usuário se solicitado
    if user_out_path:
        try:
            shutil.copy2(shm_out_path, user_out_path)
        except Exception as exc:
            print(f"[WARN] Falhou ao copiar para {user_out_path}: {exc}", file=sys.stderr)

    # 6) escrever marcador .ok (em /dev/shm) para o Qt
    try:
        shm_ok.write_text('ok', encoding='utf-8')
    except Exception:
        pass

    # 7) log final
    extra = f" | copy: {user_out_path}" if user_out_path else ""
    print(f"[OK] GIF saved: {shm_out_path}{extra} | rev-sec={rev_sec} | frames={n_frames} | "
          f"duration={frame_duration:.4f}s | seamless={bool(args.seamless)}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
