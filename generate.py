"""Image to 3D model (GLB) with TRELLIS.2, holding one model in memory at a time.

Same weights, same computation and the same result as the original pipeline
for a given image and seed; it needs far less memory and takes longer.

    python generate.py IMAGE.png [--output model.glb] [--type 1024_cascade] [--seed 42]

The image needs a transparent background (PNG with alpha): background removal
(RMBG-2.0 in the original) is not included.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "blodsom"))
import env  # noqa: E402,F401  (TRELLIS.2 on the path, backends set)

import argparse  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

GB = 2**30


def limit_ram(gb):
    """Windows: caps this process's physical memory; pages beyond the cap go to
    disk, as on a machine with that much RAM."""
    import ctypes
    kernel = ctypes.windll.kernel32
    kernel.GetCurrentProcess.restype = ctypes.c_void_p
    ok = kernel.SetProcessWorkingSetSizeEx(ctypes.c_void_p(kernel.GetCurrentProcess()),
                                           ctypes.c_size_t(256 << 20), ctypes.c_size_t(int(gb * GB)),
                                           ctypes.c_uint32(0x4 | 0x2))  # hard max, min disabled
    if not ok:
        print(f"RAM cap not set (error {ctypes.GetLastError()})", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Image to 3D with TRELLIS.2 in little memory")
    parser.add_argument("image", help="PNG with a transparent background")
    parser.add_argument("--output", help="GLB to write (default: next to the image)")
    parser.add_argument("--type", default="1024_cascade", choices=["512", "1024", "1024_cascade", "1536_cascade"],
                        help="pipeline type (default 1024_cascade, the original's default)")
    parser.add_argument("--seed", type=int, default=42, help="random seed (default 42, as the original)")
    parser.add_argument("--faces", type=int, default=0,
                        help="simplify the GLB to about this many faces (default: keep the full mesh)")
    parser.add_argument("--save-mesh", action="store_true", help="also save the raw mesh and voxels (.pt)")
    parser.add_argument("--stream", action="store_true",
                        help="send flow-model blocks to the GPU one at a time (for GPUs under 4 GB; much slower)")
    parser.add_argument("--no-prefetch", action="store_true",
                        help="do not read the next model while the current one runs (less RAM, slower)")
    parser.add_argument("--ram-limit", type=float, help="Windows: cap physical memory at this many GB")
    a = parser.parse_args()

    image = Image.open(a.image)
    if image.mode != "RGBA" or np.all(np.array(image.convert("RGBA"))[..., 3] == 255):
        raise SystemExit("The image has no transparent background. Remove the background first "
                         "(any tool), save as PNG with transparency, and run again.")
    output = a.output or os.path.splitext(a.image)[0] + ".glb"
    if a.ram_limit:
        limit_ram(a.ram_limit)

    import conv_lean
    import trellis2_lite as lite
    from export_glb import save_glb

    device = lite.default_device()
    conv_lean.install()
    lite.TRIM_DECODER_CACHE = True
    lite.STREAM_FLOW_BLOCKS = a.stream
    # Prefetching uses a page-locked buffer for fast copies to an NVIDIA GPU;
    # on Apple Silicon memory is shared, so there is nothing to gain.
    lite.PREFETCH = device.type == "cuda" and not a.no_prefetch

    print(f"device: {torch.cuda.get_device_name(0) if device.type == 'cuda' else device.type}", flush=True)
    pipe = lite.from_pretrained(device=device, pipeline_types=[a.type],
                                on_load=lambda name: print(f"  {name}", flush=True))
    started = time.time()
    with torch.no_grad():
        mesh = pipe.run(image.convert("RGBA"), seed=a.seed, pipeline_type=a.type)[0]
    generated = time.time() - started
    vertices = mesh.vertices.float().cpu().numpy()
    faces = mesh.faces.long().cpu().numpy()
    coords = mesh.coords.cpu().numpy()
    attrs = mesh.attrs.float().cpu().numpy()
    if a.save_mesh:
        torch.save({"vertices": mesh.vertices.cpu(), "faces": mesh.faces.cpu(), "attrs": mesh.attrs.cpu(),
                    "coords": mesh.coords.cpu()}, os.path.splitext(output)[0] + ".pt")
    del mesh, pipe
    v, f = save_glb(output, vertices, faces, coords, attrs, a.faces)
    print(f"generated in {generated:.0f}s; wrote {output} ({v} vertices, {f} faces)")


if __name__ == "__main__":
    main()
