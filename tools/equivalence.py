"""Checks that trellis2_lite gives exactly the stock pipeline's result.

1. weights: every tensor from the lite loader equals the stock loader's
   (value and dtype), for each checkpoint the pipeline type uses;
2. end to end: the stock pipeline (all models loaded up front, stock loader,
   low_vram moves) and the lite pipeline, same image and seed, give identical
   latents and mesh.

Both use the same DINOv3 (timm). The original side needs memory for every
model at once (about 24 GB with the 1024 types).

    python tools/equivalence.py IMAGE.png --type 512 [--skip-weights] [--loader-only]
"""
import os
import sys

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("ATTN_BACKEND", "sdpa")
os.environ.setdefault("SPARSE_ATTN_BACKEND", "sdpa")
os.environ.setdefault("SPARSE_CONV_BACKEND", "none")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "blodsom"))
import env  # noqa: E402,F401  (TRELLIS.2 on the path, backends set)

import argparse
import gc

import torch
from PIL import Image

import trellis2_lite as lite
from trellis2 import models as trellis_models
from trellis2.pipelines.trellis2_image_to_3d import Trellis2ImageTo3DPipeline

CPU = torch.device("cpu")


def check_weights(pipe):
    ok = True
    for key, proxy in pipe.models.items():
        stock = trellis_models.from_pretrained(proxy.path).state_dict()
        mine = lite.load_direct(proxy.path, CPU).state_dict()
        bad = [n for n in stock if n not in mine or stock[n].dtype != mine[n].dtype or not torch.equal(stock[n], mine[n])]
        extra = [n for n in mine if n not in stock]
        print(f"weights {key:<32} tensors={len(stock):5d} mismatched={len(bad)} extra={len(extra)}", flush=True)
        ok &= not bad and not extra
        del stock, mine
        gc.collect()
    return ok


def run(pipe, image, kind, seed):
    with torch.no_grad():
        meshes, (shape, tex, res) = pipe.run(image, seed=seed, pipeline_type=kind, return_latent=True)
    mesh = meshes[0]
    return {
        "shape_coords": shape.coords.cpu(), "shape_feats": shape.feats.float().cpu(),
        "tex_feats": tex.feats.float().cpu(), "res": res,
        "vertices": mesh.vertices.cpu(), "faces": mesh.faces.cpu(),
    }


def stock_pipeline(lite_pipe, device):
    loaded = {key: trellis_models.from_pretrained(proxy.path) for key, proxy in lite_pipe.models.items()}
    pipe = Trellis2ImageTo3DPipeline(
        models=loaded,
        sparse_structure_sampler=lite_pipe.sparse_structure_sampler,
        shape_slat_sampler=lite_pipe.shape_slat_sampler,
        tex_slat_sampler=lite_pipe.tex_slat_sampler,
        sparse_structure_sampler_params=lite_pipe.sparse_structure_sampler_params,
        shape_slat_sampler_params=lite_pipe.shape_slat_sampler_params,
        tex_slat_sampler_params=lite_pipe.tex_slat_sampler_params,
        shape_slat_normalization=lite_pipe.shape_slat_normalization,
        tex_slat_normalization=lite_pipe.tex_slat_normalization,
        image_cond_model=lite.TimmDinoV3(device=device),
        rembg_model=None,
        low_vram=True,
        default_pipeline_type=lite_pipe.default_pipeline_type,
    )
    pipe._device = device
    return pipe


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    parser.add_argument("--type", default="512")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--skip-weights", action="store_true")
    parser.add_argument("--loader-only", action="store_true", help="compare loading and residency only (original convolution code)")
    a = parser.parse_args()
    device = lite.default_device()
    image = Image.open(a.image).convert("RGBA")

    pipe = lite.from_pretrained(device=device, pipeline_types=[a.type])
    if not a.skip_weights:
        print("weights identical:", check_weights(pipe), flush=True)

    # Blodsom-1 as generate.py runs it: lean convolution, ConvNeXt and mesh
    # extraction; the original code is put back for the stock run.
    import importlib
    import conv_lean
    from trellis2.models.sc_vaes import fdg_vae, sparse_unet_vae
    backend = importlib.import_module("trellis2.modules.sparse.conv.conv_none")
    saved = (backend.sparse_conv3d_forward, sparse_unet_vae.SparseConvNeXtBlock3d._forward,
             fdg_vae.flexible_dual_grid_to_mesh)
    if not a.loader_only:
        conv_lean.install()
        lite.TRIM_DECODER_CACHE = True
    mine = run(pipe, image, a.type, a.seed)
    pipe._residency.clear()
    del pipe
    lite.release(device)
    (backend.sparse_conv3d_forward, sparse_unet_vae.SparseConvNeXtBlock3d._forward,
     fdg_vae.flexible_dual_grid_to_mesh) = saved

    stock = stock_pipeline(lite.from_pretrained(device=device, pipeline_types=[a.type]), device)
    theirs = run(stock, image, a.type, a.seed)

    for name in mine:
        x, y = mine[name], theirs[name]
        if isinstance(x, torch.Tensor):
            same = x.shape == y.shape and torch.equal(x, y)
            diff = (x.float() - y.float()).abs().max().item() if x.shape == y.shape else float("nan")
            print(f"{name:<14} shape={tuple(x.shape)} identical={same} max_abs_diff={diff:.3g}")
        else:
            print(f"{name:<14} {x} vs {y} identical={x == y}")


if __name__ == "__main__":
    main()
