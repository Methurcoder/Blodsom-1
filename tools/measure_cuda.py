"""Exact per-stage memory of trellis2_lite on an NVIDIA GPU.

For each model in use: PyTorch's peak allocated (what the computation needs),
peak reserved (what PyTorch's allocator held), and the device's total use as
nvidia-smi sees it (includes the CUDA context), plus process RSS.

    python tools/measure_cuda.py IMAGE.png OUT_DIR --type 1024_cascade [--stream]
"""
import os
import sys

os.environ.setdefault("ATTN_BACKEND", "sdpa")
os.environ.setdefault("SPARSE_ATTN_BACKEND", "sdpa")
os.environ.setdefault("SPARSE_CONV_BACKEND", "none")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "blodsom"))
import env  # noqa: E402,F401  (TRELLIS.2 on the path, backends set)

import argparse
import threading
import time

import psutil
import torch
from PIL import Image

import conv_lean
import trellis2_lite as lite

GB = 2**30


class Stages:
    def __init__(self):
        self.order, self.rows = [], {}
        self.current = None
        self.process = psutil.Process()
        self.device_peak = 0.0
        self.rss_peak = 0.0
        self.private_peak = 0.0
        self._stop = False
        threading.Thread(target=self._poll, daemon=True).start()

    def _poll(self):
        while not self._stop:
            free, total = torch.cuda.mem_get_info()
            self.device_peak = max(self.device_peak, (total - free) / GB)
            info = self.process.memory_info()
            self.rss_peak = max(self.rss_peak, info.rss / GB)
            # Private (committed) memory: excludes file pages the OS can drop.
            self.private_peak = max(self.private_peak, getattr(info, "private", info.rss) / GB)
            time.sleep(0.02)

    def close(self):
        if self.current is not None:
            torch.cuda.synchronize()
            self.rows[self.current] = {
                "allocated": torch.cuda.max_memory_allocated() / GB,
                "reserved": torch.cuda.max_memory_reserved() / GB,
                "device": self.device_peak,
                "rss": self.rss_peak,
                "private": self.private_peak,
                "sec": time.time() - self.started,
            }

    def enter(self, name):
        self.close()
        torch.cuda.reset_peak_memory_stats()
        self.device_peak = 0.0
        self.rss_peak = 0.0
        self.private_peak = 0.0
        self.current = name
        self.started = time.time()
        if name not in self.order:
            self.order.append(name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    parser.add_argument("out")
    parser.add_argument("--type", default="1024_cascade")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--no-lean", action="store_true")
    parser.add_argument("--prefetch", action="store_true", help="read the next model into RAM while the current one runs")
    parser.add_argument("--attn", choices=["default", "cudnn"], default="default",
                        help="cudnn: prefer cuDNN attention (faster on this GPU; last-bit rounding differs)")
    parser.add_argument("--compile", action="store_true", help="torch.compile each flow-model block (needs Triton)")
    parser.add_argument("--steps", type=int, help="sampling steps for every stage (default: the model's 12)")
    parser.add_argument("--ram-limit", type=float, help="hard cap on this process's physical memory (GB, Windows)")
    a = parser.parse_args()
    if a.ram_limit:
        # Working set beyond the cap is paged out, as on a machine with less RAM.
        import ctypes
        kernel = ctypes.windll.kernel32
        kernel.GetCurrentProcess.restype = ctypes.c_void_p
        ok = kernel.SetProcessWorkingSetSizeEx(ctypes.c_void_p(kernel.GetCurrentProcess()),
                                               ctypes.c_size_t(256 << 20), ctypes.c_size_t(int(a.ram_limit * GB)),
                                               ctypes.c_uint32(0x4 | 0x2))  # hard max, min disabled
        print("working-set cap", a.ram_limit, "GB:", "set" if ok else f"failed ({ctypes.GetLastError()})", flush=True)
    os.makedirs(a.out, exist_ok=True)
    device = torch.device("cuda")
    if not a.no_lean:
        conv_lean.install()
        lite.TRIM_DECODER_CACHE = True
    lite.STREAM_FLOW_BLOCKS = a.stream
    lite.PREFETCH = a.prefetch
    stages = Stages()

    class StagedDino(lite.TimmDinoV3):
        def __call__(self, images):
            previous = stages.current
            stages.enter("dinov3")
            try:
                return super().__call__(images)
            finally:
                stages.enter(previous or "after dinov3")

    pipe = lite.from_pretrained(device=device, image_cond_model=StagedDino(device=device),
                                pipeline_types=[a.type], on_load=stages.enter)
    image = Image.open(a.image).convert("RGBA")
    lite.COMPILE_BLOCKS = a.compile
    import contextlib
    from torch.nn.attention import SDPBackend, sdpa_kernel
    attention = (sdpa_kernel([SDPBackend.CUDNN_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH], set_priority=True)
                 if a.attn == "cudnn" else contextlib.nullcontext())
    stages.enter("start")
    started = time.time()
    with torch.no_grad(), attention:
        params = {"steps": a.steps} if a.steps else {}
        meshes, (shape, tex, res) = pipe.run(image, seed=a.seed, pipeline_type=a.type, return_latent=True,
                                             sparse_structure_sampler_params=params, shape_slat_sampler_params=params,
                                             tex_slat_sampler_params=params)
    total = time.time() - started
    stages.enter("finish")
    stages.close()
    stages._stop = True
    mesh = meshes[0]
    torch.save({"vertices": mesh.vertices.cpu(), "faces": mesh.faces.cpu(), "attrs": mesh.attrs.cpu(),
                "coords": mesh.coords.cpu(), "tokens": int(shape.coords.shape[0])}, os.path.join(a.out, "mesh.pt"))
    lines = [f"gpu={torch.cuda.get_device_name(0)} type={a.type} attn={a.attn} compile={a.compile} prefetch={a.prefetch} steps={a.steps or 12} ram_limit={a.ram_limit} "
             f"tokens={int(shape.coords.shape[0])} vertices={mesh.vertices.shape[0]} total={total:.1f}s",
             f"{'model in use':<32}{'needed':>8}{'held':>8}{'device':>8}{'RSS':>8}{'private':>8}{'sec':>7}{'load':>7}"]
    for name in stages.order:
        r = stages.rows.get(name)
        if r:
            lines.append(f"{name:<32}{r['allocated']:8.2f}{r['reserved']:8.2f}{r['device']:8.2f}{r['rss']:8.2f}{r['private']:8.2f}{r['sec']:7.1f}{lite.LOAD_SECONDS.get(name, 0):7.1f}")
    text = "\n".join(lines)
    print(text)
    open(os.path.join(a.out, "memory.txt"), "w").write(text + "\n")


if __name__ == "__main__":
    main()
