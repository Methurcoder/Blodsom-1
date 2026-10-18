"""TRELLIS.2 with the same weights and computation, holding at most one large
model in memory at a time.

The stock pipeline loads every model at start (13 GB of flow models alone) and
its low-VRAM mode only moves them between GPU and CPU, which on Apple Silicon is
the same memory. Here each model is a proxy that loads its weights straight onto
the device on first use and unloads whatever other model was resident, so the
pipeline code runs unchanged for every pipeline type (512, 1024, 1024_cascade,
1536_cascade), including the cascade that uses two flow models in one call.

Loading never consumes the random number generator, so a seed gives the same
result as the stock pipeline.
"""
import gc
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from PIL import Image

from trellis2 import models as trellis_models
from trellis2.pipelines import samplers
from trellis2.pipelines.trellis2_image_to_3d import Trellis2ImageTo3DPipeline


def default_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def release(device):
    gc.collect()
    if device.type == "mps":
        torch.mps.synchronize()
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()


TRIM_DECODER_CACHE = False
LOAD_SECONDS = {}  # seconds spent reading each model's weights
# Read the next stage's weights into RAM while the current stage computes.
PREFETCH = False

# Order in which the pipeline uses its models (Trellis2ImageTo3DPipeline.run).
USE_ORDER = {
    "512": ["sparse_structure_flow_model", "sparse_structure_decoder", "shape_slat_flow_model_512",
            "tex_slat_flow_model_512", "shape_slat_decoder", "tex_slat_decoder"],
    "1024": ["sparse_structure_flow_model", "sparse_structure_decoder", "shape_slat_flow_model_1024",
             "tex_slat_flow_model_1024", "shape_slat_decoder", "tex_slat_decoder"],
    "cascade": ["sparse_structure_flow_model", "sparse_structure_decoder", "shape_slat_flow_model_512",
                "shape_slat_decoder", "shape_slat_flow_model_1024", "tex_slat_flow_model_1024",
                "shape_slat_decoder", "tex_slat_decoder"],
}

_DTYPES = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32, "F64": torch.float64,
           "I64": torch.int64, "I32": torch.int32, "I16": torch.int16, "I8": torch.int8, "U8": torch.uint8,
           "BOOL": torch.bool}


def read_safetensors(path, pinned=None):
    """Reads a .safetensors file front to back into one RAM buffer and returns
    tensors that are views into it (no second copy). Same values as safe_open.
    With `pinned` (a page-locked uint8 tensor at least the file's size) the
    GPU copies straight from it, several times faster than from pageable RAM."""
    import struct
    size = os.path.getsize(path)
    if pinned is not None:
        buffer = pinned[:size]
        view = memoryview(buffer.numpy())
    else:
        buffer = torch.empty(size, dtype=torch.uint8)
        view = memoryview(buffer.numpy())
    with open(path, "rb", buffering=0) as f:
        read = 0
        while read < size:
            read += f.readinto(view[read:read + (64 << 20)])
    header_size = struct.unpack("<Q", bytes(view[:8]))[0]
    header = json.loads(bytes(view[8:8 + header_size]))
    base = 8 + header_size
    tensors = {}
    for name, info in header.items():
        if name == "__metadata__":
            continue
        dtype = _DTYPES[info["dtype"]]
        start, end = info["data_offsets"]
        raw = buffer[base + start:base + end]
        tensors[name] = raw.view(dtype).reshape(info["shape"]) if end > start else torch.empty(info["shape"], dtype=dtype)
    return tensors


class Prefetcher:
    """One background reader: the next model's file goes into RAM while the
    GPU works on the current one."""

    def __init__(self, device, largest):
        from concurrent.futures import ThreadPoolExecutor
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.pending = {}
        # One page-locked buffer, reused: a model is copied to the GPU from it
        # before the next model is read into it.
        self.pinned = torch.empty(largest, dtype=torch.uint8).pin_memory() if device.type == "cuda" else None

    def schedule(self, path):
        if path not in self.pending:
            pinned = self.pinned
            if pinned is not None and os.path.getsize(f"{path}.safetensors") > pinned.numel():
                pinned = None
            self.pending[path] = self.pool.submit(read_safetensors, f"{path}.safetensors", pinned)

    def release_pinned(self):
        """Gives the page-locked buffer back once only small models remain;
        they are read into ordinary memory instead."""
        if self.pinned is not None:
            self.pinned = None
            import gc
            gc.collect()
            empty = getattr(torch._C, "_host_emptyCache", None)
            if empty:
                empty()

    def take(self, path):
        future = self.pending.pop(path, None)
        return future.result() if future else None
# Keep the transformer blocks of flow models in CPU memory and send one block
# at a time to the device while it runs (for GPUs with about 4 GB).
STREAM_FLOW_BLOCKS = False
PIN_STREAMED_BLOCKS = False
# torch.compile every transformer block of the flow models (fuses the many
# small element-wise operations; rounding of the last bits may differ).
COMPILE_BLOCKS = False


def _stream_blocks(model, device):
    """Each block's weights stay in CPU memory; before a block runs its
    parameters point to a device copy, and after it they point back to the
    CPU tensors, so the device holds about one block at a time."""
    # Pinned copies would speed transfers but double the CPU memory of the
    # blocks; the weights stay as the memory-mapped tensors from the file.
    pin = PIN_STREAMED_BLOCKS and device.type == "cuda"
    for block in model.blocks:
        params = [(module, name, param) for module in block.modules()
                  for name, param in module.named_parameters(recurse=False)]
        cpu = [param.data.pin_memory() if pin else param.data for _, _, param in params]

        def before(_module, _inputs, params=params, cpu=cpu):
            for (_, _, param), host in zip(params, cpu):
                param.data = host.to(device, non_blocking=pin)

        def after(_module, _inputs, _output, params=params, cpu=cpu):
            for (_, _, param), host in zip(params, cpu):
                param.data = host

        block.register_forward_pre_hook(before)
        block.register_forward_hook(after)


def _trim_hook(module, inputs, output):
    torch.mps.empty_cache() if torch.backends.mps.is_available() else torch.cuda.empty_cache()


def load_direct(path, device, tensors=None):
    """Builds the model with empty parameters, then copies the saved tensors
    one at a time onto the device, each in its parameter's own dtype. The stock
    loader builds the whole model in float32 on the CPU first."""
    from accelerate import init_empty_weights
    from accelerate.utils import set_module_tensor_to_device
    from safetensors import safe_open
    with open(f"{path}.json") as f:
        config = json.load(f)
    cls = getattr(trellis_models, config["name"])
    with torch.random.fork_rng(devices=[]):
        with init_empty_weights(include_buffers=False):
            model = cls(**config["args"])
    dtypes = {name: p.dtype for name, p in model.named_parameters()}
    buffers = dict(model.named_buffers())
    stream = STREAM_FLOW_BLOCKS and "Flow" in config["name"] and hasattr(model, "blocks")
    cpu = torch.device("cpu")
    def place(name, value):
        if name in dtypes:
            target = cpu if stream and name.startswith("blocks.") else device
            set_module_tensor_to_device(model, name, target, value=value, dtype=dtypes[name])
        elif name in buffers:
            with torch.no_grad():
                buffers[name].copy_(value)

    if tensors is not None:
        # Already read into RAM (prefetched): copy each tensor to the device.
        for name, value in tensors.items():
            place(name, value)
    else:
        with safe_open(f"{path}.safetensors", "pt", device="cpu") as f:
            for name in f.keys():
                place(name, f.get_tensor(name))
    meta_buffers = [name for name, b in model.named_buffers() if b.device.type == "meta"]
    if meta_buffers:
        # Buffers computed in __init__ from parameters (rope phases): take them
        # from a zero-block copy, which costs almost nothing to build.
        args = dict(config["args"])
        args["num_blocks"] = 0
        with torch.random.fork_rng(devices=[]):
            small = dict(cls(**args).named_buffers())
        for name in meta_buffers:
            module_name, _, buffer_name = name.rpartition(".")
            module = model.get_submodule(module_name) if module_name else model
            module.register_buffer(buffer_name, small[name])
    empty = [name for name, p in model.named_parameters() if p.device.type == "meta"]
    if empty:
        raise RuntimeError(f"{os.path.basename(path)}: no saved weights for {empty[:5]}")
    if stream:
        # Everything but the blocks goes to the device; blocks stream.
        for child_name, child in model.named_children():
            if child_name != "blocks":
                child.to(device)
        for name, buffer in list(model.named_buffers(recurse=False)):
            model.register_buffer(name, buffer.to(device))
        _stream_blocks(model, device)
        model.eval()
    else:
        model = model.to(device).eval()
    if COMPILE_BLOCKS and "Flow" in config["name"] and hasattr(model, "blocks"):
        for i, block in enumerate(model.blocks):
            model.blocks[i] = torch.compile(block, dynamic=True)
    if TRIM_DECODER_CACHE and hasattr(model, "blocks") and "Decoder" in type(model).__name__:
        # Sparse decoders allocate differently sized temporaries in every
        # block; releasing the allocator cache between blocks keeps what the
        # device holds close to what is live.
        for stage in model.blocks:
            for block in (stage if isinstance(stage, torch.nn.ModuleList) else [stage]):
                block.register_forward_hook(_trim_hook)
        # Before every layer runs, hand back a large unused cache (what the
        # CUDA allocator does on its own when an allocation would fail).
        import conv_lean
        for module in model.modules():
            module.register_forward_pre_hook(lambda _m, _i: conv_lean.relieve())
    return model


class Residency:
    """Keeps at most one large model loaded; with PREFETCH, reads the model
    used next into RAM while the current one runs."""

    def __init__(self, device, on_load=None):
        self.device = device
        self.current = None
        self.on_load = on_load
        self.on_first_take = None
        self.proxies = {}
        self.order = []
        self.cursor = -1
        self.prefetcher = None

    def start(self, pipeline_type):
        if PREFETCH and self.prefetcher is None:
            largest = max(os.path.getsize(f"{p.path}.safetensors") for p in self.proxies.values())
            self.prefetcher = Prefetcher(self.device, largest)
        kind = "cascade" if "cascade" in pipeline_type else pipeline_type
        self.order = [name for name in USE_ORDER[kind] if name in self.proxies]
        self.cursor = -1
        if self.prefetcher and self.order:
            self.prefetcher.schedule(self.proxies[self.order[0]].path)

    def _prefetch_after(self, name):
        if not self.prefetcher or not self.order:
            return
        for i in range(self.cursor + 1, len(self.order)):
            if self.order[i] == name:
                self.cursor = i
                break
        following = self.order[self.cursor + 1] if self.cursor + 1 < len(self.order) else None
        remaining = self.order[self.cursor + 1:]
        if self.prefetcher.pinned is not None and not any("flow" in other for other in remaining):
            self.prefetcher.release_pinned()
        if following and following != name:
            self.prefetcher.schedule(self.proxies[following].path)

    def take(self, proxy):
        if self.current is proxy:
            return
        if self.current is not None:
            self.current._unload()
        self.current = proxy
        if self.on_first_take:
            self.on_first_take()
            self.on_first_take = None
        if self.on_load:
            self.on_load(proxy.name)

    def clear(self):
        if self.current is not None:
            self.current._unload()
        self.current = None


class LazyModel:
    """Stands in for a model; loads it on first use, unloads when another is used.
    `.to()` and `.cpu()` are no-ops: the weights live on the device while loaded."""

    def __init__(self, name, path, residency):
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "path", path)
        object.__setattr__(self, "residency", residency)
        object.__setattr__(self, "_model", None)

    def _get(self):
        self.residency.take(self)
        if self._model is None:
            import time
            started = time.time()
            prefetched = self.residency.prefetcher.take(self.path) if self.residency.prefetcher else None
            object.__setattr__(self, "_model", load_direct(self.path, self.residency.device, prefetched))
            del prefetched
            if self.residency.device.type == "cuda":
                torch.cuda.synchronize()  # copies out of the shared buffer are done
            self.residency._prefetch_after(self.name)
            LOAD_SECONDS[self.name] = LOAD_SECONDS.get(self.name, 0.0) + time.time() - started
        return self._model

    def _unload(self):
        object.__setattr__(self, "_model", None)
        release(self.residency.device)

    def to(self, *a, **k):
        return self

    def cpu(self):
        return self

    def cuda(self, *a, **k):
        return self

    def eval(self):
        return self

    def __call__(self, *a, **k):
        return self._get()(*a, **k)

    def __getitem__(self, index):
        # The pipeline reads `flow_model[0]` when the model is not an
        # nn.Module (a list of models); a single proxy stands for itself.
        if index == 0:
            return self
        raise IndexError(index)

    def __getattr__(self, attr):
        return getattr(self._get(), attr)

    def __setattr__(self, attr, value):
        setattr(self._get(), attr, value)


class TimmDinoV3:
    """TRELLIS.2's DINOv3 features (all tokens after the last block, layer-
    normalised without affine parameters) from timm's redistribution of the same
    weights, which needs no gated download. Loaded per call and dropped after."""

    def __init__(self, model_name="vit_large_patch16_dinov3.lvd1689m", image_size=512, device=None):
        self.model_name = model_name
        self.image_size = image_size
        self.device = device or default_device()

    def to(self, device):
        pass

    def cpu(self):
        pass

    model = None

    def _load_direct(self, timm):
        """Builds the timm model with empty parameters and copies the saved
        float32 weights one tensor at a time onto the device, instead of
        loading the whole model into CPU memory first (1.2 GB)."""
        from accelerate import init_empty_weights
        from accelerate.utils import set_module_tensor_to_device
        from huggingface_hub import hf_hub_download
        from safetensors import safe_open
        path = hf_hub_download(f"timm/{self.model_name}", "model.safetensors")
        with torch.random.fork_rng(devices=[]):
            with init_empty_weights(include_buffers=False):
                model = timm.create_model(self.model_name, pretrained=False, num_classes=0)
        dtypes = {name: p.dtype for name, p in model.named_parameters()}
        buffers = dict(model.named_buffers())
        with safe_open(path, "pt", device="cpu") as f:
            for name in f.keys():
                if name in dtypes:
                    set_module_tensor_to_device(model, name, self.device, value=f.get_tensor(name), dtype=dtypes[name])
                elif name in buffers:
                    with torch.no_grad():
                        buffers[name].copy_(f.get_tensor(name))
        missing = [name for name, p in model.named_parameters() if p.device.type == "meta"]
        if missing:
            raise RuntimeError(f"DINOv3: no saved weights for {missing[:5]}")
        model.norm = torch.nn.Identity()
        return model.to(self.device).eval()

    def drop(self):
        """Releases the model; the pipeline calls this when the first flow
        model is needed (after both conditioning resolutions are done)."""
        if self.model is not None:
            self.model = None
            release(self.device)

    @torch.no_grad()
    def __call__(self, images):
        import timm
        if self.model is None:
            self.model = self._load_direct(timm)
        model = self.model
        images = [i.resize((self.image_size, self.image_size), Image.LANCZOS) for i in images]
        arrays = [np.array(i.convert("RGB")).astype(np.float32) / 255 for i in images]
        batch = torch.stack([torch.from_numpy(a).permute(2, 0, 1) for a in arrays]).to(self.device)
        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device)[None, :, None, None]
        std = torch.tensor([0.229, 0.224, 0.225], device=self.device)[None, :, None, None]
        hidden = model.forward_features((batch - mean) / std)
        return F.layer_norm(hidden, hidden.shape[-1:])


class LazyRembg:
    """Background removal loaded only when an image has no alpha, dropped after."""

    def __init__(self, factory, device):
        self.factory = factory
        self.device = device

    def to(self, device):
        pass

    def cpu(self):
        pass

    def __call__(self, image):
        with torch.random.fork_rng(devices=[]):
            model = self.factory()
        model.to(self.device)
        try:
            return model(image)
        finally:
            del model
            release(self.device)


def checkpoint_path(repo, name):
    hf_hub_download(repo, f"{name}.json")
    path = hf_hub_download(repo, f"{name}.safetensors")
    return path[: -len(".safetensors")]


def from_pretrained(repo="microsoft/TRELLIS.2-4B", device=None, image_cond_model=None,
                    rembg_factory=None, pipeline_types=None, on_load=None):
    """Same pipeline.json as the stock pipeline. `pipeline_types` limits which
    flow models get downloaded (all are listed, none is loaded until used)."""
    device = device or default_device()
    with open(hf_hub_download(repo, "pipeline.json")) as f:
        args = json.load(f)["args"]
    wanted = None
    if pipeline_types:
        wanted = {"sparse_structure_decoder", "sparse_structure_flow_model", "shape_slat_decoder", "tex_slat_decoder"}
        for kind in pipeline_types:
            if kind in ("512",):
                wanted |= {"shape_slat_flow_model_512", "tex_slat_flow_model_512"}
            elif kind == "1024":
                wanted |= {"shape_slat_flow_model_1024", "tex_slat_flow_model_1024"}
            else:  # cascades
                wanted |= {"shape_slat_flow_model_512", "shape_slat_flow_model_1024", "tex_slat_flow_model_1024"}
    residency = Residency(device, on_load)
    lazy = {}
    for key, ref in args["models"].items():
        if wanted is not None and key not in wanted:
            continue
        parts = ref.split("/")
        if len(parts) > 2 and not ref.startswith("ckpts/"):
            path = checkpoint_path("/".join(parts[:2]), "/".join(parts[2:]))
        else:
            path = checkpoint_path(repo, ref)
        lazy[key] = LazyModel(key, path, residency)
        residency.proxies[key] = lazy[key]

    pipe = Trellis2ImageTo3DPipeline.__new__(Trellis2ImageTo3DPipeline)
    pipe.models = lazy
    pipe.sparse_structure_sampler = getattr(samplers, args["sparse_structure_sampler"]["name"])(**args["sparse_structure_sampler"]["args"])
    pipe.sparse_structure_sampler_params = args["sparse_structure_sampler"]["params"]
    pipe.shape_slat_sampler = getattr(samplers, args["shape_slat_sampler"]["name"])(**args["shape_slat_sampler"]["args"])
    pipe.shape_slat_sampler_params = args["shape_slat_sampler"]["params"]
    pipe.tex_slat_sampler = getattr(samplers, args["tex_slat_sampler"]["name"])(**args["tex_slat_sampler"]["args"])
    pipe.tex_slat_sampler_params = args["tex_slat_sampler"]["params"]
    pipe.shape_slat_normalization = args["shape_slat_normalization"]
    pipe.tex_slat_normalization = args["tex_slat_normalization"]
    pipe.image_cond_model = image_cond_model or TimmDinoV3(device=device)
    pipe.rembg_model = LazyRembg(rembg_factory, device) if rembg_factory else None
    pipe.low_vram = False  # models load onto the device on use; nothing to move
    pipe.default_pipeline_type = args.get("default_pipeline_type", "1024_cascade")
    pipe.pbr_attr_layout = {
        "base_color": slice(0, 3),
        "metallic": slice(3, 4),
        "roughness": slice(4, 5),
        "alpha": slice(5, 6),
    }
    pipe._device = device
    pipe._residency = residency
    original_run = pipe.run

    def run(image, *a, pipeline_type=None, **k):
        residency.start(pipeline_type or pipe.default_pipeline_type)
        if hasattr(pipe.image_cond_model, "drop"):
            residency.on_first_take = pipe.image_cond_model.drop
        try:
            return original_run(image, *a, pipeline_type=pipeline_type, **k)
        finally:
            if hasattr(pipe.image_cond_model, "drop"):
                pipe.image_cond_model.drop()

    pipe.run = run
    return pipe
