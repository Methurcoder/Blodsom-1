"""Sparse submanifold 3D convolution with the same arithmetic as trellis-mac's
conv_none, using far less memory and time.

conv_none finds neighbours with a Python loop over every voxel and offset, keeps
three int64 tables, and adds each offset's contribution with scatter_add_ over
an index expanded to every output channel (E_k x C_out int64, gigabytes at
1024 resolution). Here neighbours are found with sorted keys and
searchsorted, kept per offset as int32, and added with index_add_. Within one
offset every output voxel has at most one neighbour, so the additions happen in
the same order on the same values as conv_none: the result is identical.

Install with `install()` after TRELLIS's sparse modules are imported.
"""
import torch


def _keys(coords, span):
    b, z, y, x = (coords[:, i].long() for i in range(4))
    return ((b * span + z) * span + y) * span + x


def _neighbours(coords, kernel, dilation):
    """Per kernel offset k (z, y, x order as conv_none): (source, target) int32,
    targets ascending, matching conv_none's edge order."""
    Kd, Kh, Kw = kernel
    dz, dy, dx = dilation
    pad = max(Kd // 2 * dz, Kh // 2 * dy, Kw // 2 * dx) + 1
    span = int(coords[:, 1:].max().item()) + 2 * pad + 1
    shifted = coords.long().clone()
    shifted[:, 1:] += pad
    keys = _keys(shifted, span)
    order = torch.argsort(keys)
    sorted_keys = keys[order]
    targets = torch.arange(coords.shape[0], device=coords.device)
    table = []
    for kz in range(Kd):
        for ky in range(Kh):
            for kx in range(Kw):
                oz, oy, ox = (kz - Kd // 2) * dz, (ky - Kh // 2) * dy, (kx - Kw // 2) * dx
                query = keys + ((oz * span + oy) * span + ox)
                pos = torch.searchsorted(sorted_keys, query).clamp_(max=sorted_keys.numel() - 1)
                found = sorted_keys[pos] == query
                table.append((order[pos[found]].int(), targets[found].int()))
    return table


# Off when memory is plentiful (resident mode): each call can wait for the GPU.
RELIEVE = True


def relieve():
    """Hands cached-but-unused device memory back before a large allocation.
    The CUDA allocator does this by itself when an allocation would fail; the
    MPS allocator reports out of memory instead, with the cache still held."""
    if RELIEVE and torch.backends.mps.is_available():
        if torch.mps.driver_allocated_memory() - torch.mps.current_allocated_memory() > 256 * 2**20:
            # Buffers of queued GPU work are only freed once it has run.
            torch.mps.synchronize()
            torch.mps.empty_cache()


def sparse_conv3d_forward(self, x):
    Co, Kd, Kh, Kw, Ci = self.weight.shape
    feats = x.feats
    cache_key = f"SubMConv3d_lean_neighbor_{Kw}x{Kh}x{Kd}_dilation{self.dilation}"
    table = x.get_spatial_cache(cache_key)
    if table is None:
        table = _neighbours(x.coords, (Kd, Kh, Kw), self.dilation)
        x.register_spatial_cache(cache_key, table)
    w = self.weight.reshape(Co, Kd * Kh * Kw, Ci).permute(1, 2, 0)
    relieve()
    out = torch.zeros(feats.shape[0], Co, device=feats.device, dtype=feats.dtype)
    # One gather buffer and one product buffer, sized for the busiest offset
    # and reused for all: the allocator never sees 27 different sizes.
    largest = max(source.numel() for source, _ in table)
    if largest:
        gathered = torch.empty(largest, Ci, device=feats.device, dtype=feats.dtype)
        product = torch.empty(largest, Co, device=feats.device, dtype=feats.dtype)
    for k, (source, target) in enumerate(table):
        n = source.numel()
        if n == 0:
            continue
        torch.index_select(feats, 0, source, out=gathered[:n])
        torch.mm(gathered[:n], w[k], out=product[:n])
        out.index_add_(0, target, product[:n])
        if NOTE and k == 0:
            NOTE()
    if self.bias is not None:
        out = out + self.bias
    return x.replace(out)


ROWS = 262144
NOTE = None  # measurement hook: called where temporaries peak


def convnext_forward(self, x):
    """SparseConvNeXtBlock3d._forward with the per-voxel norm and MLP run on
    ROWS voxels at a time: the MLP widens every voxel to four times its
    channels, which at 1024 resolution is gigabytes in one step."""
    h = self.conv(x)
    feats = h.feats
    out = torch.empty_like(feats)
    for start in range(0, feats.shape[0], ROWS):
        relieve()
        rows = feats[start:start + ROWS]
        hidden = self.mlp[1](self.mlp[0](self.norm(rows)))
        if NOTE:
            NOTE()
        out[start:start + ROWS] = self.mlp[2](hidden)
        del hidden
    return h.replace(out) + x


def install():
    """Replaces conv_none's forward (TRELLIS looks it up on every call) and the
    ConvNeXt block's forward."""
    import importlib
    backend = importlib.import_module("trellis2.modules.sparse.conv.conv_none")
    backend.sparse_conv3d_forward = sparse_conv3d_forward
    from trellis2.models.sc_vaes import sparse_unet_vae
    sparse_unet_vae.SparseConvNeXtBlock3d._forward = convnext_forward
    import mesh_lean
    mesh_lean.install()
    return backend
