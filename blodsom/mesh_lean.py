"""The stub mesh extraction (trellis-mac's stubs/o_voxel_override_convert.py)
with its two Python dictionary loops replaced by sorted-key lookups.

The stub looks up ~2 million voxels and ~7 million edge neighbours one at a
time in a dict, which takes many minutes on a slower CPU. Here the same lookup
runs as array operations: a stable sort plus searchsorted(right) - 1 returns,
for each query, the last voxel with that coordinate (what the dict holds), or
0xFFFFFFFF when there is none. Everything else is the stub's own code.
"""
import os

import env

STUB = os.path.join(env.TRELLIS_MAC, "stubs", "o_voxel_override_convert.py")

START = "    # Build coordinate lookup on CPU"
END = "    connected_voxel_valid = "

LOOKUP = '''    # Coordinate lookup as array operations (replaces the dict loops).
    edge_neighbor_voxel = coords.reshape(N, 1, 1, 3) + _edge_neighbor_voxel_offset
    connected_voxel = edge_neighbor_voxel[intersected_flag]
    M = connected_voxel.shape[0]

    if M == 0:
        return torch.zeros(0, 3, device=device), torch.zeros(0, 3, dtype=torch.long, device=device)

    _c = coords.long()
    _span = int(_c.max().item()) + 3
    _keys = (_c[:, 0] * _span + _c[:, 1]) * _span + _c[:, 2]
    _sorted, _order = torch.sort(_keys, stable=True)
    _q = connected_voxel.reshape(-1, 3).long()
    _qkeys = (_q[:, 0] * _span + _q[:, 1]) * _span + _q[:, 2]
    _pos = (torch.searchsorted(_sorted, _qkeys, right=True) - 1).clamp_(min=0)
    _found = (_sorted[_pos] == _qkeys) & (_q >= 0).all(dim=1)
    connected_voxel_indices = torch.where(
        _found, _order[_pos], torch.full_like(_pos, 0xFFFFFFFF)).to(torch.int64).reshape(M, 4)
'''


def lean_function():
    with open(STUB) as f:
        source = f.read()
    start, end = source.index(START), source.index(END)
    patched = source[:start] + LOOKUP + source[end:]
    namespace = {"__name__": "o_voxel_override_convert_lean", "__file__": STUB}
    exec(compile(patched, STUB + " (lean lookup)", "exec"), namespace)
    return namespace["flexible_dual_grid_to_mesh"]


def install():
    from trellis2.models.sc_vaes import fdg_vae
    fdg_vae.flexible_dual_grid_to_mesh = lean_function()
