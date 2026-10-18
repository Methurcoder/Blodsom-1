"""Writes a TRELLIS.2 result as a GLB that 3D tools and browsers open
directly.

Each vertex takes the base colour of the decoded voxel it lies in (the voxel
attributes hold base colour, metallic, roughness and alpha). The original
pipeline bakes these into a texture with CUDA/Metal helpers; vertex colours
need none. `--faces` simplifies the mesh first (fast-simplification).

    python blodsom/export_glb.py mesh.pt out.glb [--faces 400000]
"""
import argparse

import numpy as np
import torch


def vertex_colours(vertices, coords, attrs):
    """Base colour of the voxel each vertex falls in (nearest voxel when the
    vertex sits on a face between voxels)."""
    coords = coords.astype(np.int64)
    resolution = int(coords.max()) + 1
    span = resolution + 2
    keys = (coords[:, 0] * span + coords[:, 1]) * span + coords[:, 2]
    order = np.argsort(keys, kind="stable")
    sorted_keys = keys[order]
    cells = np.clip(np.floor((vertices + 0.5) * resolution).astype(np.int64), 0, resolution - 1)
    query = (cells[:, 0] * span + cells[:, 1]) * span + cells[:, 2]
    pos = np.clip(np.searchsorted(sorted_keys, query), 0, len(sorted_keys) - 1)
    found = sorted_keys[pos] == query
    index = order[pos]
    if not found.all():
        from scipy.spatial import cKDTree
        _, nearest = cKDTree(coords).query(cells[~found])
        index[~found] = nearest
    return np.clip(attrs[index, 0:3], 0, 1)


def save_glb(path, vertices, faces, coords, attrs, max_faces=0):
    """vertices (V, 3) float, faces (F, 3) int, coords (N, 3) voxel indices and
    attrs (N, 6) voxel attributes, all numpy; returns the written counts."""
    import trimesh
    if max_faces and len(faces) > max_faces:
        import fast_simplification
        vertices, faces = fast_simplification.simplify(vertices, faces.astype(np.int32),
                                                       target_reduction=1 - max_faces / len(faces))
    colours = vertex_colours(vertices, coords, attrs)
    # The original pipeline stores base colour in a texture, which glTF reads
    # as sRGB; glTF vertex colours are linear, so convert to look the same.
    linear = np.where(colours <= 0.04045, colours / 12.92, ((colours + 0.055) / 1.055) ** 2.4)
    rgba = np.concatenate([linear, np.ones((len(linear), 1))], 1)
    # TRELLIS.2 meshes are z-up; glTF is y-up.
    z_up_to_y_up = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float32)
    mesh = trimesh.Trimesh(vertices=vertices @ z_up_to_y_up.T, faces=faces,
                           vertex_colors=(rgba * 255).astype(np.uint8), process=False)
    mesh.export(path)
    return len(vertices), len(faces)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mesh")
    parser.add_argument("out")
    parser.add_argument("--faces", type=int, default=0, help="simplify to about this many faces (0 = keep all)")
    a = parser.parse_args()
    data = torch.load(a.mesh)
    v, f = save_glb(a.out, data["vertices"].float().numpy(), data["faces"].long().numpy(),
                    data["coords"].numpy(), data["attrs"].float().numpy(), a.faces)
    print(f"wrote {a.out}: {v} vertices, {f} faces")


if __name__ == "__main__":
    main()
