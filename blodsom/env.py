"""Puts the patched TRELLIS.2 (from trellis-mac, fetched by setup.sh /
setup.ps1) and this folder on the import path, with the pure-PyTorch backends
Blodsom-1 runs on. Import this before anything from TRELLIS.2."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TRELLIS_MAC = os.environ.get("BLODSOM_TRELLIS_MAC", os.path.join(ROOT, "third_party", "trellis-mac"))

# Must be set before torch is imported: a few sparse ops fall back to the CPU
# on Apple Silicon.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("ATTN_BACKEND", "sdpa")
os.environ.setdefault("SPARSE_ATTN_BACKEND", "sdpa")
os.environ.setdefault("SPARSE_CONV_BACKEND", "none")

if not os.path.isdir(os.path.join(TRELLIS_MAC, "TRELLIS.2", "trellis2")):
    raise SystemExit(f"TRELLIS.2 not found under {TRELLIS_MAC}. Run setup.sh (macOS / Linux) or setup.ps1 (Windows) first.")
sys.path.insert(0, os.path.join(TRELLIS_MAC, "TRELLIS.2"))
# Appended, so a real o_voxel install would win over the stub package.
sys.path.append(os.path.join(TRELLIS_MAC, "stubs"))
sys.path.insert(0, HERE)
