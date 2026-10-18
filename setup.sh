#!/usr/bin/env bash
# Sets up Blodsom-1 on macOS (Apple Silicon) or Linux (NVIDIA):
# fetches trellis-mac and TRELLIS.2 at the tested commits, patches TRELLIS.2
# for the pure-PyTorch backends, and creates .venv with the dependencies.
set -euo pipefail
cd "$(dirname "$0")"

TRELLIS_MAC_COMMIT=d58628f4f5b9c3de8274cb110074154f4b31cef2
TRELLIS2_COMMIT=75fbf0183001ed9876c8dbb35de6b68552ee08bd
UTILS3D_COMMIT=9a4eb15e4021b67b12c460c7057d642626897ec8

fetch() {  # dir url commit
    if [ -d "$1/.git" ]; then
        echo "  $1 already fetched"
        return
    fi
    git init -q "$1"
    git -C "$1" remote add origin "$2"
    git -C "$1" fetch -q --depth 1 origin "$3"
    git -C "$1" checkout -q FETCH_HEAD
}

echo "Fetching trellis-mac and TRELLIS.2..."
fetch third_party/trellis-mac https://github.com/shivampkumar/trellis-mac.git "$TRELLIS_MAC_COMMIT"
fetch third_party/trellis-mac/TRELLIS.2 https://github.com/microsoft/TRELLIS.2.git "$TRELLIS2_COMMIT"

if [ ! -d .venv ]; then
    echo "Creating .venv..."
    if command -v uv >/dev/null; then
        uv venv -q .venv --python 3.11
    else
        python3 -m venv .venv
    fi
fi
# shellcheck disable=SC1091
source .venv/bin/activate
if command -v uv >/dev/null; then PIP="uv pip install"; else python -m pip install -q --upgrade pip; PIP="python -m pip install"; fi

echo "Installing dependencies..."
$PIP torch torchvision transformers accelerate huggingface_hub safetensors pillow numpy trimesh scipy \
    tqdm easydict kornia timm imageio opencv-python-headless psutil fast-simplification
$PIP "git+https://github.com/EasternJournalist/utils3d.git@$UTILS3D_COMMIT"

echo "Patching TRELLIS.2..."
(cd third_party/trellis-mac && python patches/mps_compat.py >/dev/null)

echo
echo "Done. Generate a model with:"
echo "  source .venv/bin/activate"
echo "  python generate.py your_image.png"
