<p align="center"><img src="assets/banner.jpg" alt="Blodsom-1" width="100%"></p>

<p align="center"><b>TRELLIS.2 image-to-3D in 4 GB of VRAM — same weights, same result.</b><br>
<a href="README.ja.md">日本語</a></p>

Blodsom-1 runs Microsoft's [TRELLIS.2](https://github.com/microsoft/TRELLIS.2) (4B, image to 3D)
on small machines. The original asks for an NVIDIA GPU with at least 24 GB. Blodsom-1 finishes
the default 1024 cascade on a **laptop RTX 3050 with 4 GB of VRAM and 6.5 GB of RAM**, and on an
Apple Silicon Mac in about 6 GB.

Nothing about the model is changed: no quantisation, no pruning, no fewer steps. For the same image
and seed, the latents and the mesh are **bit-identical** to the original pipeline running on the same device.

## ⚠️ It is slow

Blodsom-1 trades time for memory. It is for running TRELLIS.2 **at all** on hardware that could not
before, not for running it fast.

| Machine | 1024 cascade, one image |
|---|---|
| NVIDIA H100 (original TRELLIS.2, from its README, 1024³) | about **17 s** |
| Apple M5, 32 GB (original as trellis-mac ships it, no Metal extensions, every model loaded) | about **330 s** |
| RTX 3050 Laptop 4 GB, 16 GB RAM (original as trellis-mac ships it, every model loaded, moved between CPU and GPU) | about **4,900 s** (82 min) |
| RTX 3050 Laptop 4 GB + 6.5 GB RAM (Blodsom-1) | about **174 s** (3 min) |
| Apple M5, 32 GB (Blodsom-1) | about **165–180 s** |

- Expect **around 3 minutes per model**, and more on slower GPUs.
- Every run loads each model from disk again (about 10 s on the RTX 3050).
- `--stream` (for GPUs under 4 GB) is slower again: about 300 s on the RTX 3050 in an earlier measurement.
- `--resident` helps a little when memory is plentiful: on the M5, 178 s → 157 s for the first image and 138 s from the second (same result), with up to 24 GB in use. Large models gain little (3.3 M vertices: 363 s → 362 s).
- A faster mode (fewer steps / distillation, results change slightly) is planned as a separate option.

## How it fits

| | Original | Blodsom-1 |
|---|---|---|
| Models in memory | all of them (13 GB of flow models alone) | one at a time, loaded straight to the GPU, dropped after use |
| Model loading | builds each model in float32 on the CPU first | builds an empty model and copies each saved tensor to the GPU in its own dtype |
| Sparse convolution (pure PyTorch) | Python loop over neighbours, int64 tables, gigabytes of scatter indices | sorted keys + `searchsorted`, int32, `index_add_` — same additions in the same order |
| Decoder MLP | all voxels at once | 262,144 voxels at a time |
| Mesh extraction (pure PyTorch) | Python dict lookups, millions of them | sorted-key array lookup |
| While a model runs | — | the next model's file is read into RAM (NVIDIA) |
| Image features (DINOv3) | gated Hugging Face download | the same weights from timm (no sign-up) |

Measured on the RTX 3050 Laptop (4 GB), 1024 cascade:

| Model in use | VRAM needed | time |
|---|---|---|
| sparse structure flow | 2.69 GB | 44 s |
| shape flow 512 → decoder → shape flow 1024 | 2.73 GB | 70 s |
| texture flow 1024 → decoder | 2.81 GB | 37 s |
| **whole run** | **4.0 GB device total** | **174 s** |

RAM working set peaked at 6.1 GB, and with a hard 6.5 GB cap the run took 173.5 s and gave the identical mesh.

## Install

You need Git and Python 3.10 or later. Model weights (about 15 GB) download on first run.

**macOS (Apple Silicon) / Linux (NVIDIA)**

```bash
git clone https://github.com/Methurcoder/Blodsom-1.git
cd Blodsom-1
bash setup.sh
```

**Windows (NVIDIA)**

```powershell
git clone https://github.com/Methurcoder/Blodsom-1.git
cd Blodsom-1
powershell -ExecutionPolicy Bypass -File setup.ps1
```

`setup` fetches [trellis-mac](https://github.com/shivampkumar/trellis-mac) and TRELLIS.2 at tested
commits into `third_party/`, applies trellis-mac's patches (pure-PyTorch attention, convolution and
mesh extraction), and installs the dependencies into `.venv`. No compiler and no CUDA toolkit are needed.

## Use

```bash
.venv/bin/python generate.py character.png              # macOS / Linux
.venv\Scripts\python generate.py character.png          # Windows
```

This writes `character.glb`. Several images can be given at once (`generate.py a.png b.png c.png`). Options:

| Option | |
|---|---|
| `--type 512 / 1024 / 1024_cascade / 1536_cascade` | pipeline type (default `1024_cascade`, as the original) |
| `--seed N` | random seed (default 42, as the original) |
| `--faces N` | simplify the GLB to about N faces (default: keep the full mesh, about 2 million vertices) |
| `--stream` | GPUs under 4 GB: send one transformer block at a time (much slower) |
| `--no-prefetch` | less RAM, a little slower |
| `--ram-limit GB` | Windows: cap physical memory to test a smaller machine |
| `--save-mesh` | also save the raw mesh and voxel attributes (`.pt`) |
| `--resident` | machines with memory to spare (about 24 GB): keep every model loaded and skip memory trimming. Same result |

**Limits**

- The image needs a **transparent background** (PNG with alpha). The original removes backgrounds
  with RMBG-2.0, which needs a sign-up and is non-commercial; Blodsom-1 does not include it. With a
  transparent image the original skips that step too, so results match.
- The GLB has **vertex colours**, not baked PBR textures: texture baking needs compiled CUDA / Metal
  extensions. The geometry and the colour of every voxel are the original's.

## Checking the "same result" claim

`tools/equivalence.py` runs the original pipeline (all models loaded, original loader) and
Blodsom-1 on the same image and seed and compares every weight tensor, the latents and the mesh bit
for bit. `tools/measure_cuda.py` prints the per-model memory table above.

Bit-identical means: on the same device with the same PyTorch. Different GPUs (or Mac versus NVIDIA)
round differently, as they do for the original.

Tested: Apple M5 (macOS, PyTorch 2.14) and RTX 3050 Laptop 4 GB (Windows, PyTorch 2.11 + CUDA 12.8).
Linux has not been tested.

## Licenses

- Blodsom-1's own code: MIT ([LICENSE](LICENSE)).
- TRELLIS.2 code and weights: MIT, Microsoft. trellis-mac: MIT. DINOv3 weights (downloaded from
  timm at run time): [DINOv3 License](https://ai.meta.com/resources/models-and-libraries/dinov3-license/).
  See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

Blodsom-1 is not affiliated with Microsoft or Meta.
