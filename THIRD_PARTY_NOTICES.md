# Third-party notices

Blodsom-1 contains no third-party code or weights. `setup.sh` / `setup.ps1` fetch the projects
below into `third_party/`, and model weights are downloaded from Hugging Face at run time.
Their licenses apply to them.

| Component | Source | License |
|---|---|---|
| TRELLIS.2 (code) | https://github.com/microsoft/TRELLIS.2 | MIT, Copyright (c) Microsoft Corporation — [licenses/TRELLIS.2-MIT.txt](licenses/TRELLIS.2-MIT.txt) |
| TRELLIS.2-4B (weights) | https://huggingface.co/microsoft/TRELLIS.2-4B | MIT |
| trellis-mac (Apple Silicon port, patches and pure-PyTorch backends) | https://github.com/shivampkumar/trellis-mac | MIT — [licenses/trellis-mac-MIT.txt](licenses/trellis-mac-MIT.txt) |
| DINOv3 ViT-L/16 (weights, via timm) | https://huggingface.co/timm/vit_large_patch16_dinov3.lvd1689m | [DINOv3 License](https://ai.meta.com/resources/models-and-libraries/dinov3-license/) |
| utils3d | https://github.com/EasternJournalist/utils3d | MIT |

The files under `blodsom/` that change how TRELLIS.2 runs (`conv_lean.py`, `mesh_lean.py`) are
Blodsom-1's own code; `mesh_lean.py` loads trellis-mac's mesh extraction and replaces its lookup
loops at run time.

The README banner font is Cormorant Garamond (SIL Open Font License 1.1); the banner picture was
made by the author with an image generator.
