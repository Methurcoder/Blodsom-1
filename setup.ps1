# Sets up Blodsom-1 on Windows with an NVIDIA GPU:
# fetches trellis-mac and TRELLIS.2 at the tested commits, patches TRELLIS.2
# for the pure-PyTorch backends, and creates .venv with the dependencies
# (PyTorch for CUDA 12.8).
#
#   powershell -ExecutionPolicy Bypass -File setup.ps1
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$TrellisMacCommit = "d58628f4f5b9c3de8274cb110074154f4b31cef2"
$Trellis2Commit = "75fbf0183001ed9876c8dbb35de6b68552ee08bd"
$Utils3dCommit = "9a4eb15e4021b67b12c460c7057d642626897ec8"

function Invoke-Checked {
    & $args[0] $args[1..($args.Length - 1)]
    if ($LASTEXITCODE -ne 0) { throw "failed: $($args -join ' ')" }
}

function Get-Source($Dir, $Url, $Commit) {
    if (Test-Path "$Dir\.git") { Write-Host "  $Dir already fetched"; return }
    Invoke-Checked git init -q $Dir
    Invoke-Checked git -C $Dir remote add origin $Url
    Invoke-Checked git -C $Dir fetch -q --depth 1 origin $Commit
    Invoke-Checked git -C $Dir checkout -q FETCH_HEAD
}

Write-Host "Fetching trellis-mac and TRELLIS.2..."
Get-Source "third_party\trellis-mac" "https://github.com/shivampkumar/trellis-mac.git" $TrellisMacCommit
Get-Source "third_party\trellis-mac\TRELLIS.2" "https://github.com/microsoft/TRELLIS.2.git" $Trellis2Commit

if (-not (Test-Path ".venv")) {
    Write-Host "Creating .venv..."
    if (Get-Command py -ErrorAction SilentlyContinue) {
        Invoke-Checked py -3 -m venv .venv
    } else {
        Invoke-Checked python -m venv .venv
    }
}
$Python = ".venv\Scripts\python.exe"

Write-Host "Installing dependencies..."
Invoke-Checked $Python -m pip install -q --upgrade pip
Invoke-Checked $Python -m pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cu128
Invoke-Checked $Python -m pip install -q transformers accelerate huggingface_hub safetensors pillow numpy trimesh scipy `
    tqdm easydict kornia timm imageio opencv-python-headless psutil fast-simplification
Invoke-Checked $Python -m pip install -q "git+https://github.com/EasternJournalist/utils3d.git@$Utils3dCommit"

Write-Host "Patching TRELLIS.2..."
Push-Location "third_party\trellis-mac"
try {
    # -X utf8: the patch script reads and writes source files with the default encoding.
    Invoke-Checked "..\..\$Python" -X utf8 patches\mps_compat.py | Out-Null
} finally {
    Pop-Location
}

Write-Host ""
Write-Host "Done. Generate a model with:"
Write-Host "  .venv\Scripts\python generate.py your_image.png"
