# SPAG-4D: 360° Panorama to Gaussian Splat

![SPAG-4D Demo](assets/demo.gif)

Convert 360° equirectangular panoramas into viewable 3D Gaussian Splat files.

## Features

- **Native 360° Depth Estimation** - Uses DAP (Depth Any Panoramas) for equirectangular-aware depth
- **SHARP Refinement** - Optional high-frequency detail enhancement using Apple's ML-SHARP
- **360° Video Support** - Convert video sequences with frame extraction and trimming
- **Metric Depth Output** - Real-world scale with manual adjustment option
- **Standard 3DGS PLY Output** - Compatible with gsplat, SuperSplat, SHARP viewers
- **Compressed SPLAT Format** - ~8x smaller for web delivery
- **Web UI** - Preview 360° input (with Flat/Sphere modes) and 3D result
- **CLI** - Batch processing and automation

## Installation (Beginner-friendly)

This section assumes a clean install on a new machine and walks you through each step.

### 0) Before you start (one-time setup)

- Install Python 3.10+ from https://www.python.org/downloads/ (check "Add Python to PATH")
- Install Git from https://git-scm.com/downloads
- If you have an NVIDIA GPU, update your driver (recommended for speed)
- Make sure you have an internet connection and a few GB of free space (model weights download on first use)

### 1) Download SPAG-4D (includes DAP)

Open PowerShell and run:

```bash
git clone --recurse-submodules https://github.com/cedarconnor/SPAG4d.git
cd SPAG4d
```

If you already cloned the repo, run this once:

```bash
git submodule update --init --recursive
```

### 2) Create and activate a virtual environment

```bash
python -m venv .venv
.\.venv\Scripts\activate
```

On macOS/Linux:

```bash
source .venv/bin/activate
```

If PowerShell blocks activation, run:

```bash
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

Then activate the environment again.

### 3) Install PyTorch

If you have an NVIDIA GPU:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

If you do not have an NVIDIA GPU:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### 4) Install SPAG-4D

Recommended (Web UI + downloads, no SHARP):

```bash
pip install -e ".[server,download]"
```

Full install (includes Magic Fix / SHARP):

```bash
pip install -e ".[all]"
```

### 5) Optional: Magic Fix (SHARP)

If you installed the standard version and want SHARP later:

```bash
pip install -e ".[sharp]"
```

SHARP weights (~3GB) are downloaded automatically the first time you use Magic Fix.

Verify SHARP install (optional):

```bash
python -c "import sharp; print('ML-SHARP installed successfully')"
```

### Common installation issues

- `No module named 'spag4d.dap_arch.DAP.networks'`: DAP submodule is missing. Run `git submodule update --init --recursive`.
- `python` or `pip` not found: reinstall Python and check "Add Python to PATH".
- PowerShell blocks activation: run the `Set-ExecutionPolicy` command above, then re-activate the venv.

### SHARP Projection Modes

SHARP works by projecting the 360° image to perspective faces, running inference, and reprojecting:

| Mode | Faces | Quality | Speed | VRAM |
|------|-------|---------|-------|------|
| `cubemap` | 6 | Good | Fast | ~6GB |
| `icosahedral` | 20 | Better | ~3x slower | ~12GB |

Select the projection mode in the UI (dropdown next to "Magic Fix") or via CLI:
```bash
python -m spag4d.cli convert input.jpg out.ply --sharp-refine --sharp-projection icosahedral
```

## Usage

### Web UI

1. Start the server:
   ```bash
   .\start_spag4d.bat
   # Or manually: python -m spag4d.cli serve --port 7860
   ```
2. Open http://localhost:7860 in your browser.
3. Upload a panoramic image or video.
4. **SHARP Refinement**: Check the **"Magic Fix (SHARP)"** box to enable detail enhancement.
   - Adjust **"Detail Blend"** slider to control the strength.
5. Click **Convert** and view the result in the 3D viewer.

- **Input Preview**: Toggle between 360° Sphere and Flat Equirectangular views.
- **Splat Viewer**: Use WASD + Mouse to fly, Scroll to zoom. "Outside" center view.

### CLI

```bash
# Basic conversion
python -m spag4d.cli convert panorama.jpg output.ply

# With SHARP refinement
python -m spag4d.cli convert panorama.jpg output.ply \
    --sharp-refine \
    --scale-blend 0.5 \
    --format splat

# Batch processing
python -m spag4d.cli convert ./input/ ./output/ --batch

# Video conversion (automatic frame extraction)
python -m spag4d.cli convert_video input.mp4 output_dir --fps 10 --start 0.0 --duration 5.0
```

### Python API

```python
from spag4d import SPAG4D

# Initialize with SHARP support
converter = SPAG4D(
    device='cuda',
    use_sharp_refinement=True
)

result = converter.convert(
    input_path='panorama.jpg',
    output_path='output.ply',
    stride=2,
    scale_factor=1.5,
    # Enable SHARP for this conversion
    use_sharp_refinement=True,
    scale_blend=0.5
)

print(f"Generated {result.splat_count:,} Gaussians")
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `stride` | 2 | Spatial downsampling (1, 2, 4, 8) |
| `scale_factor` | 1.5 | Gaussian scale multiplier |
| `thickness` | 0.1 | Radial thickness ratio |
| `global_scale` | 1.0 | Depth scale correction |
| `depth_min` | 0.1 | Minimum depth (meters) |
| `depth_max` | 100.0 | Maximum depth (meters) |
| `sky_threshold` | 80.0 | Filter points beyond this distance |

### SHARP Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `sharp_refine` | False | Enable SHARP refinement |
| `scale_blend` | 0.5 | Blend ratio for geometric vs learned scales (0=Geo, 1=Learned) |
| `opacity_blend` | 1.0 | Blend ratio for opacities |

## Requirements

- Python 3.10+
- CUDA-capable GPU (8GB+ VRAM recommended)
- ffmpeg (for video processing)

## Project Structure

```
SPAG-4D/
├── spag4d/              # Main package
│   ├── dap_arch/        # DAP model wrapper
│   ├── core.py          # Main orchestrator
│   ├── sharp_refiner.py # SHARP integration
│   └── ...
├── static/              # Web UI assets
├── api.py               # FastAPI backend
├── ml-sharp/            # (Optional) Local ml-sharp dependency
└── TestImage/           # Sample panoramas
```

## References

- [DAP - Depth Any Panoramas](https://github.com/Insta360-Research-Team/DAP)
- [ML-SHARP](https://github.com/apple/ml-sharp)
- [3D Gaussian Splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)
