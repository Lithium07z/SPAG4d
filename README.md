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

## Quick Start

```bash
# Clone with DAP submodule
git clone --recurse-submodules https://github.com/cedarconnor/SPAG4d.git
cd SPAG4d
# If you already cloned, fetch submodules
git submodule update --init --recursive

# Create virtual environment
python -m venv .venv
.\.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# Install PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install SPAG-4D (Standard)
pip install -e ".[all]"
```

### ML-SHARP Installation (Optional)

To use the **Magic Fix (SHARP)** feature for enhanced detail, you need the `ml-sharp` package from Apple.

> **Note:** SHARP weights (~3GB) are downloaded automatically on first use.

#### Option A: Install from Local Clone (Recommended)

The `ml-sharp` submodule is included in this repo:

```bash
# Initialize submodule (if not already)
git submodule update --init --recursive

# Install ml-sharp from local directory
pip install .\ml-sharp

# Or install SPAG-4D with sharp extras
pip install -e ".[sharp]"
```

#### Option B: Install from GitHub

```bash
pip install "ml-sharp @ git+https://github.com/apple/ml-sharp.git"
```

#### Verify Installation

```bash
python -c "import sharp; print('ML-SHARP installed successfully')"
```

#### SHARP Projection Modes

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
