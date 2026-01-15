# SPAG-4D: 360° Panorama to Gaussian Splat


![SPAG-4D Screenshot](tests/Screenshot%202026-01-14%20214507.png)

Convert 360° equirectangular panoramas into viewable 3D Gaussian Splat files.

## Features

- **Native 360° Depth Estimation** - Uses DAP (Depth Any Panoramas) for equirectangular-aware depth
- **Metric Depth Output** - Real-world scale with manual adjustment option
- **Standard 3DGS PLY Output** - Compatible with gsplat, SuperSplat, SHARP viewers
- **Compressed SPLAT Format** - ~8x smaller for web delivery
- **Web UI** - Preview 360° input and 3D result
- **CLI** - Batch processing and automation

## Quick Start

```bash
# Clone with DAP submodule
git clone https://github.com/cedarconnor/SPAG4d.git
cd SPAG4d

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# Install PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install SPAG-4D and DAP dependencies
pip install -e ".[all]"
pip install torchmetrics mmengine safetensors einops opencv-python

# Run the web UI
.\start_spag4d.bat  # Windows
# Or manually: python -m spag4d.cli serve --port 7860
```

## Usage

### Web UI

Double-click `start_spag4d.bat` or run:
```bash
python -m spag4d.cli serve --port 7860
```

Open http://localhost:7860 in your browser.

### CLI

```bash
# Basic conversion
python -m spag4d.cli convert panorama.jpg output.ply

# With options
python -m spag4d.cli convert panorama.jpg output.ply \
    --stride 2 \
    --scale-factor 1.5 \
    --format both

# Batch processing
python -m spag4d.cli convert ./input/ ./output/ --batch
```

### Python API

```python
from spag4d import SPAG4D

converter = SPAG4D(device='cuda')

result = converter.convert(
    input_path='panorama.jpg',
    output_path='output.ply',
    stride=2,
    scale_factor=1.5
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

## Output Formats

- **PLY** - Full precision, viewer-compatible (gsplat, SuperSplat)
- **SPLAT** - Compressed for web (~8x smaller)

## Requirements

- Python 3.10+
- CUDA-capable GPU (8GB+ VRAM recommended)
- ffmpeg (for video processing)

### DAP Dependencies

The DAP model requires these additional packages:
- `torchmetrics` - For model metrics
- `mmengine` - For configuration handling  
- `safetensors` - For safe model loading
- `einops` - For tensor operations
- `opencv-python` - For image processing

## Project Structure

```
SPAG-4D/
├── spag4d/              # Main package
│   ├── dap_arch/        # DAP model wrapper
│   │   └── DAP/         # DAP repository (included)
│   ├── core.py          # Main orchestrator
│   ├── cli.py           # Command-line interface
│   └── ...
├── static/              # Web UI assets
├── api.py               # FastAPI backend
├── start_spag4d.bat     # Windows launcher
└── TestImage/           # Sample panoramas
```

## References

- [DAP - Depth Any Panoramas](https://github.com/Insta360-Research-Team/DAP)
- [3D Gaussian Splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)
- [SuperSplat Viewer](https://playcanvas.com/supersplat/editor)

## License

MIT
