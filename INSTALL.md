# SPAG-4D Installation Guide

## Prerequisites

- **Windows 10/11** (64-bit)
- **NVIDIA GPU** with 8GB+ VRAM (16GB+ recommended for high-res images)
- **NVIDIA Driver** 525+ with CUDA 12.1 support
- No separate Python or CUDA install needed — everything is bundled

### Optional: Depth Anything V3
To use the new DA3 model, you must install it manually in the environment:
```cmd
.venv\Scripts\pip install -e "git+https://github.com/ByteDance-Seed/depth-anything-3.git#egg=depth_anything_3"
```

## Quick Start

1. **Extract** this archive to any folder (e.g. `C:\SPAG-4D`)
2. **Double-click** `SPAG4D.bat`
3. The web UI opens at **http://localhost:7860**
4. Upload a 2:1 equirectangular panorama and click **Convert**

## First Run

On the first conversion, SPAG-4D will download two AI models:

| Model | Size | Purpose |
|-------|------|---------|
| PanDA (Panoramic Depth Anything) | ~1.5 GB | 360° depth estimation (default) |
| DA3 (Depth Anything V3) | ~1.5 GB | General-purpose metric depth (requires install) |
| ML-SHARP | ~3 GB | Perceptual quality refinement |

Optionally, you can use the legacy DAP model instead:

| Model | Size | Purpose |
|-------|------|---------|
| DAP (Depth Any Panoramas) | ~1.5 GB | 360° depth estimation (legacy, use `--depth-model dap`) |

These are cached to `~/.cache/spag4d` and `~/.cache/sharp` respectively. Subsequent runs are instant.

## Troubleshooting

| Issue | Fix |
|-------|-----|
| "PyTorch failed to load" | Update NVIDIA driver to 525+ |
| CUDA out of memory | Use higher Stride (4 or 8) or smaller input images |
| Port 7860 in use | Edit `SPAG4D.bat` and change the port number |
| Firewall prompt | Allow access — the server runs locally only |

## Command Line Usage

Activate the bundled environment and use the CLI directly:

```
.venv\Scripts\activate
python -m spag4d.cli convert input.jpg output.ply --stride 2
python -m spag4d.cli serve --port 8080
```

## License

See README.md for license information and credits.
