# SPAG-4D Installation Guide

This guide is designed for Windows users who want the easiest, 1-click installation method.

## Prerequisites

- **Windows 10/11** (64-bit)
- **NVIDIA GPU** with 8GB+ VRAM (16GB+ recommended for high-res images)
- **NVIDIA Driver** 525+ with CUDA 12.1 support
- **A fast internet connection** (The initial download of models and libraries is several GBs)

> 💡 **No separate Python or CUDA installation is required!** SPAG-4D manages its own embedded Python environment so it won't conflict with anything else on your computer.

---

## Automatic Installation (Recommended)

1. **Extract** the downloaded `.zip` file to a permanent folder on your computer (e.g. `C:\SPAG-4D` or a folder on your Desktop).
2. Inside the folder, **double-click** the file named `install.bat`.
    - *Note:* If Windows SmartScreen blocks it, click "More Info" and then "Run anyway". 
3. A command prompt window will open. **Wait patiently.** The script will automatically:
    - Download an embedded Python distribution specifically for SPAG-4D.
    - Install PyTorch and CUDA libraries.
    - Install all required AI models, including DA3 and SHARP.
4. When it says "Installation Complete!", press any key to close the window.

---

## Quick Start

1. **Double-click** `run.bat`.
2. A command window will open and start the local server. Do not close this window while using the app.
3. Your default web browser will automatically open to **http://localhost:7860**.
4. The test image will load automatically. Click **Convert** to test your installation!

---

## First Run Models

During your first installation or conversion, SPAG-4D downloads several AI models:

| Model | Size | Purpose |
|-------|------|---------|
| PanDA (Panoramic Depth) | ~1.5 GB | 360° depth estimation (default for most scenes) |
| DA3 (Depth Anything V3) | ~1.5 GB | Latest general-purpose metric depth |
| ML-SHARP | ~3 GB | Apple's perceptual quality refinement engine |

These are cached locally. Subsequent runs are instant.

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| "PyTorch failed to load" | Ensure you have an NVIDIA GPU and update your NVIDIA driver to 525+. |
| Installation crashes or freezes | Close the window and run `install.bat` again. Sometimes network drops cause `pip` to fail. |
| CUDA out of memory | Use a higher Stride (4 or 8) or smaller input images. |
| Port 7860 in use | Edit `run.bat` and change the port number in the script. |
| Firewall prompt | Allow access — the server only runs on your local network. |

---

## Advanced Installation (Linux/Mac or Developers)

If you are not on Windows or prefer to manage your own virtual environment, please refer to the advanced installation instructions located in the main `README.md`.
