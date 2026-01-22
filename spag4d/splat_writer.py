# spag4d/splat_writer.py
"""
SPLAT format export for web-optimized Gaussian Splat viewing.

SPLAT is a compressed binary format ~8x smaller than PLY,
designed for web viewers.
"""

import numpy as np
import struct
from pathlib import Path
from typing import Optional
import torch


def save_splat(gaussians: dict, path: str) -> None:
    """
    Save Gaussians to compressed SPLAT format for web viewing.
    
    SPLAT format (per vertex, 32 bytes):
    - position: 3 × float16 (6 bytes)
    - scale: 3 × uint8 log-encoded (3 bytes)
    - color: 3 × uint8 RGB (3 bytes)
    - opacity: 1 × uint8 (1 byte)
    - quaternion: 4 × int8 normalized (4 bytes)
    - padding: 15 bytes (alignment)
    
    Total: 32 bytes/splat vs ~62 bytes/splat for PLY
    
    Args:
        gaussians: Dict with means, scales, quats, colors, opacities
        path: Output SPLAT file path
    """
    means = gaussians['means'].cpu().numpy()
    scales = gaussians['scales'].cpu().numpy()
    quats = gaussians['quats'].cpu().numpy()
    colors = gaussians['colors'].cpu().numpy()
    opacities = gaussians['opacities'].cpu().numpy()
    
    N = means.shape[0]
    if N == 0:
        raise ValueError("No valid Gaussians to save")
    
    # Note: No coordinate transform - matches PLY output frame (Y-up internal)
    
    # ─────────────────────────────────────────────────────────────────
    # Quantize
    # ─────────────────────────────────────────────────────────────────
    pos_f16 = means.astype(np.float16)
    
    # Log-scale to uint8 (clamp to reasonable range)
    log_scales = np.log(np.clip(scales, 1e-7, 1e3))
    scale_min, scale_max = -16, 8  # Reasonable log range
    scale_u8 = np.clip(
        ((log_scales - scale_min) / (scale_max - scale_min) * 255),
        0, 255
    ).astype(np.uint8)
    
    # Colors to uint8
    color_u8 = (colors * 255).clip(0, 255).astype(np.uint8)
    
    # Opacity to uint8 - use flatten() to ensure 1D array even for N=1
    opacity_u8 = (opacities.flatten() * 255).clip(0, 255).astype(np.uint8)
    
    # Quaternion to int8 (normalized, -127 to 127)
    quats_normalized = quats / (np.linalg.norm(quats, axis=-1, keepdims=True) + 1e-8)
    quat_i8 = (quats_normalized * 127).clip(-127, 127).astype(np.int8)
    
    # ─────────────────────────────────────────────────────────────────
    # Write binary
    # ─────────────────────────────────────────────────────────────────
    with open(path, 'wb') as f:
        # Header
        f.write(b'SPLAT')
        f.write(struct.pack('<I', N))  # Vertex count
        
        # Data (interleaved)
        for i in range(N):
            f.write(pos_f16[i].tobytes())       # 6 bytes
            f.write(scale_u8[i].tobytes())      # 3 bytes
            f.write(color_u8[i].tobytes())      # 3 bytes
            f.write(struct.pack('B', opacity_u8[i]))  # 1 byte
            f.write(quat_i8[i].tobytes())       # 4 bytes
            f.write(b'\x00' * 15)               # Padding to 32 bytes
    
    print(f"Saved {N:,} Gaussians to {path} (SPLAT format)")


def convert_ply_to_splat(ply_path: str, splat_path: str) -> None:
    """
    Convert existing PLY to SPLAT format.
    
    Args:
        ply_path: Input PLY file path
        splat_path: Output SPLAT file path
    """
    from .ply_writer import load_ply_gaussians
    
    gaussians = load_ply_gaussians(ply_path)
    
    # Note: PLY already has coordinates transformed, so we save without
    # re-transforming by using internal _save_splat_no_transform
    _save_splat_no_transform(gaussians, splat_path)


def _save_splat_no_transform(gaussians: dict, path: str) -> None:
    """
    Save SPLAT without coordinate transform (for pre-transformed data).
    """
    means = gaussians['means'].cpu().numpy()
    scales = gaussians['scales'].cpu().numpy()
    quats = gaussians['quats'].cpu().numpy()
    colors = gaussians['colors'].cpu().numpy()
    opacities = gaussians['opacities'].cpu().numpy()
    
    N = means.shape[0]
    if N == 0:
        raise ValueError("No valid Gaussians to save")
    
    pos_f16 = means.astype(np.float16)
    
    log_scales = np.log(np.clip(scales, 1e-7, 1e3))
    scale_min, scale_max = -16, 8
    scale_u8 = np.clip(
        ((log_scales - scale_min) / (scale_max - scale_min) * 255),
        0, 255
    ).astype(np.uint8)
    
    color_u8 = (colors * 255).clip(0, 255).astype(np.uint8)
    # Use flatten() to ensure 1D array even for N=1
    opacity_u8 = (opacities.flatten() * 255).clip(0, 255).astype(np.uint8)
    
    quats_normalized = quats / (np.linalg.norm(quats, axis=-1, keepdims=True) + 1e-8)
    quat_i8 = (quats_normalized * 127).clip(-127, 127).astype(np.int8)
    
    with open(path, 'wb') as f:
        f.write(b'SPLAT')
        f.write(struct.pack('<I', N))
        
        for i in range(N):
            f.write(pos_f16[i].tobytes())
            f.write(scale_u8[i].tobytes())
            f.write(color_u8[i].tobytes())
            f.write(struct.pack('B', opacity_u8[i]))
            f.write(quat_i8[i].tobytes())
            f.write(b'\x00' * 15)
    
    print(f"Saved {N:,} Gaussians to {path} (SPLAT format)")


def _quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Quaternion multiplication (XYZW order)."""
    if q1.ndim == 1:
        q1 = q1[np.newaxis, :]
    
    x1, y1, z1, w1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    x2, y2, z2, w2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    
    return np.stack([
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
    ], axis=-1)


def load_splat(path: str) -> dict:
    """
    Load Gaussians from SPLAT file.
    
    Args:
        path: Path to SPLAT file
    
    Returns:
        Dict with decoded means, scales, quats, colors, opacities
    """
    with open(path, 'rb') as f:
        # Header
        magic = f.read(5)
        if magic != b'SPLAT':
            raise ValueError(f"Invalid SPLAT file: expected SPLAT magic, got {magic}")
        
        N = struct.unpack('<I', f.read(4))[0]
        
        means = []
        scales = []
        colors = []
        opacities = []
        quats = []
        
        scale_min, scale_max = -16, 8
        
        for i in range(N):
            # Position (6 bytes float16)
            pos = np.frombuffer(f.read(6), dtype=np.float16).astype(np.float32)
            means.append(pos)
            
            # Scale (3 bytes uint8)
            scale_u8 = np.frombuffer(f.read(3), dtype=np.uint8)
            scale_log = scale_u8 / 255 * (scale_max - scale_min) + scale_min
            scales.append(np.exp(scale_log))
            
            # Color (3 bytes uint8)
            color = np.frombuffer(f.read(3), dtype=np.uint8) / 255
            colors.append(color)
            
            # Opacity (1 byte uint8)
            opacity = struct.unpack('B', f.read(1))[0] / 255
            opacities.append(opacity)
            
            # Quaternion (4 bytes int8)
            quat = np.frombuffer(f.read(4), dtype=np.int8) / 127
            quats.append(quat)
            
            # Skip padding
            f.read(15)
    
    return {
        'means': torch.from_numpy(np.stack(means)).float(),
        'scales': torch.from_numpy(np.stack(scales)).float(),
        'quats': torch.from_numpy(np.stack(quats)).float(),
        'colors': torch.from_numpy(np.stack(colors)).float(),
        'opacities': torch.from_numpy(np.array(opacities)[:, np.newaxis]).float(),
    }
