# spag4d/gaussian_converter.py
"""
Convert equirectangular panorama + depth to 3D Gaussians.

This implements the SPAG (Spherical Panorama to Gaussians) algorithm
with latitude-aware anisotropic scaling.
"""

import torch
import math
from typing import Optional

from .spherical_grid import SphericalGrid, rotation_matrix_to_quaternion


def equirect_to_gaussians(
    image: torch.Tensor,
    depth: torch.Tensor,
    grid: SphericalGrid,
    scale_factor: float = 1.5,
    thickness_ratio: float = 0.1,
    depth_min: float = 0.1,
    depth_max: float = 100.0,
    pole_rows: int = 3,
    default_opacity: float = 0.95,
    validity_mask: Optional[torch.Tensor] = None
) -> dict:
    """
    Convert equirectangular panorama with depth to 3D Gaussians.
    
    Args:
        image: RGB image [H, W, 3] uint8 or float [0,1]
        depth: Depth map [H, W] in meters
        grid: Precomputed SphericalGrid
        scale_factor: Gaussian scale multiplier (larger = more overlap)
        thickness_ratio: Radial thickness as fraction of tangent scales
        depth_min: Minimum valid depth
        depth_max: Maximum valid depth
        pole_rows: Rows to exclude at top/bottom poles
        default_opacity: Opacity for all Gaussians
        validity_mask: Optional [H, W] mask from depth model (0-1 or bool)
    
    Returns:
        Dict with:
            means: [N, 3] 3D positions (Y-up frame)
            scales: [N, 3] Gaussian scales (azimuth, elevation, normal)
            quats: [N, 4] Quaternions in XYZW order
            colors: [N, 3] RGB colors [0, 1]
            opacities: [N, 1] Opacity values
    """
    device = grid.device
    H, W = grid.original_H, grid.original_W
    stride = grid.stride
    
    # Convert image to float if needed
    if image.dtype == torch.uint8:
        colors = image.float() / 255.0
    else:
        colors = image.clone()
    
    # Downsample image to match grid
    H_grid, W_grid = grid.theta.shape
    if colors.shape[0] != H_grid or colors.shape[1] != W_grid:
        # Use strided sampling (not interpolation for speed)
        colors = colors[stride//2::stride, stride//2::stride]
    
    # Ensure depth matches grid dimensions
    if depth.shape[0] != H_grid or depth.shape[1] != W_grid:
        depth = depth[stride//2::stride, stride//2::stride]
    
    # Downsample validity_mask to match grid if provided
    if validity_mask is not None:
        if validity_mask.shape[0] != H_grid or validity_mask.shape[1] != W_grid:
            validity_mask = validity_mask[stride//2::stride, stride//2::stride]
    
    # ─────────────────────────────────────────────────────────────────
    # Validity Mask
    # ─────────────────────────────────────────────────────────────────
    valid_mask = (depth > depth_min) & (depth < depth_max)
    
    # Apply learned validity mask if provided
    if validity_mask is not None:
        # Convert to bool if float (threshold at 0.5)
        if validity_mask.dtype == torch.float32 or validity_mask.dtype == torch.float16:
            valid_mask = valid_mask & (validity_mask > 0.5)
        else:
            valid_mask = valid_mask & validity_mask.bool()
    
    # Exclude poles
    if pole_rows > 0:
        pole_mask = torch.ones_like(valid_mask, dtype=torch.bool)
        pole_mask[:pole_rows, :] = False
        pole_mask[-pole_rows:, :] = False
        valid_mask = valid_mask & pole_mask
    
    # ─────────────────────────────────────────────────────────────────
    # 3D Positions: P = depth * r̂
    # ─────────────────────────────────────────────────────────────────
    means = depth.unsqueeze(-1) * grid.rhat
    
    # ─────────────────────────────────────────────────────────────────
    # Anisotropic Scale (latitude-aware)
    # ─────────────────────────────────────────────────────────────────
    # Angular extent per pixel
    delta_theta = 2 * math.pi / W  # Horizontal angular width
    delta_phi = math.pi / H        # Vertical angular height
    
    # sin(φ) compensates for ERP distortion at poles
    sin_phi = torch.sin(grid.phi).clamp(min=0.01)  # Avoid zero at poles
    
    # Tangent plane footprint at distance d
    # s_azimuth: horizontal extent (varies with latitude)
    s_azimuth = scale_factor * depth * sin_phi * delta_theta * stride
    
    # s_elevation: vertical extent (constant with latitude in ERP)
    s_elevation = scale_factor * depth * delta_phi * stride
    
    # s_normal: thin in radial direction
    s_normal = torch.minimum(s_azimuth, s_elevation) * thickness_ratio
    
    # Stack: [azimuth, elevation, normal]
    scales = torch.stack([s_azimuth, s_elevation, s_normal], dim=-1)
    
    # ─────────────────────────────────────────────────────────────────
    # Rotation Matrix → Quaternion
    # ─────────────────────────────────────────────────────────────────
    # Build rotation matrix R where columns are [right, up, normal]
    # Normal points inward (toward camera), same as -rhat
    normal = -grid.rhat
    right = grid.tangent_right
    up = grid.tangent_up
    
    # R = [right | up | normal] as column vectors
    R = torch.stack([right, up, normal], dim=-1)  # [H, W, 3, 3]
    
    # Convert to quaternion
    quats = rotation_matrix_to_quaternion(R)  # [H, W, 4] in XYZW
    
    # ─────────────────────────────────────────────────────────────────
    # Flatten and filter by validity mask
    # ─────────────────────────────────────────────────────────────────
    valid_flat = valid_mask.flatten()
    
    means_flat = means.reshape(-1, 3)[valid_flat]
    scales_flat = scales.reshape(-1, 3)[valid_flat]
    quats_flat = quats.reshape(-1, 4)[valid_flat]
    colors_flat = colors.reshape(-1, 3)[valid_flat]
    
    N = means_flat.shape[0]
    opacities_flat = torch.full((N, 1), default_opacity, device=device)
    
    return {
        'means': means_flat,
        'scales': scales_flat,
        'quats': quats_flat,
        'colors': colors_flat,
        'opacities': opacities_flat
    }


def filter_sky(
    gaussians: dict,
    sky_threshold: float = 80.0
) -> dict:
    """
    Remove Gaussians beyond sky threshold distance.
    
    Args:
        gaussians: Gaussian dict from equirect_to_gaussians
        sky_threshold: Max distance in meters
    
    Returns:
        Filtered Gaussian dict
    """
    distances = gaussians['means'].norm(dim=-1)
    valid = distances < sky_threshold
    
    return {
        'means': gaussians['means'][valid],
        'scales': gaussians['scales'][valid],
        'quats': gaussians['quats'][valid],
        'colors': gaussians['colors'][valid],
        'opacities': gaussians['opacities'][valid]
    }


def equirect_to_gaussians_refined(
    image: torch.Tensor,
    depth: torch.Tensor,
    grid: SphericalGrid,
    refined_attrs: Optional[object] = None,  # Avoid circular import, pass RefinedAttributes object
    scale_factor: float = 1.5,
    thickness_ratio: float = 0.1,
    depth_min: float = 0.1,
    depth_max: float = 100.0,
    pole_rows: int = 3,
    default_opacity: float = 0.95,
    validity_mask: Optional[torch.Tensor] = None,
    scale_blend: float = 0.8,
    opacity_blend: float = 1.0,
    color_blend: float = 0.5,
) -> dict:
    """
    Convert ERP panorama to Gaussians with optional SHARP refinements.

    When refined_attrs is provided:
    - Opacities are taken from SHARP (blended by opacity_blend)
    - Scales are blended between geometric and SHARP (by scale_blend)
    - Colors can optionally be refined

    Positions and rotations always come from geometric computation
    to maintain 360° consistency.
    """
    import torch.nn.functional as F

    # 1. Compute base Gaussians
    base_gaussians = equirect_to_gaussians(
        image, depth, grid,
        scale_factor, thickness_ratio,
        depth_min, depth_max, pole_rows,
        default_opacity, validity_mask
    )

    if refined_attrs is None:
        return base_gaussians

    # 2. Apply SHARP refinements
    # Note: Attributes in base_gaussians are flat [N, ...]
    # We need to sample refined maps at the valid positions

    H_grid, W_grid = grid.theta.shape
    
    # Re-compute validity mask to find indices (must match equirect_to_gaussians exactly)
    # Ideally equirect_to_gaussians should return indices, but we recompute for now.
    
    # IMPORTANT: Downsample depth to match grid resolution BEFORE computing mask
    # This matches equirect_to_gaussians behavior: "depth = depth[stride//2::stride, stride//2::stride]"
    stride = grid.stride
    depth_downsampled = depth
    if depth.shape[0] != H_grid or depth.shape[1] != W_grid:
        depth_downsampled = depth[stride//2::stride, stride//2::stride]
        
    valid_mask = (depth_downsampled > depth_min) & (depth_downsampled < depth_max)
    
    if validity_mask is not None:
        # Downsample passed validity mask too
        mask_downsampled = validity_mask
        if validity_mask.shape[0] != H_grid or validity_mask.shape[1] != W_grid:
            mask_downsampled = validity_mask[stride//2::stride, stride//2::stride]

        if mask_downsampled.dtype == torch.float32 or mask_downsampled.dtype == torch.float16:
            valid_mask = valid_mask & (mask_downsampled > 0.5)
        else:
            valid_mask = valid_mask & mask_downsampled.bool()
    
    if pole_rows > 0:
        pole_mask = torch.ones_like(valid_mask, dtype=torch.bool)
        pole_mask[:pole_rows, :] = False
        pole_mask[-pole_rows:, :] = False
        valid_mask = valid_mask & pole_mask

    # Ensure mask matches grid downsampling if simple sub-sampling used in equirect_to_gaussians
    # In equirect_to_gaussians:
    # if depth.shape != grid.shape: depth = downsampled
    # But valid_mask calculation uses `depth`.
    # `depth` passed to this function SHOULD match grid steps if called correctly from core.py
    # But let's be safe and assume `depth` is the one used for `equirect_to_gaussians`.
    
    # We essentially need the flattened mask indices
    valid_flat = valid_mask.flatten()

    # Helper to sample and flatten map
    def sample_map(feature_map: torch.Tensor, channels: int) -> torch.Tensor:
        # feature_map: [H, W] or [H, W, C]
        # Resize to grid size if needed
        if feature_map.shape[0] != H_grid or feature_map.shape[1] != W_grid:
            # Add dims for interpolate: [1, C, H, W]
            if feature_map.dim() == 2:
                inp = feature_map.unsqueeze(0).unsqueeze(0)
            else:
                inp = feature_map.permute(2, 0, 1).unsqueeze(0)
                
            out = F.interpolate(inp, size=(H_grid, W_grid), mode='bilinear', align_corners=True)
            
            if feature_map.dim() == 2:
                resized = out.squeeze()
            else:
                resized = out.squeeze(0).permute(1, 2, 0)
        else:
            resized = feature_map
            
        if channels == 1:
             return resized.flatten()[valid_flat].unsqueeze(-1)
        else:
             return resized.reshape(-1, channels)[valid_flat]

    # Refine Opacity
    if refined_attrs.opacities is not None and opacity_blend > 0:
        ref_opacities_flat = sample_map(refined_attrs.opacities, 1)
        # Blend
        base_gaussians['opacities'] = (
            (1 - opacity_blend) * base_gaussians['opacities'] +
            opacity_blend * ref_opacities_flat
        ).clamp(0.01, 0.99)

    # Refine Scales
    if refined_attrs.scales is not None and scale_blend > 0:
        ref_scales_flat = sample_map(refined_attrs.scales, 3)
        
        # SHARP scales are in perspective space.
        # We treat them as a refinement multiplier for our geometric scales.
        # Normalize by mean to get relative variation
        scale_mult = ref_scales_flat / (ref_scales_flat.mean() + 1e-6)
        scale_mult = scale_mult.clamp(0.5, 2.0)
        
        base_gaussians['scales'] = (
            (1 - scale_blend) * base_gaussians['scales'] +
            scale_blend * base_gaussians['scales'] * scale_mult
        )

    # Refine Colors (blend with source to preserve fidelity)
    if refined_attrs.colors is not None and color_blend > 0:
        ref_colors_flat = sample_map(refined_attrs.colors, 3).clamp(0, 1)
        base_gaussians['colors'] = (
            (1 - color_blend) * base_gaussians['colors'] +
            color_blend * ref_colors_flat
        )

    return base_gaussians
