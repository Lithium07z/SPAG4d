"""
pipeline/utils/cubemap.py

Utility functions for converting between equirectangular projection (ERP)
and cubemap (6-face) representation.

Face index convention (consistent throughout this module):
    0: +X  (right)
    1: -X  (left)
    2: +Y  (top)
    3: -Y  (bottom)
    4: +Z  (front)   ← ERP centre column (θ = 0)
    5: -Z  (back)    ← ERP seam (θ = ±π)

Coordinate conventions:
    Longitude  θ ∈ [−π, π]    :  0 = scene front (+Z), positive = right (+X)
    Latitude   φ ∈ [−π/2, π/2]:  0 = equator, +π/2 = top pole (+Y)
    Face UV    ∈ [−1, 1]²     :  u positive = right, v positive = up
    Face image                 :  column ↔ u (left→right),
                                  row    ↔ v (top→bottom, i.e. v is negated)

Direction vectors for each face (face_uv_to_direction):
    Face 0 (+X): dir = (  1,  v, −u )
    Face 1 (−X): dir = ( −1,  v,  u )
    Face 2 (+Y): dir = (  u,  1, −v )
    Face 3 (−Y): dir = (  u, −1,  v )
    Face 4 (+Z): dir = (  u,  v,  1 )
    Face 5 (−Z): dir = ( −u,  v, −1 )

These formulas are mutually consistent: erp_to_cube followed by cube_to_erp
is an identity (up to bilinear interpolation rounding).
"""

from __future__ import annotations

import numpy as np
import cv2


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

# Precomputed per-face direction coefficients.
# Each row: (x_coeff, y_coeff, z_coeff) expressed as
#   dir_x = c[0][0]*1  + c[0][1]*uu + c[0][2]*vv   (1 = unit scalar)
# using sign flags rather than a 3×3 matrix for readability.
# Stored as: (x_base, x_u, x_v,  y_base, y_u, y_v,  z_base, z_u, z_v)
# where base/u/v are ±1 or 0 multiplied by ones/uu/vv respectively.
_FACE_DIR_COEFFS = [
    # face 0 +X:  dir = ( 1,  v, -u)   x=1·1, y=1·v, z=-1·u
    ( 1,  0,  0,   0,  0,  1,   0, -1,  0),
    # face 1 -X:  dir = (-1,  v,  u)   x=-1·1, y=1·v, z=1·u
    (-1,  0,  0,   0,  0,  1,   0,  1,  0),
    # face 2 +Y:  dir = ( u,  1, -v)   x=1·u, y=1·1, z=-1·v
    ( 0,  1,  0,   1,  0,  0,   0,  0, -1),
    # face 3 -Y:  dir = ( u, -1,  v)   x=1·u, y=-1·1, z=1·v
    ( 0,  1,  0,  -1,  0,  0,   0,  0,  1),
    # face 4 +Z:  dir = ( u,  v,  1)   x=1·u, y=1·v, z=1·1
    ( 0,  1,  0,   0,  0,  1,   1,  0,  0),
    # face 5 -Z:  dir = (-u,  v, -1)   x=-1·u, y=1·v, z=-1·1
    ( 0, -1,  0,   0,  0,  1,  -1,  0,  0),
]


def _face_uv_to_direction(
    face_idx: int,
    uu: np.ndarray,
    vv: np.ndarray,
) -> np.ndarray:
    """
    Convert face UV grids to unnormalised 3D direction vectors.

    Args:
        face_idx: Face index 0–5.
        uu: (face_w, face_w) float64 array, u ∈ [−1, 1] left→right.
        vv: (face_w, face_w) float64 array, v ∈ [−1, 1] bottom→top.

    Returns:
        dirs: (face_w, face_w, 3) float64 array (x, y, z), unnormalised.
    """
    c = _FACE_DIR_COEFFS[face_idx]
    ones = np.ones_like(uu)
    # x component: c[0]·1 + c[1]·uu + c[2]·vv
    dx = c[0] * ones + c[1] * uu + c[2] * vv
    # y component: c[3]·1 + c[4]·uu + c[5]·vv
    dy = c[3] * ones + c[4] * uu + c[5] * vv
    # z component: c[6]·1 + c[7]·uu + c[8]·vv
    dz = c[6] * ones + c[7] * uu + c[8] * vv
    return np.stack([dx, dy, dz], axis=-1)


def _direction_to_erp_coords(
    dirs_norm: np.ndarray,
    H: int,
    W: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Map normalised 3D direction vectors to ERP pixel coordinates.

    Args:
        dirs_norm: (…, 3) float64 array of unit vectors (x, y, z).
        H, W: ERP image dimensions.

    Returns:
        erp_x: (…) float array, pixel x ∈ [0, W].
        erp_y: (…) float array, pixel y ∈ [0, H].
    """
    x = dirs_norm[..., 0]
    y = dirs_norm[..., 1]
    z = dirs_norm[..., 2]

    theta = np.arctan2(x, z)                       # longitude ∈ [−π, π]
    phi   = np.arcsin(np.clip(y, -1.0, 1.0))       # latitude  ∈ [−π/2, π/2]

    erp_x = (theta / (2.0 * np.pi) + 0.5) * W     # ∈ [0, W]
    erp_y = (0.5 - phi / np.pi) * H                # ∈ [0, H]; top = 0
    return erp_x, erp_y


def _sample_erp(
    img: np.ndarray,
    map_x: np.ndarray,
    map_y: np.ndarray,
) -> np.ndarray:
    """
    Sample ERP image at floating-point pixel coordinates.

    Horizontal axis wraps (equirectangular seam continuity);
    vertical axis clamps (poles have no valid data beyond them).

    Args:
        img: (H, W) or (H, W, C) array, any dtype.
        map_x: Float array of x coordinates (columns).
        map_y: Float array of y coordinates (rows).

    Returns:
        Sampled array matching map_x.shape + optional channel dim, dtype float32.
    """
    H, W = img.shape[:2]
    # Wrap x horizontally; clamp y at poles
    map_x_w = (map_x % W).astype(np.float32)
    map_y_c = np.clip(map_y, 0, H - 1).astype(np.float32)

    src = img.astype(np.float32) if img.dtype != np.float32 else img
    return cv2.remap(
        src,
        map_x_w,
        map_y_c,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,  # only reached at poles after clamp
    )


def _direction_to_face_uv(
    x3d: np.ndarray,
    y3d: np.ndarray,
    z3d: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert 3D direction vectors to cubemap face index and UV coordinates.

    The dominant axis (largest absolute component) determines the face.
    Ties: X dominance > Y dominance > Z dominance (consistent with the
    forward-projection formulas in _face_uv_to_direction).

    Args:
        x3d, y3d, z3d: (H, W) float64 arrays (need not be normalised).

    Returns:
        face_map: (H, W) int32 array, values 0–5.
        uc: (H, W) float64 array, face u coordinate ∈ [−1, 1].
        vc: (H, W) float64 array, face v coordinate ∈ [−1, 1].
    """
    abs_x = np.abs(x3d)
    abs_y = np.abs(y3d)
    abs_z = np.abs(z3d)

    # Dominance masks (strict ordering: x > y > z for tie-breaking)
    x_dom = (abs_x >= abs_y) & (abs_x >= abs_z)
    y_dom = (~x_dom) & (abs_y >= abs_z)
    z_dom = ~(x_dom | y_dom)

    # Dominant absolute value (divisor for UV normalisation)
    dom = np.where(x_dom, abs_x, np.where(y_dom, abs_y, abs_z))
    dom = np.maximum(dom, 1e-9)  # guard against exact zero direction

    # Face index assignment
    face_map = np.select(
        [
            x_dom & (x3d > 0),   # +X
            x_dom & (x3d <= 0),  # -X
            y_dom & (y3d > 0),   # +Y
            y_dom & (y3d <= 0),  # -Y
            z_dom & (z3d > 0),   # +Z
        ],
        [0, 1, 2, 3, 4],
        default=5,               # -Z
    ).astype(np.int32)

    # u coordinate per face  (must match _face_uv_to_direction)
    #   +X: uc = -z/|x|   -X: uc = +z/|x|
    #   +Y: uc = +x/|y|   -Y: uc = +x/|y|
    #   +Z: uc = +x/|z|   -Z: uc = -x/|z|
    uc = np.select(
        [face_map == f for f in range(6)],
        [
            -z3d / dom,   # face 0 +X
             z3d / dom,   # face 1 -X
             x3d / dom,   # face 2 +Y
             x3d / dom,   # face 3 -Y
             x3d / dom,   # face 4 +Z
            -x3d / dom,   # face 5 -Z
        ],
    )

    # v coordinate per face
    #   +X: vc = +y/|x|   -X: vc = +y/|x|
    #   +Y: vc = -z/|y|   -Y: vc = +z/|y|
    #   +Z: vc = +y/|z|   -Z: vc = +y/|z|
    vc = np.select(
        [face_map == f for f in range(6)],
        [
             y3d / dom,   # face 0 +X
             y3d / dom,   # face 1 -X
            -z3d / dom,   # face 2 +Y
             z3d / dom,   # face 3 -Y
             y3d / dom,   # face 4 +Z
             y3d / dom,   # face 5 -Z
        ],
    )

    return face_map, uc, vc


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def erp_to_cube(erp_img: np.ndarray, face_w: int) -> np.ndarray:
    """
    Convert an equirectangular image to 6 cubemap faces.

    Face order: [+X, −X, +Y, −Y, +Z, −Z]  (indices 0–5).

    Args:
        erp_img: ERP image of shape ``(H, W, C)`` or ``(H, W)``.
            Any dtype is accepted; output is always ``float32``.
            ``W == 2 × H`` is expected for a valid 360° panorama but
            arbitrary aspect ratios are handled gracefully.
        face_w: Side length of each output face in pixels.

    Returns:
        cube: ``np.ndarray`` of shape ``(6, face_w, face_w, C)`` or
            ``(6, face_w, face_w)`` (matching the input dimensionality),
            dtype ``float32``.

    Example::

        cube = erp_to_cube(frame_rgb, face_w=512)
        # cube.shape == (6, 512, 512, 3)
        # cube.dtype == np.float32
    """
    is_2d = erp_img.ndim == 2
    if is_2d:
        erp_img = erp_img[:, :, np.newaxis]  # (H, W) → (H, W, 1)

    H, W, C = erp_img.shape

    # Pixel-centre aligned UV grid for one face.
    # half = (face_w − 1) / face_w so that linspace endpoints land on pixel
    # centres rather than the very edge of the ±1 square.
    # Example face_w=4: centres at ±0.75, ±0.25  ✓
    half = (face_w - 1.0) / face_w
    u_vals = np.linspace(-half,  half, face_w, dtype=np.float64)  # left→right
    v_vals = np.linspace( half, -half, face_w, dtype=np.float64)  # top→bottom (v inverted)
    uu, vv = np.meshgrid(u_vals, v_vals)  # both (face_w, face_w)

    result = np.zeros((6, face_w, face_w, C), dtype=np.float32)

    for fi in range(6):
        # UV → unnormalised 3D direction → normalise → ERP pixel coords
        dirs = _face_uv_to_direction(fi, uu, vv)              # (fw, fw, 3)
        norms = np.linalg.norm(dirs, axis=-1, keepdims=True)
        dirs_norm = dirs / np.maximum(norms, 1e-9)

        erp_x, erp_y = _direction_to_erp_coords(dirs_norm, H, W)

        result[fi] = _sample_erp(erp_img, erp_x, erp_y)      # (fw, fw, C)

    return result.squeeze(-1) if is_2d else result


def cube_to_erp(cube_faces: np.ndarray, h: int, w: int) -> np.ndarray:
    """
    Back-project 6 cubemap faces to an equirectangular image.

    For each ERP pixel the function determines which cubemap face it maps to,
    computes the face UV coordinates, and samples that face with bilinear
    interpolation.  This is the exact inverse of :func:`erp_to_cube` (up to
    interpolation rounding).

    Args:
        cube_faces: Cubemap array of shape ``(6, face_w, face_w, C)`` or
            ``(6, face_w, face_w)``.  Prefer ``float32`` input for quality.
        h: Output ERP height in pixels.
        w: Output ERP width in pixels.

    Returns:
        erp: ``np.ndarray`` of shape ``(h, w, C)`` or ``(h, w)``,
            dtype ``float32``.

    Example::

        reconstructed = cube_to_erp(cube, h=1024, w=2048)
        # reconstructed.shape == (1024, 2048, 3)
    """
    is_2d = cube_faces.ndim == 3                # (6, fw, fw) → treat as 1-channel
    if is_2d:
        cube_faces = cube_faces[:, :, :, np.newaxis]  # (6, fw, fw, 1)

    face_w = cube_faces.shape[2]
    C      = cube_faces.shape[3]

    # ERP pixel coordinates (pixel-centre: shift by +0.5 before normalising)
    px = np.arange(w, dtype=np.float64)
    py = np.arange(h, dtype=np.float64)
    xx, yy = np.meshgrid(px, py)   # (h, w)

    theta = ((xx + 0.5) / w - 0.5) * 2.0 * np.pi   # longitude ∈ [−π, π]
    phi   = (0.5 - (yy + 0.5) / h) * np.pi          # latitude  ∈ [π/2, −π/2]

    x3d = np.cos(phi) * np.sin(theta)
    y3d = np.sin(phi)
    z3d = np.cos(phi) * np.cos(theta)

    face_map, uc, vc = _direction_to_face_uv(x3d, y3d, z3d)

    # UV ∈ [−1, 1]  →  face pixel coordinates ∈ [0, face_w]
    face_px = np.clip((uc + 1.0) / 2.0 * face_w, 0.0, face_w - 1).astype(np.float32)
    face_py = np.clip((1.0 - vc) / 2.0 * face_w, 0.0, face_w - 1).astype(np.float32)

    result = np.zeros((h, w, C), dtype=np.float32)

    for fi in range(6):
        m = face_map == fi
        if not np.any(m):
            continue

        face_img = cube_faces[fi].astype(np.float32)  # (fw, fw, C)

        # cv2.remap samples face_img at every (face_px, face_py) position.
        # Pixels outside this face's region are garbage — we copy only the
        # pixels actually belonging to face fi via the boolean mask m.
        sampled = cv2.remap(
            face_img,
            face_px,     # (h, w) map — same shape as the output ERP
            face_py,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        result[m] = sampled[m]  # (h, w, C)[mask] = (h, w, C)[mask]

    return result.squeeze(-1) if is_2d else result


def get_face_masks(erp_mask: np.ndarray, face_w: int) -> np.ndarray:
    """
    Project a binary ERP mask onto 6 cubemap faces.

    Internally calls :func:`erp_to_cube` on the mask cast to ``float32``,
    then thresholds at 0.5 to restore binary values after bilinear
    interpolation blurs the edges.

    Args:
        erp_mask: Binary mask of shape ``(H, W)``, dtype ``uint8``
            (values 0 = background, 1 = foreground).
        face_w: Side length of each output face in pixels.

    Returns:
        face_masks: ``np.ndarray`` of shape ``(6, face_w, face_w)``,
            dtype ``uint8``, values 0 or 1.

    Example::

        face_masks = get_face_masks(masks[0], face_w=512)
        # face_masks.shape == (6, 512, 512)
        # face_masks.dtype == np.uint8
    """
    assert erp_mask.ndim == 2, (
        f"erp_mask must be 2-D (H, W), got shape {erp_mask.shape}"
    )

    # Project via float to preserve sub-pixel coverage, then re-binarise
    face_float = erp_to_cube(erp_mask.astype(np.float32), face_w)  # (6, fw, fw)
    face_masks = (face_float >= 0.5).astype(np.uint8)
    return face_masks
