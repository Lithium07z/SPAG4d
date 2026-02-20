"""
pipeline/depth.py

Conservative background depth completion for foreground-masked regions.

Problem:
    Depth models (PanDA, DAP) estimate depth from the *original* frame which
    contains foreground objects.  The depth values under those objects are
    unreliable — the model may leak foreground depth into the background or
    produce noisy estimates.  When the inpainted background video is used as
    input, the model can estimate cleaner background depth, but residual
    artefacts near mask edges still occur.

Strategy (conservative push):
    1. Run the chosen depth model on the inpainted background frame to get a
       full-frame depth estimate (``_estimate_full_depth``).
    2. For each masked region, compute the median depth of the background
       pixels just outside the mask boundary (border ring).
    3. Replace all masked depth values with  ``border_median + delta``.
       The positive ``delta`` (from ``configs/default.yaml``) ensures the
       synthesised background depth is slightly *farther* than the measured
       boundary, preventing z-fighting when the foreground layer is placed
       on top during composition (``pipeline.composer``).

Import paths for spag4d depth models (verified from spag4d/ source):
    from spag4d.panda_model import PanDAModel   # preferred — CVPR 2025 360° model
    from spag4d.dap_model   import DAPModel     # legacy fallback
    from spag4d.da3_model   import DA3Model     # Depth Anything V3

    These classes are NOT re-exported from ``spag4d.__init__``; always import
    from the submodule directly.  All three share the same interface:
        model = XxxModel.load(model_path=None, device=device)
        depth_tensor, mask = model.predict(image_tensor)   # image: [H,W,3] uint8 tensor
        # depth_tensor: torch.Tensor [H, W], float32, metric metres (PanDA: pseudo-metric)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import yaml


logger = logging.getLogger(__name__)

# Module-level model cache: keyed by (model_name, device) to avoid
# reloading the depth model for every frame in complete_depth().
# Populated lazily inside _estimate_full_depth on first call.
_DEPTH_MODEL_CACHE: dict[tuple[str, str], Any] = {}


def _estimate_full_depth(
    frame: np.ndarray,
    device: str,
) -> np.ndarray:
    """
    Estimate metric depth for a single equirectangular frame using PanDA.

    The depth model is loaded once and cached at module level in
    ``_DEPTH_MODEL_CACHE[(model_name, device)]`` to avoid the 5–30 s
    reload penalty on every frame call.  The model name is read from
    ``configs/default.yaml`` under the ``depth.model`` key (default: "panda").

    Import paths used internally (verified from spag4d source):
        ``from spag4d.panda_model import PanDAModel``  (depth.model == "panda")
        ``from spag4d.dap_model   import DAPModel``    (depth.model == "dap")
        ``from spag4d.da3_model   import DA3Model``    (depth.model == "da3")

    All three expose the same interface::

        model = XxxModel.load(model_path=None, device=torch.device(device))
        depth_t, _ = model.predict(image_tensor)   # [H, W, 3] uint8 torch.Tensor
        depth_np = depth_t.cpu().numpy()           # (H, W) float32

    Args:
        frame: RGB image as ``np.ndarray`` of shape ``(H, W, 3)``, dtype ``uint8``.
            Must be an equirectangular (2:1 aspect ratio) frame from the
            *inpainted background* video (output of
            :func:`pipeline.inpainting.cubemap_inpaint_video`).
        device: Torch device string, e.g. ``"cuda"`` or ``"cpu"``.

    Returns:
        depth: ``np.ndarray`` of shape ``(H, W)``, dtype ``float32``, in metres
            (or pseudo-metric for PanDA).  All values positive.
    """
    config_path = Path(__file__).parent.parent / "configs" / "default.yaml"
    if not config_path.exists():
        raise FileNotFoundError(
            f"configs/default.yaml not found at {config_path}. "
            "Create it or copy from the repository root."
        )
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    model_name = cfg.get("depth", {}).get("model", "dap")

    cache_key = (model_name, device)
    model = _DEPTH_MODEL_CACHE.get(cache_key)
    if model is None:
        if model_name == "panda":
            from spag4d.panda_model import PanDAModel

            model = PanDAModel.load(model_path=None, device=torch.device(device))
        elif model_name == "da3":
            from spag4d.da3_model import DA3Model

            model = DA3Model.load(device=torch.device(device))
        else:  # "dap" or unknown — fall back to DAPModel
            if model_name not in ("dap",):
                logger.warning(
                    "Unknown depth.model=%s; falling back to 'dap'", model_name
                )
            from spag4d.dap_model import DAPModel

            model = DAPModel.load(model_path=None, device=torch.device(device))
        _DEPTH_MODEL_CACHE[cache_key] = model

    assert frame.ndim == 3 and frame.shape[2] == 3, (
        f"frame must be (H, W, 3), got {frame.shape}"
    )

    image_tensor = torch.from_numpy(frame).to(model.device)
    with torch.inference_mode():
        depth_t, _ = model.predict(image_tensor)

    depth_np = depth_t.detach().float().cpu().numpy()
    assert depth_np.ndim == 2, f"Depth must be 2D, got {depth_np.shape}"
    return depth_np


def _conservative_push(
    depth: np.ndarray,
    mask: np.ndarray,
    delta: float,
) -> np.ndarray:
    """
    Replace foreground-masked depth values with a conservative background estimate.

    For each connected masked region the function:
      1. Dilates ``mask`` by 1 pixel to form a border ring of background pixels
         immediately adjacent to the foreground boundary.
      2. Computes the median depth of those border-ring pixels from the input
         ``depth`` map (which was estimated on the inpainted background frame and
         is therefore reliable in background-only regions).
      3. Fills the entire masked region with ``border_median + delta``.

    This "push" places the background surface slightly *behind* the measured
    boundary depth, preventing z-fighting when the foreground Gaussian layer
    is composited on top.

    ``delta`` should be read from ``configs/default.yaml``::

        depth:
            delta: 0.5   # metres

    Args:
        depth: Full-frame depth map, shape ``(H, W)``, dtype ``float32``,
            in metres.  Estimated on the *inpainted background* frame so
            values under the mask are already background-like but may still
            contain mild artefacts.
        mask: Binary foreground mask, shape ``(H, W)``, dtype ``uint8``
            (values 0 = background, 1 = foreground), as produced by
            :func:`pipeline.segmentation.generate_masks`.
        delta: Positive offset in metres added to the border median.
            Loaded from ``configs/default.yaml`` key ``depth.delta``.
            Typical value: 0.3–1.0 m.

    Returns:
        depth_out: ``np.ndarray`` of shape ``(H, W)``, dtype ``float32``.
            Identical to ``depth`` outside the mask; inside the mask all
            values are replaced with ``border_ring_median(depth) + delta``.
    """
    assert depth.ndim == 2, f"depth must be 2D, got {depth.shape}"
    assert mask.ndim == 2, f"mask must be 2D, got {mask.shape}"
    assert depth.shape == mask.shape, (
        f"depth/mask shape mismatch: {depth.shape} vs {mask.shape}"
    )

    mask_bool = mask.astype(bool)
    background_vals = depth[~mask_bool]
    assert background_vals.size > 0, "No background pixels to compute median depth."

    median_val = float(np.median(background_vals))
    fill_val = median_val + float(delta)

    depth_out = depth.copy()
    depth_out[mask_bool] = fill_val
    depth_out = np.clip(depth_out, 0.0, 100.0)
    return depth_out.astype(np.float32)


def complete_depth(
    video_path: str,
    masks: dict[int, np.ndarray],
    device: str = "cuda",
) -> dict[int, np.ndarray]:
    """
    Produce per-frame completed background depth maps for a 360° video.

    Orchestrates the full depth-completion pipeline:
      1. Reads ``depth.model`` and ``depth.delta`` from
         ``configs/default.yaml``.
      2. Opens ``video_path`` (the *inpainted background* video produced by
         :func:`pipeline.inpainting.cubemap_inpaint_video`) frame by frame.
      3. Calls :func:`_estimate_full_depth` for each frame using the configured
         model.  The model is cached after the first call so loading happens
         only once.
      4. Calls :func:`_conservative_push` to replace masked-region depth
         values with a stable background estimate.
      5. Returns a dict of completed depth maps, one per frame.

    Args:
        video_path: Path to the inpainted background video file (MP4 or similar),
            as returned by :func:`pipeline.inpainting.cubemap_inpaint_video`.
            Each frame is an equirectangular image with foreground regions
            filled by LaMa inpainting.
        masks: Per-frame binary foreground masks produced by
            :func:`pipeline.segmentation.generate_masks`.
            Keys are 0-based frame indices; values are ``(H, W)`` arrays of
            dtype ``uint8`` (0 = background, 1 = foreground).
        device: Torch device string, e.g. ``"cuda"`` or ``"cpu"``.

    Returns:
        A dict mapping frame index (0-based ``int``) to a completed depth array
        of shape ``(H, W)`` with dtype ``float32``, in metres (pseudo-metric for
        PanDA).  Within the foreground mask the depth equals
        ``background_border_median + delta``; outside the mask the depth is the
        raw model estimate.

    Raises:
        AssertionError: If ``video_path`` cannot be opened, or if a frame index
            present in the video has no corresponding entry in ``masks``.
        FileNotFoundError: If ``configs/default.yaml`` does not exist.

    Example::

        from pipeline.segmentation import generate_masks
        from pipeline.inpainting   import cubemap_inpaint_video
        from pipeline.depth        import complete_depth

        masks    = generate_masks("scene.mp4", "/weights/sam3.pth", "cuda")
        bg_video = cubemap_inpaint_video("scene.mp4", masks, "/weights/lama", "cuda")
        bg_depth = complete_depth(bg_video, masks, device="cuda")
        # bg_depth[0].shape == (1024, 2048), dtype float32
    """
    # ── Load config ────────────────────────────────────────────────────────────
    config_path = Path(__file__).parent.parent / "configs" / "default.yaml"
    if not config_path.exists():
        raise FileNotFoundError(
            f"configs/default.yaml not found at {config_path}. "
            "Create it or copy from the repository root."
        )
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    depth_cfg = cfg.get("depth", {})
    delta: float = depth_cfg.get("delta", 0.5)
    model_name = depth_cfg.get("model", "dap")

    # ── Open video ─────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), f"Failed to open video: {video_path}"

    results: dict[int, np.ndarray] = {}
    frame_idx = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total_frames <= 0:
        total_frames = len(masks)
    expected_shape: tuple[int, int] | None = None

    logger.info(
        "Depth completion started: %s  model=%s  delta=%.3f",
        video_path,
        model_name,
        delta,
    )

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        assert frame_idx in masks, f"Missing mask for frame {frame_idx}"
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mask = masks[frame_idx]

        depth = _estimate_full_depth(frame_rgb, device)
        depth = _conservative_push(depth, mask, delta)

        results[frame_idx] = depth
        if expected_shape is None:
            expected_shape = depth.shape
        logger.info("프레임 %d/%d depth 완료", frame_idx + 1, total_frames)
        frame_idx += 1

    cap.release()
    if expected_shape is not None:
        for idx, depth in results.items():
            assert depth.shape == expected_shape, (
                f"Depth for frame {idx} must be {expected_shape}, got {depth.shape}"
            )

    logger.info("Depth completion done: %d frames", frame_idx)
    return results
