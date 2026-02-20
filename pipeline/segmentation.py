"""
pipeline/segmentation.py

Foreground mask generation using SAM3 (Segment Anything Model 3).
Handles 360° equirectangular seam wrap-around so objects crossing the
left/right boundary are segmented as a single connected region.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
import yaml  # 직접 임포트 — importlib 지연 로드 불필요
from sam3.model_builder import build_sam3_video_predictor


def generate_masks(
    video_path: str,
    sam3_ckpt: str,
    device: str,
) -> dict[int, np.ndarray]:
    """
    Generate per-frame binary foreground masks for a 360° video.

    Runs SAM3 on each frame and post-processes masks to handle the
    equirectangular seam: the image is horizontally mirrored and overlaid
    so that objects straddling the left/right edge are treated as contiguous.

    Args:
        video_path: Path to input equirectangular video file (e.g. .mp4).
        sam3_ckpt: Path to SAM3 model checkpoint (.pth).
        device: Torch device string, e.g. ``"cuda"`` or ``"cpu"``.

    Returns:
        A dict mapping frame index (0-based int) to a binary mask array of
        shape ``(H, W)`` with dtype ``np.uint8``.  Foreground pixels have
        value ``1``, background pixels have value ``0``.

    Example::

        masks = generate_masks("scene.mp4", "/weights/sam3.pth", "cuda")
        # masks[0].shape == (1024, 2048), dtype uint8
    """
    logger = logging.getLogger(__name__)

    # CLAUDE.md: 하드코딩 경로 금지 → __file__ 기준 절대 경로로 CWD 독립성 확보
    config_path = Path(__file__).parent.parent / "configs" / "default.yaml"
    assert config_path.exists(), (
        f"configs/default.yaml not found at {config_path}. "
        "Provide segmentation.prompts in the config file."
    )

    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    assert isinstance(cfg, dict) and "segmentation" in cfg, (
        "Invalid config: missing 'segmentation' section in configs/default.yaml."
    )
    prompts = cfg["segmentation"].get("prompts")
    assert isinstance(prompts, (list, tuple)) and len(prompts) > 0, (
        "Invalid config: segmentation.prompts must be a non-empty list."
    )

    predictor = build_sam3_video_predictor(
        checkpoint_path=sam3_ckpt,
        device=device,
    )

    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), f"Failed to open video: {video_path}"

    masks: dict[int, np.ndarray] = {}
    frame_idx = 0
    frame_shape = None

    logger.info("SAM3 segmentation started: frames from %s", video_path)
    logger.info("Using prompts: %s", ", ".join([str(p) for p in prompts]))

    def _predict_mask(image_rgb: np.ndarray) -> np.ndarray:
        """Aggregate masks across prompts for a single frame."""
        mask_accum = None
        for prompt in prompts:
            # TODO: Confirm SAM3 text-prompt API and output format.
            pred = predictor.predict(image_rgb, text_prompt=prompt)

            if isinstance(pred, dict) and "masks" in pred:
                mask = pred["masks"]
            elif isinstance(pred, (list, tuple)):
                mask = pred[0]
            else:
                mask = pred

            assert mask is not None, "SAM3 returned empty mask."

            if mask.ndim == 3:
                mask = np.any(mask, axis=0)
            assert mask.ndim == 2, "SAM3 mask must be 2D (H, W)."

            mask_bool = mask.astype(bool)
            if mask_accum is None:
                mask_accum = mask_bool
            else:
                mask_accum = np.logical_or(mask_accum, mask_bool)

        assert mask_accum is not None, "No masks produced from prompts."
        return mask_accum

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        H, W = frame_rgb.shape[:2]
        if frame_shape is None:
            frame_shape = (H, W)

        mask_main = _predict_mask(frame_rgb)

        rolled_rgb = np.roll(frame_rgb, W // 2, axis=1)
        mask_rolled = _predict_mask(rolled_rgb)
        mask_rolled = np.roll(mask_rolled, -(W // 2), axis=1)

        mask_combined = np.logical_or(mask_main, mask_rolled)
        # bool → uint8 변환: depth.py·composer.py 모두 uint8(0/1)을 기대하며
        # docstring 계약과도 일치해야 함
        masks[frame_idx] = mask_combined.astype(np.uint8)

        frame_idx += 1

    cap.release()

    if frame_shape is not None:
        expected_h, expected_w = frame_shape
        for idx, mask in masks.items():
            assert mask.shape == (expected_h, expected_w), (
                f"Mask shape mismatch at frame {idx}: "
                f"expected {(expected_h, expected_w)}, got {mask.shape}"
            )
            # .astype(np.uint8) 변환이 실제로 적용됐는지 확인
            assert mask.dtype == np.uint8, (
                f"Mask dtype must be np.uint8 at frame {idx}, got {mask.dtype}"
            )

    logger.info("SAM3 segmentation done: %d frames", len(masks))
    return masks
