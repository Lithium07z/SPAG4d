"""
pipeline/inpainting.py

Background inpainting for 360° video.

Workflow:
  1. Project each equirectangular frame to CubeMap faces (6 × face_size²).
  2. Run LaMa or ProPainter inpainting on each face individually.
  3. Back-project CubeMap faces to equirectangular.
  4. Write result frames to a new video file.

CubeMap projection avoids the severe perspective distortion near the poles
that makes inpainting of the raw equirectangular image unreliable.
"""

from __future__ import annotations

import logging
import shutil
import tempfile
from inspect import signature
from pathlib import Path

import cv2
import numpy as np
from simple_lama_inpainting import SimpleLama

from pipeline.utils.cubemap import cube_to_erp, erp_to_cube, get_face_masks


def handle_seam_face(
    face_img: np.ndarray,
    face_mask: np.ndarray,
    lama_model: SimpleLama,
) -> np.ndarray:
    """
    Inpaint a cubemap face with seam-aware handling.

    If the mask touches left/right edges, performs an additional inpaint on
    a horizontally rolled version and merges the result to reduce seam artifacts.

    Args:
        face_img: Face RGB image, shape (F, F, 3), dtype uint8.
        face_mask: Face mask, shape (F, F), dtype uint8 (0 or 255).
        lama_model: Initialized SimpleLama model.

    Returns:
        Inpainted face image, shape (F, F, 3), dtype uint8.
    """
    assert face_img.ndim == 3 and face_img.shape[2] == 3, (
        f"face_img must be (F, F, 3), got {face_img.shape}"
    )
    assert face_mask.ndim == 2, f"face_mask must be 2D, got {face_mask.shape}"

    base = inpaint_face(face_img, face_mask, lama_model)

    edge_touch = np.any(face_mask[:, 0] > 0) or np.any(face_mask[:, -1] > 0)
    if not edge_touch:
        return base

    F = face_img.shape[1]
    shift = F // 2
    rolled_img = np.roll(face_img, shift, axis=1)
    rolled_mask = np.roll(face_mask, shift, axis=1)
    rolled_inpaint = inpaint_face(rolled_img, rolled_mask, lama_model)
    rolled_back = np.roll(rolled_inpaint, -shift, axis=1)

    # TODO: Consider a smoother blend instead of hard overwrite.
    return np.where(face_mask[..., None] > 0, rolled_back, base)


def inpaint_face(
    face_img: np.ndarray,
    face_mask: np.ndarray,
    lama_model: SimpleLama,
) -> np.ndarray:
    """
    Run LaMa inpainting on a single face.

    Args:
        face_img: Face RGB image, shape (F, F, 3), dtype uint8.
        face_mask: Face mask, shape (F, F), dtype uint8 (0 or 255).
        lama_model: Initialized SimpleLama model.

    Returns:
        Inpainted face image, shape (F, F, 3), dtype uint8.
    """
    assert face_img.ndim == 3 and face_img.shape[2] == 3, (
        f"face_img must be (F, F, 3), got {face_img.shape}"
    )
    assert face_mask.ndim == 2, f"face_mask must be 2D, got {face_mask.shape}"

    if hasattr(lama_model, "inpaint"):
        result = lama_model.inpaint(face_img, face_mask)
    else:
        result = lama_model(face_img, face_mask)

    if isinstance(result, np.ndarray):
        return result.astype(np.uint8)

    # Fall back to PIL image output
    try:
        result_np = np.array(result)
    except Exception as exc:  # pragma: no cover - defensive
        raise AssertionError(
            "LaMa output must be a numpy array or PIL image."
        ) from exc

    return result_np.astype(np.uint8)


def cubemap_inpaint_single(
    erp_img: np.ndarray,
    erp_mask: np.ndarray,
    lama_model: SimpleLama,
    overlap_pad: float = 0.1,
) -> np.ndarray:
    """
    Inpaint a single ERP frame by projecting to cubemap faces.

    Args:
        erp_img: ERP RGB image, shape (H, W, 3), dtype uint8.
        erp_mask: ERP mask, shape (H, W), dtype uint8 (0/1 or 0/255).
        lama_model: Initialized SimpleLama model.
        overlap_pad: Fractional padding (relative to face size).

    Returns:
        Inpainted ERP image, shape (H, W, 3), dtype uint8.
    """
    assert erp_img.ndim == 3 and erp_img.shape[2] == 3, (
        f"erp_img must be (H, W, 3), got {erp_img.shape}"
    )
    assert erp_mask.ndim == 2, f"erp_mask must be 2D, got {erp_mask.shape}"

    H, W = erp_img.shape[:2]
    face_w = H // 2
    assert face_w > 0, f"Invalid face size derived from H={H}"

    cube_faces = erp_to_cube(erp_img, face_w=face_w)  # (6, F, F, 3)
    face_masks = get_face_masks(
        (erp_mask > 0).astype(np.uint8), face_w=face_w
    )  # (6, F, F)

    pad_px = int(round(face_w * overlap_pad))
    pad_px = max(pad_px, 0)

    inpainted_faces = []
    for fi in range(6):
        face_img = cube_faces[fi].astype(np.uint8)
        face_mask = (face_masks[fi] > 0).astype(np.uint8) * 255

        if pad_px > 0:
            face_img_pad = cv2.copyMakeBorder(
                face_img,
                pad_px,
                pad_px,
                pad_px,
                pad_px,
                borderType=cv2.BORDER_REFLECT_101,
            )
            face_mask_pad = cv2.copyMakeBorder(
                face_mask,
                pad_px,
                pad_px,
                pad_px,
                pad_px,
                borderType=cv2.BORDER_CONSTANT,
                value=0,
            )
            inpaint_pad = handle_seam_face(face_img_pad, face_mask_pad, lama_model)
            inpaint_face_img = inpaint_pad[
                pad_px:-pad_px,
                pad_px:-pad_px,
            ]
        else:
            inpaint_face_img = handle_seam_face(face_img, face_mask, lama_model)

        inpainted_faces.append(inpaint_face_img.astype(np.float32))

    inpainted_cube = np.stack(inpainted_faces, axis=0)  # (6, F, F, 3)
    inpainted_erp = cube_to_erp(inpainted_cube, h=H, w=W)

    return np.clip(inpainted_erp, 0, 255).astype(np.uint8)


def cubemap_inpaint_video(
    video_path: str,
    masks: dict[int, np.ndarray],
    lama_path: str,
    device: str,
) -> str:
    """
    Inpaint foreground-masked regions in a 360° video using CubeMap projection.

    Each frame is projected to six CubeMap faces, the foreground mask is
    reprojected accordingly, LaMa (or ProPainter for temporal consistency)
    fills the masked regions, and the result is projected back to
    equirectangular layout.  The inpainted video is written next to
    ``video_path`` with a ``_bg`` suffix.

    Args:
        video_path: Path to the original equirectangular video file.
        masks: Per-frame binary foreground masks as returned by
            :func:`pipeline.segmentation.generate_masks`.
            Keys are 0-based frame indices; values are ``(H, W)`` uint8 arrays.
        lama_path: Path to LaMa model checkpoint directory or weights file.
        device: Torch device string, e.g. ``"cuda"`` or ``"cpu"``.

    Returns:
        Path (str) to the inpainted background video file
        (equirectangular, same resolution as ``video_path``).

    Example::

        bg_video = cubemap_inpaint_video(
            "scene.mp4", masks, "/weights/lama", "cuda"
        )
        # bg_video == "scene_bg.mp4"
    """
    logger = logging.getLogger(__name__)

    lama_sig = signature(SimpleLama.__init__)
    lama_kwargs = {}
    if "model_path" in lama_sig.parameters:
        lama_kwargs["model_path"] = lama_path
    elif "model_dir" in lama_sig.parameters:
        lama_kwargs["model_dir"] = lama_path
    if "device" in lama_sig.parameters:
        lama_kwargs["device"] = device

    lama_model = SimpleLama(**lama_kwargs)

    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), f"Failed to open video: {video_path}"

    # 비디오 속성은 루프 진입 전에 읽어야 함 — cap.release() 이후에는 접근 불가
    fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_dir = Path(tempfile.mkdtemp(prefix="spag4d_inpaint_"))
    logger.info("Writing inpainted frames to %s", output_dir)

    frame_idx = 0
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        assert frame_idx in masks, f"Missing mask for frame {frame_idx}"
        mask = masks[frame_idx]

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        inpainted_rgb = cubemap_inpaint_single(
            frame_rgb, mask, lama_model, overlap_pad=0.1
        )

        out_path = output_dir / f"frame_{frame_idx:05d}.png"
        cv2.imwrite(
            str(out_path),
            cv2.cvtColor(inpainted_rgb, cv2.COLOR_RGB2BGR),
        )

        frame_idx += 1

    cap.release()

    # BUG FIX: 원래 str(output_dir) 반환 → 디렉토리 경로였음.
    # complete_depth / generate_layered_ply 모두 video_path(str)를 기대하므로
    # 프레임 PNG들을 MP4로 인코딩한 뒤 파일 경로를 반환하도록 수정.
    out_video_path = str(
        Path(video_path).with_name(Path(video_path).stem + "_bg.mp4")
    )
    writer = cv2.VideoWriter(
        out_video_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (src_w, src_h),
    )
    for i in range(frame_idx):
        frame = cv2.imread(str(output_dir / f"frame_{i:05d}.png"))
        if frame is not None:
            writer.write(frame)
    writer.release()

    shutil.rmtree(str(output_dir))  # 임시 프레임 디렉토리 정리
    logger.info("Inpainting done: %d frames → %s", frame_idx, out_video_path)
    return out_video_path
