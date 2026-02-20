"""
pipeline/composer.py

End-to-end layered 3DGS composer for occlusion-complete 360° Gaussian video.

Pipeline per frame:
  1. Foreground layer
       - Estimate original-frame depth via SPAG4D's built-in depth model
         (converter.dap.predict) — reuses the already-loaded model, no
         extra inference cost.
       - Push background pixels to depth > sky_threshold so
         validity_mask = (depth <= sky_threshold) zeroes them out.
       - Black out background pixels in the frame image (alpha=0 equivalent).
       - Call SPAG4D.convert(..., depth_override=fg_depth, sky_dome=False).

  2. Background layer
       - Load the inpainted background frame from bg_video_dir (PNG directory
         or MP4 video — both are handled automatically).
       - Call SPAG4D.convert(..., depth_override=bg_depth[frame_idx],
         sky_dome=True).

  3. PLY merge
       - Concatenate vertex arrays from both PLY files using plyfile.
       - All 3DGS vertex attributes are preserved:
           x, y, z, nx, ny, nz
           f_dc_0, f_dc_1, f_dc_2
           opacity
           scale_0, scale_1, scale_2
           rot_0, rot_1, rot_2, rot_3
       - open3d.io.read_point_cloud silently drops these custom fields and is
         therefore NOT suitable for 3DGS PLY merging.  If open3d is installed
         its presence is logged; the actual merge always uses plyfile.

SPAG4D initialisation:
  - One SPAG4D instance is created before the frame loop (expensive model
    load happens once).
  - use_sharp_refinement=False  — SHARP is too slow for per-frame video.
  - use_guided_filter=True      — keeps depth edges sharp for both layers.
  - Config keys read from configs/default.yaml:
      depth.model            → depth model ("panda" | "da3" | "dap")
      composer.stride        → bg_stride  (background layer stride, default 2)
      composer.fg_stride     → fg_stride  (foreground layer stride, default 1)
      composer.sky_threshold → sky depth cutoff (default 80.0 m)
"""

from __future__ import annotations

import logging
import shutil
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

from spag4d.core import SPAG4D

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# PLY merge
# ─────────────────────────────────────────────────────────────────────────────

def _merge_ply_files(fg_path: str, bg_path: str, out_path: str) -> int:
    """
    Concatenate two 3DGS PLY files into a single merged PLY.

    Why plyfile, not open3d:
        open3d.io.read_point_cloud reads only {x, y, z, rgb} and silently
        discards all custom vertex properties.  A 3DGS PLY stores 20+ fields
        per Gaussian (positions, normals, SH DC/rest, opacity, log-scales,
        quaternions).  Dropping them produces a geometry-only point cloud that
        cannot be rendered as Gaussians.  plyfile concatenates raw structured
        numpy arrays, preserving every field unchanged.

    open3d presence:
        If open3d is installed its import is attempted (logs to DEBUG);
        the merge path is identical regardless.

    Args:
        fg_path: Foreground PLY file.
        bg_path: Background PLY file.
        out_path: Destination merged PLY.

    Returns:
        Total Gaussian count in the merged file.
    """
    from plyfile import PlyData, PlyElement

    # open3d presence check — merge logic is the same either way
    try:
        import open3d  # noqa: F401
        logger.debug(
            "open3d available; using plyfile for field-complete 3DGS merge "
            "(open3d.read_point_cloud drops custom fields)"
        )
    except ImportError:
        logger.debug("open3d absent; using numpy/plyfile for PLY merge")

    fg_ply = PlyData.read(fg_path)
    bg_ply = PlyData.read(bg_path)

    fg_verts = fg_ply["vertex"].data
    bg_verts = bg_ply["vertex"].data

    merged = np.concatenate([fg_verts, bg_verts])
    el = PlyElement.describe(merged, "vertex")
    PlyData([el], text=False).write(out_path)

    return int(merged.shape[0])


# ─────────────────────────────────────────────────────────────────────────────
# Background frame source  (PNG directory  OR  video file)
# ─────────────────────────────────────────────────────────────────────────────

def _open_bg_source(bg_video_dir: str) -> tuple:
    """
    Open a background frame source.

    Accepts two formats produced by different versions of
    cubemap_inpaint_video:
      - Directory of PNG frames: frame_00000.png, frame_00001.png, …
      - Single video file (MP4 or similar): returned by the updated
        cubemap_inpaint_video that encodes to MP4 before returning.

    Returns:
        ("dir", Path)               — for directory sources
        ("cap", cv2.VideoCapture)   — for video file sources
    """
    p = Path(bg_video_dir)
    if p.is_dir():
        return ("dir", p)
    cap = cv2.VideoCapture(str(p))
    assert cap.isOpened(), f"Failed to open background video: {bg_video_dir}"
    return ("cap", cap)


def _read_bg_frame(bg_source: tuple, frame_idx: int) -> np.ndarray:
    """
    Read a single background frame as RGB uint8 (H, W, 3).

    For directory sources: reads frame_{frame_idx:05d}.png directly.
    For video sources: seeks to frame_idx then reads one frame.
        Seeking in compressed video is approximate; for best results
        use a PNG-directory source (lossless, O(1) access).

    Args:
        bg_source: Tuple returned by _open_bg_source.
        frame_idx: 0-based frame index.

    Returns:
        (H, W, 3) uint8 RGB array.
    """
    kind, obj = bg_source
    if kind == "dir":
        frame_path = obj / f"frame_{frame_idx:05d}.png"
        assert frame_path.exists(), f"Missing background frame: {frame_path}"
        bgr = cv2.imread(str(frame_path))
        assert bgr is not None, f"Failed to read background frame: {frame_path}"
    else:
        # Seek to requested frame (VideoCapture positional seek)
        obj.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, bgr = obj.read()
        assert ret, f"Failed to read background frame {frame_idx} from video"
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _close_bg_source(bg_source: tuple) -> None:
    """Release VideoCapture if bg_source is a video file."""
    kind, obj = bg_source
    if kind == "cap":
        obj.release()


# ─────────────────────────────────────────────────────────────────────────────
# Main API
# ─────────────────────────────────────────────────────────────────────────────

def generate_layered_ply(
    video_path: str,
    masks: dict[int, np.ndarray],
    bg_video_dir: str,
    bg_depth: dict[int, np.ndarray],
    output_dir: str,
    device: str = "cuda",
) -> list[str]:
    """
    Generate per-frame layered 3DGS PLY files from foreground and background.

    For each frame the function:
      1. Estimates foreground depth from the original frame using the SPAG4D
         internal depth model (``converter.dap.predict``).
      2. Generates a foreground PLY: background pixels are excluded by pushing
         their depth above ``sky_threshold`` so the validity mask zeroes them.
      3. Generates a background PLY using the inpainted frame and the
         pre-computed ``bg_depth`` via ``depth_override``.
      4. Merges both PLYs (plyfile concatenation — preserves all 3DGS fields).
      5. Writes ``output_dir/frame_{idx:04d}_merged.ply``.

    Args:
        video_path: Path to the original equirectangular video file.
        masks: Per-frame foreground masks from
            :func:`pipeline.segmentation.generate_masks`.
            Keys are 0-based frame indices; values are ``(H, W)`` uint8 arrays
            (1 = foreground, 0 = background).
        bg_video_dir: Path to the inpainted background.  Accepts:
            - A directory of PNG frames (``frame_00000.png``, …) as originally
              produced by ``cubemap_inpaint_video``, OR
            - An MP4 video file as produced by the updated version that encodes
              to video before returning.
        bg_depth: Per-frame completed background depth maps from
            :func:`pipeline.depth.complete_depth`.
            Keys are 0-based frame indices; values are ``(H, W)`` float32 arrays
            in metres.
        output_dir: Directory where merged PLY files are written.
            Created if it does not exist.
        device: Torch device string, e.g. ``"cuda"`` or ``"cpu"``.

    Returns:
        Sorted list of absolute paths (str) to the generated merged PLY files,
        one per frame.

    Raises:
        FileNotFoundError: If ``configs/default.yaml`` is missing.
        AssertionError: If a required mask or depth entry is absent, or if
            either video source cannot be opened.

    Example::

        from pipeline.segmentation import generate_masks
        from pipeline.inpainting   import cubemap_inpaint_video
        from pipeline.depth        import complete_depth
        from pipeline.composer     import generate_layered_ply

        masks    = generate_masks("scene.mp4", "/weights/sam3.pth", "cuda")
        bg_video = cubemap_inpaint_video("scene.mp4", masks, "/weights/lama", "cuda")
        bg_depth = complete_depth(bg_video, masks, device="cuda")
        ply_paths = generate_layered_ply(
            video_path="scene.mp4",
            masks=masks,
            bg_video_dir=bg_video,
            bg_depth=bg_depth,
            output_dir="output/layered",
            device="cuda",
        )
        # ply_paths[0] == "output/layered/frame_0000_merged.ply"
    """
    # ── 1. Load config ────────────────────────────────────────────────────────
    config_path = Path(__file__).parent.parent / "configs" / "default.yaml"
    if not config_path.exists():
        raise FileNotFoundError(
            f"configs/default.yaml not found at {config_path}. "
            "Create it or copy from the repository root."
        )
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    composer_cfg     = cfg.get("composer", {})
    bg_stride        = int(composer_cfg.get("stride", 2))
    fg_stride        = int(composer_cfg.get("fg_stride", 1))
    sky_threshold    = float(composer_cfg.get("sky_threshold", 80.0))
    depth_model_name = cfg.get("depth", {}).get("model", "panda")

    # ── 2. Initialise SPAG4D once (outside the frame loop) ───────────────────
    # The depth model loaded here is reused for FG depth estimation via
    # converter.dap.predict() — no separate model load needed.
    # use_sharp_refinement=False: SHARP per-frame inference is too slow for video.
    # use_guided_filter=True: keeps depth edges sharp on both layers.
    logger.info(
        "SPAG4D 초기화 중 (depth_model=%s, device=%s, "
        "fg_stride=%d, bg_stride=%d, sky_threshold=%.1f)",
        depth_model_name, device, fg_stride, bg_stride, sky_threshold,
    )
    converter = SPAG4D(
        device=device,
        depth_model=depth_model_name,
        use_guided_filter=True,
        use_sharp_refinement=False,  # 비디오 처리 속도 우선
    )

    # ── 3. Open video sources and prepare output ──────────────────────────────
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), f"원본 비디오 열기 실패: {video_path}"
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) or len(masks)

    bg_source = _open_bg_source(bg_video_dir)

    # Temp directory for intermediate per-frame images and partial PLY files
    tmp_dir = Path(tempfile.mkdtemp(prefix="spag4d_composer_"))

    merged_paths: list[str] = []
    frame_idx = 0

    logger.info("Composer 시작: 총 ~%d 프레임", total_frames)

    try:
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break

            assert frame_idx in masks, (
                f"masks에 프레임 {frame_idx} 항목 없음"
            )
            assert frame_idx in bg_depth, (
                f"bg_depth에 프레임 {frame_idx} 항목 없음"
            )

            original_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mask     = masks[frame_idx]      # (H, W) uint8, 1=fg 0=bg
            mask_bool = mask.astype(bool)    # (H, W) bool

            # ── a. 전경 depth 추정 (SPAG4D 내장 DAP/PanDA 재사용) ────────────
            logger.info(
                "[%d/%d] 전경 depth 추정 중 ...", frame_idx + 1, total_frames
            )
            img_tensor = torch.from_numpy(original_rgb).to(converter.device)
            with torch.inference_mode():
                fg_raw_depth_t, _ = converter.dap.predict(img_tensor)
            # (H, W) float32, metres (PanDA: pseudo-metric)
            fg_depth_np = fg_raw_depth_t.detach().float().cpu().numpy()

            # ── b. FG depth_override 구성 ─────────────────────────────────────
            # 배경 픽셀: depth를 sky_threshold + 1로 설정
            # → convert() 내부에서 validity_mask = (depth <= sky_threshold)
            #   가 0으로 처리하므로 해당 Gaussian이 생성되지 않음
            fg_depth_override = fg_depth_np.copy()
            fg_depth_override[~mask_bool] = sky_threshold + 1.0

            # ── c. FG 프레임: 배경 픽셀 → 검정 (alpha=0 동등 처리) ───────────
            fg_frame = original_rgb.copy()
            fg_frame[~mask_bool] = 0

            # ── d. FG 프레임 → 임시 PNG 저장 ─────────────────────────────────
            fg_img_path = tmp_dir / f"fg_{frame_idx:05d}.png"
            cv2.imwrite(
                str(fg_img_path),
                cv2.cvtColor(fg_frame, cv2.COLOR_RGB2BGR),
            )

            # ── e. 전경 PLY 생성 ──────────────────────────────────────────────
            logger.info(
                "[%d/%d] 전경 레이어 PLY 변환 중 (stride=%d) ...",
                frame_idx + 1, total_frames, fg_stride,
            )
            fg_ply_path = tmp_dir / f"fg_{frame_idx:05d}.ply"
            fg_ply_ok = False
            try:
                fg_result = converter.convert(
                    input_path=str(fg_img_path),
                    output_path=str(fg_ply_path),
                    stride=fg_stride,
                    depth_override=fg_depth_override,
                    sky_threshold=sky_threshold,
                    # sky_dome=False: 배경 하늘은 배경 레이어가 담당
                    # 전경 레이어에서 sky dome을 생성하면 fg/bg 중복이 발생함
                    sky_dome=False,
                    force_erp=True,
                )
                logger.info(
                    "  → 전경 PLY: %s Gaussians (%.2fs)",
                    f"{fg_result.splat_count:,}",
                    fg_result.processing_time,
                )
                fg_ply_ok = True
            except ValueError as exc:
                # ply_writer.save_ply_gsplat은 N==0이면 ValueError를 발생시킴
                # 마스크가 완전히 비어있거나 sky_threshold로 전부 제거된 경우
                logger.warning(
                    "  [frame %d] 전경 Gaussian 없음 — FG 레이어 건너뜀: %s",
                    frame_idx, exc,
                )

            # ── f. 배경 프레임 로드 ───────────────────────────────────────────
            logger.info(
                "[%d/%d] 배경 레이어 PLY 변환 중 (stride=%d) ...",
                frame_idx + 1, total_frames, bg_stride,
            )
            bg_frame_rgb = _read_bg_frame(bg_source, frame_idx)

            # ── g. 배경 프레임 → 임시 PNG 저장 ───────────────────────────────
            bg_img_path = tmp_dir / f"bg_{frame_idx:05d}.png"
            cv2.imwrite(
                str(bg_img_path),
                cv2.cvtColor(bg_frame_rgb, cv2.COLOR_RGB2BGR),
            )

            # ── h. 배경 PLY 생성 (depth_override=bg_depth) ───────────────────
            # bg_depth는 conservative push로 완성된 배경 depth:
            # 마스크 내부 depth = 주변 배경 중앙값 + delta
            # depth_override를 사용하므로 SPAG4D의 내부 depth 추정을 건너뜀
            bg_ply_path = tmp_dir / f"bg_{frame_idx:05d}.ply"
            bg_result = converter.convert(
                input_path=str(bg_img_path),
                output_path=str(bg_ply_path),
                stride=bg_stride,
                depth_override=bg_depth[frame_idx],
                sky_threshold=sky_threshold,
                sky_dome=True,   # 배경 하늘 dome 포함
                force_erp=True,
            )
            logger.info(
                "  → 배경 PLY: %s Gaussians (%.2fs)",
                f"{bg_result.splat_count:,}",
                bg_result.processing_time,
            )

            # ── i. FG + BG PLY 합성 ───────────────────────────────────────────
            merged_name    = f"frame_{frame_idx:04d}_merged.ply"
            merged_ply_path = out_dir / merged_name

            if fg_ply_ok:
                # plyfile concatenate: 모든 3DGS 필드 보존
                total_splats = _merge_ply_files(
                    str(fg_ply_path),
                    str(bg_ply_path),
                    str(merged_ply_path),
                )
                logger.info(
                    "  → merged PLY: %s Gaussians → %s",
                    f"{total_splats:,}", merged_name,
                )
            else:
                # 전경 Gaussian이 없으면 배경 PLY를 그대로 복사
                shutil.copy2(str(bg_ply_path), str(merged_ply_path))
                logger.info(
                    "  → merged PLY (배경 전용): %s Gaussians → %s",
                    f"{bg_result.splat_count:,}", merged_name,
                )

            merged_paths.append(str(merged_ply_path))
            frame_idx += 1

    finally:
        # 성공/실패 어느 경우에도 자원 해제 및 임시 파일 정리
        cap.release()
        _close_bg_source(bg_source)
        shutil.rmtree(str(tmp_dir), ignore_errors=True)
        logger.info("임시 파일 정리 완료: %s", tmp_dir)

    logger.info(
        "Composer 완료: %d 프레임 → %s", frame_idx, output_dir
    )
    return sorted(merged_paths)
