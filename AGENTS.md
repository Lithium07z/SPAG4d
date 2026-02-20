# AGENTS.md — Codex 작업 가이드

## 역할 분담
| 담당 | 작업 범위 |
|---|---|
| **Codex** | `pipeline/` 함수 단위 구현, 버그 수정, 소규모 기능 추가 |
| **Claude Code** | 전체 설계, `spag4d/` 수정, 파이프라인 통합 검토, 노트북 업데이트 |

`spag4d/` 내부 파일은 직접 수정하지 않는다. `pipeline/` 범위 밖 변경이 필요하면 Claude에게 위임한다.

---

## 현재 구현 상태 (2026-02-20 기준)

모든 pipeline/ 파일이 구현 완료되어 있음. Codex는 **기존 코드를 기반으로 버그 수정 또는 기능 추가**를 수행한다.

```
pipeline/
  segmentation.py   ✅ 구현 완료 — SAM3 전경 마스크 생성
  inpainting.py     ✅ 구현 완료 — CubeMap LaMa 배경 인페인팅
  depth.py          ✅ 구현 완료 — 배경 depth 보수적 완성
  composer.py       ✅ 구현 완료 — 전경/배경 3DGS PLY 합성
  utils/cubemap.py  ✅ 구현 완료 — ERP↔CubeMap 변환 헬퍼
```

---

## 확정된 함수 시그니처

새 코드를 작성할 때 아래 시그니처를 정확히 따를 것. 변경 시 Claude와 협의.

```python
# pipeline/segmentation.py
generate_masks(video_path: str, sam3_ckpt: str, device: str) -> dict[int, np.ndarray]
# 반환: frame_idx → (H, W) uint8 배열, 1=전경 0=배경

# pipeline/inpainting.py
cubemap_inpaint_video(video_path: str, masks: dict[int, np.ndarray],
                      lama_path: str, device: str) -> str
# 반환: 인페인팅 MP4 경로 문자열 (video_bg.mp4)

# pipeline/depth.py
complete_depth(video_path: str, masks: dict[int, np.ndarray],
               device: str = "cuda") -> dict[int, np.ndarray]
# video_path: 인페인팅 배경 MP4 — 원본 비디오가 아님
# 반환: frame_idx → (H, W) float32 depth 배열 (metres)

# pipeline/composer.py
generate_layered_ply(video_path: str, masks: dict[int, np.ndarray],
                     bg_video_dir: str, bg_depth: dict[int, np.ndarray],
                     output_dir: str, device: str = "cuda") -> list[str]
# 반환: 정렬된 merged PLY 경로 목록
```

---

## 핵심 임포트 경로

### SPAG4D 클래스
```python
from spag4d import SPAG4D           # 권장 (spag4d/__init__.py에서 re-export)
from spag4d.core import SPAG4D      # 직접 경로 (동일하게 동작)

converter = SPAG4D(
    device="cuda",
    depth_model="panda",            # configs/default.yaml의 depth.model 값 사용
    use_guided_filter=True,
    use_sharp_refinement=False,     # 비디오 처리 시 속도 우선 — True는 이미지 단건용
)
```

### depth 모델 (spag4d.__init__에서 re-export 안 됨 — 반드시 서브모듈에서 직접 임포트)
```python
from spag4d.panda_model import PanDAModel   # depth.model == "panda"
from spag4d.da3_model   import DA3Model     # depth.model == "da3"
from spag4d.dap_model   import DAPModel     # depth.model == "dap"

model = PanDAModel.load(model_path=None, device=torch.device(device))
depth_t, _ = model.predict(image_tensor)   # image_tensor: (H,W,3) uint8 Tensor
# depth_t: (H,W) float32, metres (PanDA: pseudo-metric)
```

> **중요**: 어떤 depth_model을 로드해도 `SPAG4D` 인스턴스 내에서는 항상 `converter.dap`으로 접근한다.

### CubeMap 변환 (py360convert 직접 사용 금지)
```python
from pipeline.utils.cubemap import erp_to_cube, cube_to_erp, get_face_masks

cube_faces = erp_to_cube(erp_img, face_w=512)       # (6, F, F, 3) float32
erp_out    = cube_to_erp(cube_faces, h=H, w=W)      # (H, W, 3) float32
face_masks = get_face_masks(erp_mask, face_w=512)   # (6, F, F) uint8
```

### LaMa 초기화 (시그니처 가변 — inspect로 호환성 확보)
```python
from simple_lama_inpainting import SimpleLama
from inspect import signature

sig = signature(SimpleLama.__init__)
kwargs = {}
if "model_path" in sig.parameters: kwargs["model_path"] = lama_path
elif "model_dir" in sig.parameters: kwargs["model_dir"] = lama_path
if "device" in sig.parameters: kwargs["device"] = device
lama_model = SimpleLama(**kwargs)
```

### SAM3
```python
from sam3.model_builder import build_sam3_video_predictor

# SAM3 API는 device 인자를 직접 받지 않음 — 로드 후 이동
predictor = build_sam3_video_predictor(checkpoint_path=SAM3_PATH)
if hasattr(predictor, "to"):
    predictor = predictor.to(device)
elif hasattr(predictor, "model"):
    predictor.model = predictor.model.to(device)
```

---

## configs/default.yaml 구조

```yaml
segmentation:
  prompts: ["person", "car", "bicycle"]
depth:
  model: "panda"    # "panda" | "da3" | "dap"
  delta: 0.5
inpainting:
  overlap_pad: 0.1
composer:
  output_format: "ply"
  stride: 2          # 배경 레이어 stride
  fg_stride: 1       # 전경 레이어 stride
  sky_threshold: 80.0
```

config 참조 코드 (CWD 독립성 — 이 패턴만 사용할 것):
```python
from pathlib import Path
import yaml

config_path = Path(__file__).parent.parent / "configs" / "default.yaml"
with config_path.open("r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
```

---

## depth_override 사용 패턴

`SPAG4D.convert()`의 `depth_override` 파라미터는 이미 구현되어 있다.

```python
result = converter.convert(
    input_path=str(img_path),     # 반드시 이미지 파일 경로 (비디오 프레임 PNG)
    output_path=str(ply_path),
    stride=fg_stride,
    depth_override=np_depth_array,  # (H,W) float32 또는 (H,W,1) — 자동 squeeze
    sky_threshold=sky_threshold,
    sky_dome=False,                 # **kwargs로 전달됨
    force_erp=True,
)
```

---

## PLY 병합 규칙 (open3d 사용 금지)

`open3d.read_point_cloud`은 3DGS 커스텀 필드(opacity, f_dc, scale, rot 등)를 **무음으로 삭제**한다.
PLY 병합은 반드시 `plyfile`로 raw numpy array를 연결할 것.

```python
from plyfile import PlyData, PlyElement
import numpy as np

fg_verts = PlyData.read(fg_path)["vertex"].data
bg_verts = PlyData.read(bg_path)["vertex"].data
merged = np.concatenate([fg_verts, bg_verts])
PlyData([PlyElement.describe(merged, "vertex")], text=False).write(out_path)
```

---

## 360° seam 처리 패턴 (segmentation)

ERP 이미지는 좌우가 이어져 있으므로 경계를 가로지르는 객체를 처리할 때:

```python
# 1. 원본 마스크 예측
mask_orig = predictor.predict(frame_rgb)

# 2. 좌우 절반 shift 후 재예측
W = frame_rgb.shape[1]
frame_rolled = np.roll(frame_rgb, W // 2, axis=1)
mask_rolled  = predictor.predict(frame_rolled)

# 3. roll 복원 후 OR 합성
mask_rolled_back = np.roll(mask_rolled, -(W // 2), axis=1)
mask_combined = (np.logical_or(mask_orig, mask_rolled_back)).astype(np.uint8)
```

---

## 코드 작성 규칙

- 함수마다 docstring 필수 (Args / Returns / Raises)
- 경로 하드코딩 금지 — 인자로 받거나 `configs/default.yaml` 참조
- GPU 코드는 `device` 인자 항상 명시 (default `"cuda"`)
- assert + 명확한 메시지 사용, 불필요한 try/except 남발 금지
- `print` 디버깅 금지 → `logging.getLogger(__name__)` 사용
- dtype 명시: 마스크는 `np.uint8`, depth는 `np.float32`

## 자주 발생하는 버그 (이전 구현에서 발견됨)

| 버그 패턴 | 원인 | 수정 방법 |
|---|---|---|
| config 경로 `Path("configs/default.yaml")` | CWD 의존 | `Path(__file__).parent.parent / ...` 사용 |
| 마스크 dtype `bool` | `np.logical_or` 반환값 | `.astype(np.uint8)` 명시 |
| depth 모델 이름 무시 | 조건 분기 누락 | panda/da3/dap 3-way 분기 구현 |
| `return str(output_dir)` | 디렉토리 반환 | MP4 인코딩 후 파일 경로 반환 |
| `complete_depth(INPUT, ...)` | 원본 비디오 전달 | 인페인팅 결과 `bg_video` 전달 |

## ���� ���
- run_pipeline.ipynb: Colab PIL ImportError ������ ���� Pillow 10.2.0 ���� �缳ġ �� �߰�.
