# spag4d/panda_model.py
"""
Wrapper for PanDA (Panoramic Depth Anything) model.

PanDA is a CVPR 2025 model that fine-tunes Depth Anything V2 with LoRA
and Möbius spatial augmentation for robust 360° depth estimation.

Outputs relative depth (0-1 range) which is scaled to pseudo-metric
using configurable depth_min/depth_max parameters.

Reference: https://github.com/caozidong/PanDA
License: Apache 2.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Optional
import hashlib


# Model configuration
PANDA_CONFIG = {
    "url": "https://huggingface.co/ZidongC/PanDA/resolve/main/panda_large.pth",
    "repo_id": "ZidongC/PanDA",
    "filename": "panda_large.pth",
    "sha256": None,  # Skip checksum verification
    "size_mb": 1500,
}
PANDA_CACHE_DIR = Path.home() / ".cache" / "spag4d"


class PanDAModel:
    """
    Wrapper for PanDA (Panoramic Depth Anything) model.

    PanDA is specifically designed for 360° equirectangular images
    and outputs relative depth (0-1) which is scaled to a configurable range.
    """

    def __init__(self, model: nn.Module, device: torch.device, depth_min: float = 0.1, depth_max: float = 100.0, depth_mapping: str = "log"):
        self.model = model
        self.device = device
        self.depth_min = depth_min
        self.depth_max = depth_max
        self.depth_mapping = depth_mapping  # "log", "linear", or "inverse"
        self.model.eval()

    @classmethod
    def load(
        cls,
        model_path: Optional[str] = None,
        device: torch.device = torch.device('cuda'),
        depth_min: float = 0.1,
        depth_max: float = 100.0,
        depth_mapping: str = "log",
    ) -> 'PanDAModel':
        """
        Load PanDA model from path or download from HuggingFace.

        Args:
            model_path: Optional explicit path to weights
            device: Torch device to load to
            depth_min: Minimum depth for relative-to-metric scaling
            depth_max: Maximum depth for relative-to-metric scaling

        Returns:
            Loaded PanDAModel instance
        """
        if model_path is None:
            model_path = cls._get_or_download_weights()

        # Import PanDA model architecture
        try:
            from .panda_arch import build_panda_model
        except ImportError:
            raise ImportError(
                "PanDA architecture not found. Please clone the PanDA model files "
                "from https://github.com/caozidong/PanDA to spag4d/panda_arch/PanDA"
            )

        model = build_panda_model()

        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)

        # Handle different checkpoint formats
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif isinstance(checkpoint, dict) and 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint

        # Strip 'module.' prefix if model was saved with DataParallel
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('module.'):
                new_key = key[7:]  # Remove 'module.' prefix
            else:
                new_key = key
            new_state_dict[new_key] = value

        # Load with strict=False to handle any minor mismatches
        model.load_state_dict(new_state_dict, strict=False)
        model = model.to(device)

        return cls(model, device, depth_min, depth_max, depth_mapping)

    @classmethod
    def _get_or_download_weights(cls) -> str:
        """Download weights with verification."""
        PANDA_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_path = PANDA_CACHE_DIR / "panda_large.pth"

        if cache_path.exists():
            # Verify checksum if available
            if cls._verify_checksum(cache_path):
                print(f"Using cached PanDA weights: {cache_path}")
                return str(cache_path)
            else:
                print("Cached weights corrupted, re-downloading...")
                cache_path.unlink()

        # Download with progress
        print(f"Downloading PanDA weights (~{PANDA_CONFIG['size_mb']}MB)...")

        try:
            # Prefer huggingface_hub for resumable downloads
            from huggingface_hub import hf_hub_download
            downloaded_path = hf_hub_download(
                repo_id=PANDA_CONFIG["repo_id"],
                filename=PANDA_CONFIG["filename"],
                cache_dir=PANDA_CACHE_DIR,
                local_dir=PANDA_CACHE_DIR,
            )
            return downloaded_path
        except ImportError:
            # Fallback to urllib
            import urllib.request

            try:
                from tqdm import tqdm

                class DownloadProgress(tqdm):
                    def update_to(self, b=1, bsize=1, tsize=None):
                        if tsize is not None:
                            self.total = tsize
                        self.update(b * bsize - self.n)

                with DownloadProgress(unit='B', unit_scale=True, miniters=1) as t:
                    urllib.request.urlretrieve(
                        PANDA_CONFIG["url"],
                        cache_path,
                        reporthook=t.update_to
                    )
            except ImportError:
                # No tqdm, download silently
                urllib.request.urlretrieve(PANDA_CONFIG["url"], cache_path)

        return str(cache_path)

    @staticmethod
    def _verify_checksum(path: Path) -> bool:
        """Verify file SHA256 checksum."""
        if not PANDA_CONFIG.get("sha256"):
            return True  # Skip if no checksum configured

        sha256 = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)

        return sha256.hexdigest() == PANDA_CONFIG["sha256"]

    @torch.inference_mode()
    def predict(
        self,
        image: torch.Tensor,
        return_mask: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Predict depth from equirectangular image(s).

        PanDA outputs relative depth (0-1) which is scaled to
        pseudo-metric range using depth_min/depth_max.

        Args:
            image: RGB image tensor [H, W, 3] or batch [B, H, W, 3] uint8 or [0,1] float
            return_mask: Unused (PanDA doesn't output masks), kept for API compat

        Returns:
            Tuple of (depth, mask):
                - depth: [H, W] or [B, H, W] in pseudo-metric units
                - mask: None (PanDA doesn't produce validity masks)
        """
        # Handle batched vs single input
        is_batched = image.dim() == 4
        if not is_batched:
            image = image.unsqueeze(0)  # [1, H, W, 3]

        B, H, W, C = image.shape

        # Preprocess
        if image.dtype == torch.uint8:
            image = image.float() / 255.0

        # PanDA expects [B, C, H, W] normalized with ImageNet stats
        x = image.permute(0, 3, 1, 2)  # [B, 3, H, W]

        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        x = (x - mean) / std

        # PanDA requires dimensions to be multiples of 14 (ViT patch size)
        # Cap input resolution to prevent OOM
        MAX_INPUT_WIDTH = 1022
        did_downscale = False
        target_H, target_W = H, W

        if W > MAX_INPUT_WIDTH:
            scale = MAX_INPUT_WIDTH / W
            target_H = int(H * scale)
            target_W = MAX_INPUT_WIDTH
            did_downscale = True

        # Ensure multiple of 14
        target_H = target_H - (target_H % 14)
        target_W = target_W - (target_W % 14)

        if target_H != H or target_W != W:
            x = F.interpolate(x, size=(target_H, target_W), mode='bilinear', align_corners=True)
            if did_downscale:
                print(f"[PanDA] Downscaled input {W}×{H} → {target_W}×{target_H} for inference")

        # Run model with OOM fallback
        try:
            output = self.model(x)
        except torch.cuda.OutOfMemoryError:
            # Fallback: process one at a time
            torch.cuda.empty_cache()
            depths = []
            for i in range(B):
                out_i = self.model(x[i:i+1])
                if isinstance(out_i, dict):
                    depths.append(out_i['pred_depth'])
                else:
                    depths.append(out_i)
            depth = torch.cat(depths, dim=0)
            output = {'pred_depth': depth}

        # Handle different output formats
        if isinstance(output, dict):
            depth = output['pred_depth']
        else:
            depth = output

        # Remove channel dim if present [B, 1, H, W] -> [B, H, W]
        if depth.dim() == 4:
            depth = depth.squeeze(1)

        # Interpolate to original resolution if needed
        if depth.shape[-2] != H or depth.shape[-1] != W:
            depth = F.interpolate(
                depth.unsqueeze(1),
                size=(H, W),
                mode='bilinear',
                align_corners=True
            ).squeeze(1)

        # Scale relative depth (0-1) to pseudo-metric range.
        # PanDA outputs: higher value = farther away.
        # Normalize to [0, 1] first (model output may not be exactly 0-1).
        depth_min_val = depth.min()
        depth_max_val = depth.max()
        if depth_max_val > depth_min_val:
            depth_normalized = (depth - depth_min_val) / (depth_max_val - depth_min_val)
        else:
            depth_normalized = torch.zeros_like(depth)

        # Map to pseudo-metric range using the configured mapping.
        # Log-space preserves foreground detail: each order of magnitude
        # (0.1–1m, 1–10m, 10–100m) gets equal representation.
        if self.depth_mapping == "log":
            import math
            log_min = math.log(self.depth_min)
            log_max = math.log(self.depth_max)
            depth = torch.exp(log_min + depth_normalized * (log_max - log_min))
        elif self.depth_mapping == "inverse":
            # Disparity-linear (matches monocular model training signal)
            inv_max = 1.0 / self.depth_min
            inv_min = 1.0 / self.depth_max
            inv_depth = inv_max - depth_normalized * (inv_max - inv_min)
            depth = 1.0 / inv_depth.clamp(min=1e-6)
        else:
            # Linear (legacy behaviour)
            depth = self.depth_min + depth_normalized * (self.depth_max - self.depth_min)

        # Remove batch dim for single image input
        if not is_batched:
            depth = depth.squeeze(0)

        # PanDA doesn't output masks
        mask = None

        return depth, mask

