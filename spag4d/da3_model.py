# spag4d/da3_model.py
"""
Wrapper for Depth Anything V3 (DA3) model.

DA3 is a unified depth estimation model from ByteDance-Seed that supports
monocular depth, multi-view depth, pose estimation, and Gaussian prediction.

For SPAG-4D we use DA3Metric-Large for metric depth output (real-world scale)
or DA3Mono-Large for relative depth (similar to PanDA).

Install: pip install -e "git+https://github.com/ByteDance-Seed/depth-anything-3.git#egg=depth_anything_3"
Reference: https://github.com/ByteDance-Seed/Depth-Anything-3
"""

import torch
import numpy as np
from typing import Optional


# Model presets
DA3_MODELS = {
    "metric": "depth-anything/DA3METRIC-LARGE",
    "mono": "depth-anything/DA3MONO-LARGE",
    "large": "depth-anything/DA3-LARGE",
}


class DA3Model:
    """
    Wrapper for Depth Anything V3 with same predict() interface as DAP/PanDA.

    Uses DA3Metric-Large by default for metric depth output (real-world meters).
    Falls back to DA3Mono-Large (relative depth) if metric is unavailable.
    """

    def __init__(
        self,
        model,
        device: torch.device,
        variant: str = "metric",
        depth_min: float = 0.1,
        depth_max: float = 100.0,
    ):
        self.model = model
        self.device = device
        self.variant = variant
        self.depth_min = depth_min
        self.depth_max = depth_max
        # DA3Metric outputs metric depth directly; DA3Mono outputs relative
        self.is_metric = variant == "metric"

    @classmethod
    def load(
        cls,
        variant: str = "metric",
        device: torch.device = torch.device('cuda'),
        depth_min: float = 0.1,
        depth_max: float = 100.0,
    ) -> 'DA3Model':
        """
        Load DA3 model from HuggingFace.

        Args:
            variant: Model variant - "metric" (real scale) or "mono" (relative)
            device: Torch device
            depth_min: Min depth for relative-to-metric scaling (mono variant only)
            depth_max: Max depth for relative-to-metric scaling (mono variant only)

        Returns:
            Loaded DA3Model instance
        """
        try:
            from depth_anything_3.api import DepthAnything3
        except ImportError:
            raise ImportError(
                "Depth Anything V3 not installed. Install with:\n"
                '  pip install -e "git+https://github.com/ByteDance-Seed/depth-anything-3.git#egg=depth_anything_3"\n'
                "Or: pip install depth-anything-3"
            )

        model_id = DA3_MODELS.get(variant)
        if model_id is None:
            raise ValueError(f"Unknown DA3 variant '{variant}'. Choose from: {list(DA3_MODELS.keys())}")

        print(f"Loading DA3 model: {model_id}")
        model = DepthAnything3.from_pretrained(model_id)
        model = model.to(device=device)

        return cls(model, device, variant, depth_min, depth_max)

    @torch.inference_mode()
    def predict(
        self,
        image: torch.Tensor,
        return_mask: bool = False
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Predict depth from equirectangular image.

        Args:
            image: RGB image tensor [H, W, 3] uint8 or [0,1] float
            return_mask: Unused, kept for API compatibility

        Returns:
            Tuple of (depth, mask):
                - depth: [H, W] in meters (metric) or pseudo-metric (mono)
                - mask: Confidence mask [H, W] or None
        """
        H, W = image.shape[0], image.shape[1]

        # DA3 expects numpy arrays or PIL images, not tensors
        if image.dtype == torch.uint8:
            img_np = image.cpu().numpy()  # [H, W, 3] uint8
        else:
            img_np = (image.cpu().float() * 255).clamp(0, 255).byte().numpy()

        # Run inference — DA3 handles its own resizing internally
        # Default process_res is 504, which is safe for VRAM
        prediction = self.model.inference(
            [img_np],
            process_res=504,
            process_res_method="upper_bound_resize",
        )

        # prediction.depth is [N, H, W] numpy float32
        depth_np = prediction.depth[0]  # [H, W]
        depth = torch.from_numpy(depth_np).to(self.device)

        # Resize to original resolution if DA3 changed it
        if depth.shape[0] != H or depth.shape[1] != W:
            import torch.nn.functional as F
            depth = F.interpolate(
                depth.unsqueeze(0).unsqueeze(0),
                size=(H, W),
                mode='bilinear',
                align_corners=True,
            ).squeeze()

        # For non-metric variant, scale relative depth to pseudo-metric range
        if not self.is_metric:
            d_min, d_max = depth.min(), depth.max()
            if d_max > d_min:
                depth_normalized = (depth - d_min) / (d_max - d_min)
            else:
                depth_normalized = torch.zeros_like(depth)
            depth = self.depth_min + depth_normalized * (self.depth_max - self.depth_min)

        # DA3 provides confidence maps
        mask = None
        if prediction.conf is not None:
            conf_np = prediction.conf[0]  # [H, W]
            mask = torch.from_numpy(conf_np).to(self.device)
            if mask.shape[0] != H or mask.shape[1] != W:
                import torch.nn.functional as F
                mask = F.interpolate(
                    mask.unsqueeze(0).unsqueeze(0),
                    size=(H, W),
                    mode='bilinear',
                    align_corners=True,
                ).squeeze()

        return depth, mask
