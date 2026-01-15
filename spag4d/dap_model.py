# spag4d/dap_model.py
"""
Wrapper for DAP (Depth Any Panoramas) model.

DAP is specifically designed for 360° equirectangular images
and outputs metric depth in meters.

Reference: https://github.com/Insta360-Research-Team/DAP
License: MIT
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional
import hashlib


# Model configuration
DAP_CONFIG = {
    "url": "https://huggingface.co/Insta360-Research/DAP-weights/resolve/main/model.pth",
    "repo_id": "Insta360-Research/DAP-weights",
    "filename": "model.pth",
    "sha256": None,  # Will be set after first verified download
    "size_mb": 1500,
}
DAP_CACHE_DIR = Path.home() / ".cache" / "spag4d"


class DAPModel:
    """
    Wrapper for DAP (Depth Any Panoramas) model.
    
    DAP is specifically designed for 360° equirectangular images
    and outputs metric depth in meters.
    """
    
    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model
        self.device = device
        self.model.eval()
    
    @classmethod
    def load(
        cls, 
        model_path: Optional[str] = None,
        device: torch.device = torch.device('cuda')
    ) -> 'DAPModel':
        """
        Load DAP model from path or download from HuggingFace.
        
        Args:
            model_path: Optional explicit path to weights
            device: Torch device to load to
        
        Returns:
            Loaded DAPModel instance
        """
        if model_path is None:
            model_path = cls._get_or_download_weights()
        
        # Import DAP model architecture
        try:
            from .dap_arch import build_dap_model
        except ImportError:
            raise ImportError(
                "DAP architecture not found. Please copy the DAP model files "
                "from https://github.com/Insta360-Research-Team/DAP to spag4d/dap_arch/"
            )
        
        model = build_dap_model()
        
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
        
        return cls(model, device)
    
    @classmethod
    def _get_or_download_weights(cls) -> str:
        """Download weights with verification."""
        DAP_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_path = DAP_CACHE_DIR / "dap_large.pth"
        
        if cache_path.exists():
            # Verify checksum if available
            if cls._verify_checksum(cache_path):
                return str(cache_path)
            else:
                print("Cached weights corrupted, re-downloading...")
                cache_path.unlink()
        
        # Download with progress
        print(f"Downloading DAP weights (~{DAP_CONFIG['size_mb']}MB)...")
        
        try:
            # Prefer huggingface_hub for resumable downloads
            from huggingface_hub import hf_hub_download
            downloaded_path = hf_hub_download(
                repo_id=DAP_CONFIG["repo_id"],
                filename=DAP_CONFIG["filename"],
                cache_dir=DAP_CACHE_DIR,
                local_dir=DAP_CACHE_DIR,
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
                        DAP_CONFIG["url"], 
                        cache_path,
                        reporthook=t.update_to
                    )
            except ImportError:
                # No tqdm, download silently
                urllib.request.urlretrieve(DAP_CONFIG["url"], cache_path)
        
        return str(cache_path)
    
    @staticmethod
    def _verify_checksum(path: Path) -> bool:
        """Verify file SHA256 checksum."""
        if not DAP_CONFIG.get("sha256"):
            return True  # Skip if no checksum configured
        
        sha256 = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        
        return sha256.hexdigest() == DAP_CONFIG["sha256"]
    
    @torch.inference_mode()
    def predict(self, image: torch.Tensor) -> torch.Tensor:
        """
        Predict metric depth from equirectangular image.
        
        Args:
            image: RGB image tensor [H, W, 3] uint8 or [0,1] float
        
        Returns:
            Depth map [H, W] in meters
        """
        import torch.nn.functional as F
        
        H, W = image.shape[:2]
        
        # Preprocess
        if image.dtype == torch.uint8:
            image = image.float() / 255.0
        
        # DAP expects [B, C, H, W] normalized with ImageNet stats
        x = image.permute(2, 0, 1).unsqueeze(0)
        
        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        x = (x - mean) / std
        
        # Run model - DAP returns a dict with 'pred_depth'
        output = self.model(x)
        
        # Handle different output formats
        if isinstance(output, dict):
            depth = output['pred_depth']
        else:
            depth = output
        
        # Remove batch/channel dims if present
        if depth.dim() == 4:
            depth = depth.squeeze(0).squeeze(0)
        elif depth.dim() == 3:
            depth = depth.squeeze(0)
        
        # Interpolate to original resolution if needed
        if depth.shape[0] != H or depth.shape[1] != W:
            depth = F.interpolate(
                depth.unsqueeze(0).unsqueeze(0),
                size=(H, W),
                mode='bilinear',
                align_corners=True
            ).squeeze(0).squeeze(0)
        
        return depth


class MockDAPModel(DAPModel):
    """
    Mock DAP model for testing without weights.
    
    Returns synthetic depth based on image brightness.
    """
    
    def __init__(self, device: torch.device):
        self.device = device
        self.model = None
    
    @classmethod
    def load(cls, model_path: Optional[str] = None, device: torch.device = torch.device('cpu')) -> 'MockDAPModel':
        return cls(device)
    
    @torch.inference_mode()
    def predict(self, image: torch.Tensor) -> torch.Tensor:
        """Return synthetic depth based on image brightness."""
        H, W = image.shape[:2]
        
        if image.dtype == torch.uint8:
            image = image.float() / 255.0
        
        # Base depth: 5 meters
        depth = torch.ones(H, W, device=self.device) * 5.0
        
        # Add variation based on brightness (brighter = farther)
        brightness = image.mean(dim=-1)
        depth = depth + brightness * 10.0
        
        # Add some noise
        depth = depth + torch.randn(H, W, device=self.device) * 0.3
        
        return depth.clamp(min=0.1)
