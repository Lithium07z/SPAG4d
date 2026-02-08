"""
SHARP-based attribute refinement for SPAG-4D Gaussians.

Uses Apple's ML-SHARP model to extract perceptually-optimized
Gaussian attributes (opacities, scales, colors) from cubemap faces,
then transfers them to the 360° Gaussian representation.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, NamedTuple
from pathlib import Path

class RefinedAttributes(NamedTuple):
    """Refined Gaussian attributes from SHARP."""
    opacities: torch.Tensor      # [H, W] or [N]
    scales: torch.Tensor         # [H, W, 3] or [N, 3]
    colors: Optional[torch.Tensor]  # [H, W, 3] or [N, 3], optional


class SHARPRefiner:
    """
    Extracts perceptually-refined Gaussian attributes using SHARP.

    Pipeline:
    1. Convert ERP image to 6 cubemap faces
    2. Run SHARP on each face to get Gaussians3D
    3. Extract opacity, scale, color attributes
    4. Reproject to ERP coordinates
    5. Blend at seams
    """

    SHARP_WEIGHTS_URL = "https://ml-site.cdn-apple.com/models/sharp/sharp_2572gikvuh.pt"
    CACHE_DIR = Path.home() / ".cache" / "spag4d" / "sharp"

    def __init__(
        self,
        device: torch.device,
        cubemap_size: int = 1536,
        blend_width: float = 0.05,
        refine_colors: bool = True,
        projection_mode: str = "cubemap"  # "cubemap" or "icosahedral"
    ):
        """
        Args:
            device: Torch device
            cubemap_size: Size of each face (must be multiple of 384 for DINOv2)
            blend_width: Width of seam blending region (fraction of face)
            refine_colors: Whether to also refine colors from SHARP
            projection_mode: "cubemap" (6 faces) or "icosahedral" (20 faces)
        """
        self.device = device
        self.blend_width = blend_width
        self.refine_colors = refine_colors
        self.projection_mode = projection_mode

        # Validate cubemap size for DINOv2 384px patch alignment
        if cubemap_size % 384 != 0:
            valid_sizes = [1536, 1920, 2304, 3072, 3840, 4608, 6144]
            nearest = min(valid_sizes, key=lambda x: abs(x - cubemap_size))
            import warnings
            warnings.warn(
                f"cubemap_size={cubemap_size} not aligned with DINOv2 384px patches. "
                f"Adjusting to {nearest} for optimal SHARP quality."
            )
            cubemap_size = nearest
        self.cubemap_size = cubemap_size

        self.model = None
        self.projector = None  # Will be initialized in refine()

    def load_model(self, model_path: Optional[str] = None):
        """Load SHARP model weights."""
        if self.model is not None:
            return

        # Import SHARP
        try:
            from sharp.models import create_predictor, PredictorParams
        except ImportError:
            raise ImportError(
                "SHARP not installed. Install with local path using `pip install -e .[sharp]` "
                "or from git: `pip install git+https://github.com/apple/ml-sharp.git@v1.0.0`"
            )

        # Load or download weights
        if model_path is None:
            model_path = self._get_or_download_weights()

        params = PredictorParams()
        self._configure_max_quality(params)
        self.model = create_predictor(params)

        # Suppress FutureWarning about weights_only
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning, message=".*weights_only.*")
            state_dict = torch.load(model_path, map_location=self.device)
        
        self.model.load_state_dict(state_dict)
        self.model.eval()
        self.model.to(self.device)

    def _configure_max_quality(self, params):
        """Configure PredictorParams for maximum output quality.

        Based on optimized settings from 4DGS-Video-Generator research:
        - low_pass_filter_eps=0.001 preserves more detail (default is 0.01)
        - Overlapping monodepth patches reduce seam artifacts
        - All-layer color extraction for richer output
        - Delta factors tuned for fine-grained Gaussian correction
        """
        try:
            # Low-pass filter: 0.001 preserves more fine detail (SHARP default 0.01)
            params.low_pass_filter_eps = 0.001

            # Scale range
            params.max_scale = 10.0
            params.min_scale = 0.0

            # Color space: linearRGB for accurate blending
            params.color_space = "linearRGB"

            # Initializer: maximum quality extraction
            params.initializer.scale_factor = 1.0
            params.initializer.color_option = "all_layers"
            params.initializer.first_layer_depth_option = "surface_min"
            params.initializer.rest_layer_depth_option = "surface_min"
            params.initializer.normalize_depth = True

            # Delta factors: fine-tune Gaussian attribute corrections
            params.delta_factor.xy = 0.001
            params.delta_factor.z = 0.001
            params.delta_factor.color = 0.1
            params.delta_factor.opacity = 1.0
            params.delta_factor.scale = 1.0
            params.delta_factor.quaternion = 1.0

            # Monodepth: overlapping patches reduce seam artifacts (critical for 360)
            params.monodepth.use_patch_overlap = True

            # Gaussian decoder: use depth input for better geometry
            params.gaussian_decoder.use_depth_input = True
        except AttributeError as e:
            import warnings
            warnings.warn(f"Could not set some SHARP quality parameters: {e}")

    def _get_or_download_weights(self) -> str:
        """Download SHARP weights if not cached."""
        self.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_path = self.CACHE_DIR / "sharp.pt"

        if cache_path.exists():
            return str(cache_path)

        print("Downloading SHARP weights...")
        try:
            # Try direct download from Apple CDN first
            import urllib.request
            import ssl
            
            # Create unverified context to avoid SSL errors
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            
            # Simple download hook for progress
            def report(block_num, block_size, total_size):
                 if block_num % 100 == 0:
                     print(f"Downloading: {block_num * block_size / 1024 / 1024:.1f} MB", end='\r')
            
            # Use urlopen to support context, then write to file
            with urllib.request.urlopen(self.SHARP_WEIGHTS_URL, context=ctx) as response, open(cache_path, 'wb') as out_file:
                total_size = int(response.info().get('Content-Length', -1))
                downloaded = 0
                block_size = 8192
                while True:
                    buffer = response.read(block_size)
                    if not buffer:
                        break
                    downloaded += len(buffer)
                    out_file.write(buffer)
                    if downloaded % (1024*1024) == 0:
                        print(f"Downloading: {downloaded / 1024 / 1024:.1f} MB", end='\r')
                        
            print("\nDownload complete.")
            return str(cache_path)
        except Exception as e:
            print(f"Direct download failed: {e}. Trying fallback...")
            # Fallback (old behavior)
            from huggingface_hub import hf_hub_download
            return hf_hub_download(
                repo_id="apple/Sharp",
                filename="sharp.pt",
                cache_dir=self.CACHE_DIR,
                local_dir=self.CACHE_DIR,
            )

    def _init_projector(self):
        """Initialize the projector based on mode."""
        if self.projector is not None:
            return
        
        from .projection import get_projector
        self.projector = get_projector(
            mode=self.projection_mode,
            face_size=self.cubemap_size,
            device=self.device
        )

    @torch.inference_mode()
    def refine(
        self,
        erp_image: torch.Tensor,
        erp_depth: torch.Tensor,
    ) -> RefinedAttributes:
        """
        Extract refined attributes from SHARP.

        Args:
            erp_image: ERP image [H, W, 3] float [0, 1]
            erp_depth: ERP depth [H, W] from DAP (used for reference only)

        Returns:
            RefinedAttributes with opacity, scale, and optionally color maps
        """
        if self.model is None:
            self.load_model()
        
        self._init_projector()

        H, W = erp_image.shape[:2]

        # Convert image to numpy for projection
        erp_np = (erp_image.cpu().numpy() * 255).astype('uint8')

        # Project ERP to faces using the configured projector
        faces_list = self.projector.project_erp_to_faces(erp_np)
        num_faces = len(faces_list)
        
        # Run SHARP on each face
        face_opacities = []
        face_scales = []
        face_colors = []

        # Focal length based on face FOV
        f_px = self.cubemap_size / (2 * np.tan(self.projector.face_fov / 2))

        for face_idx in range(num_faces):
            face = faces_list[face_idx]  # [H, H, 3] uint8
            face_tensor = torch.from_numpy(face).float() / 255.0
            face_tensor = face_tensor.to(self.device)

            # SHARP inference
            gaussians = self._predict_face(face_tensor, f_px)

            # Extract attributes and reshape to face grid
            grid_size = self.cubemap_size // 2  # SHARP outputs at half resolution

            opacity = self._extract_opacity_map(gaussians, grid_size)
            scale = self._extract_scale_map(gaussians, grid_size)

            face_opacities.append(opacity)
            face_scales.append(scale)

            if self.refine_colors:
                color = self._extract_color_map(gaussians, grid_size)
                face_colors.append(color)
            
            # Clean up face-specific tensors
            del gaussians, face_tensor
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()

        # Upsample to match cubemap_size if needed (SHARP outputs half res)
        grid_size = self.cubemap_size // 2
        if grid_size != self.cubemap_size:
            face_opacities = [
                F.interpolate(
                    op.unsqueeze(0).unsqueeze(0),
                    size=(self.cubemap_size, self.cubemap_size),
                    mode='bilinear',
                    align_corners=True
                ).squeeze()
                for op in face_opacities
            ]
            
            face_scales = [
                F.interpolate(
                    sc.unsqueeze(0).permute(0, 3, 1, 2),
                    size=(self.cubemap_size, self.cubemap_size),
                    mode='bilinear',
                    align_corners=True
                ).squeeze().permute(1, 2, 0)
                for sc in face_scales
            ]
            
            if self.refine_colors and len(face_colors) > 0:
                face_colors = [
                    F.interpolate(
                        col.unsqueeze(0).permute(0, 3, 1, 2),
                        size=(self.cubemap_size, self.cubemap_size),
                        mode='bilinear',
                        align_corners=True
                    ).squeeze().permute(1, 2, 0)
                    for col in face_colors
                ]

        # Reproject to ERP using projector
        erp_opacities = self.projector.reproject_to_erp(face_opacities, H, W)
        erp_scales = self.projector.reproject_to_erp(face_scales, H, W)

        erp_colors = None
        if self.refine_colors and len(face_colors) > 0:
            erp_colors = self.projector.reproject_to_erp(face_colors, H, W)

        return RefinedAttributes(
            opacities=erp_opacities,
            scales=erp_scales,
            colors=erp_colors
        )

    def _predict_face(self, face: torch.Tensor, f_px: float):
        """Run SHARP on a single cubemap face."""
        # Add batch dim and run predictor
        face_batch = face.unsqueeze(0).permute(0, 3, 1, 2)  # [1, 3, H, W]

        # SHARP expects disparity_factor as normalized focal length (f_px / width)
        disparity_factor = torch.tensor([f_px / self.cubemap_size], device=self.device, dtype=torch.float32)
        gaussians = self.model(face_batch, disparity_factor=disparity_factor)

        return gaussians

    def _extract_opacity_map(
        self,
        gaussians,  # Gaussians3D
        grid_size: int
    ) -> torch.Tensor:
        """Extract opacity as 2D map from SHARP output."""
        # gaussians.opacities: [B, N]
        # SHARP uses 2 layers, so N = grid_size^2 * 2
        opacities = gaussians.opacities[0]  # [N]

        # Reshape to [2, grid_size, grid_size] and take mean across layers
        n_layers = 2
        per_layer = grid_size * grid_size

        if opacities.shape[0] == per_layer * n_layers:
            opacities = opacities.view(n_layers, grid_size, grid_size)
            opacities = opacities.mean(dim=0)  # [grid_size, grid_size]
        else:
            # Fallback: interpolate to grid if size mismatch
            # This handles cases where model might output different resolution
            N = opacities.shape[0]
            side = int(np.sqrt(N // n_layers))
            if side * side * n_layers == N:
                opacities = opacities.view(n_layers, side, side)
                opacities = opacities.mean(dim=0)
                if side != grid_size:
                    opacities = F.interpolate(
                        opacities.unsqueeze(0).unsqueeze(0), 
                        size=(grid_size, grid_size),
                        mode='bilinear'
                    ).squeeze()
            else:
                 # Last resort flat interpolation
                opacities = opacities.view(1, 1, -1, 1)
                opacities = F.interpolate(opacities, size=(grid_size, grid_size))
                opacities = opacities.squeeze()

        return opacities.clamp(0, 1)

    def _extract_scale_map(
        self,
        gaussians,  # Gaussians3D
        grid_size: int
    ) -> torch.Tensor:
        """Extract scales as 2D map from SHARP output."""
        # gaussians.singular_values: [B, N, 3]
        scales = gaussians.singular_values[0]  # [N, 3]

        n_layers = 2
        per_layer = grid_size * grid_size

        if scales.shape[0] == per_layer * n_layers:
            scales = scales.view(n_layers, grid_size, grid_size, 3)
            scales = scales.mean(dim=0)  # [grid_size, grid_size, 3]
        else:
            # Fallback
            scales = scales.view(1, -1, 1, 3).permute(0, 3, 1, 2)
            scales = F.interpolate(scales, size=(grid_size, grid_size))
            scales = scales.permute(0, 2, 3, 1).squeeze(0)

        return scales

    def _extract_color_map(
        self,
        gaussians,  # Gaussians3D
        grid_size: int
    ) -> torch.Tensor:
        """Extract colors as 2D map from SHARP output."""
        colors = gaussians.colors[0]  # [N, 3]

        n_layers = 2
        per_layer = grid_size * grid_size

        if colors.shape[0] == per_layer * n_layers:
            colors = colors.view(n_layers, grid_size, grid_size, 3)
            colors = colors.mean(dim=0)
        else:
            colors = colors.view(1, -1, 1, 3).permute(0, 3, 1, 2)
            colors = F.interpolate(colors, size=(grid_size, grid_size))
            colors = colors.permute(0, 2, 3, 1).squeeze(0)

        return colors

    def _cubemap_to_erp(
        self,
        cube_strip: torch.Tensor
    ) -> torch.Tensor:
        """
        Reproject cubemap strip to ERP.

        Args:
            cube_strip: [1, C, H, 6*H]

        Returns:
            ERP attributes [H, W, C]
        """
        # cube_strip is already formatted for Cube2Equirec
        # Cube2Equirec.forward returns [1, C, H_erp, W_erp]
        
        erp_feat = self.c2e(cube_strip)
        
        # Squeeze batch, permute to [H, W, C]
        result = erp_feat.squeeze(0).permute(1, 2, 0)
        
        # If C=1, squeeze last dim? No, prefer keeping it consistent or squeezing if single channel.
        # RefinedAttributes expects [H, W] for opacities (1 dim)
        if result.shape[-1] == 1:
            result = result.squeeze(-1)
            
        return result
