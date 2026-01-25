# spag4d/projection.py
"""
Projection utilities for ERP ↔ tangent plane conversions.

Supports multiple projection modes:
- Cubemap (6 faces, 90° FOV)
- Icosahedral (20 faces, ~72° FOV with overlap)
"""

import torch
import torch.nn.functional as F
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Tuple, Literal
import math


def _load_dap_projection_utils():
    from . import dap_arch

    dap_dir = dap_arch.DAP_DIR
    networks_dir = dap_dir / "networks"
    if not networks_dir.exists():
        raise ImportError(
            "DAP projection utils not found.\n"
            f"Expected {networks_dir} to exist.\n"
            "Run: git submodule update --init --recursive\n"
            "or clone https://github.com/Insta360-Research-Team/DAP into spag4d/dap_arch/DAP."
        )

    try:
        from networks.projection_utils import Equirec2Cube, Cube2Equirec
    except ImportError as exc:
        raise ImportError(
            "Failed to import DAP projection utils. Make sure DAP dependencies are installed "
            "(opencv-python, scipy)."
        ) from exc

    return Equirec2Cube, Cube2Equirec


class BaseProjector(ABC):
    """Base class for ERP ↔ tangent plane projectors."""
    
    def __init__(self, face_size: int, device: torch.device):
        self.face_size = face_size
        self.device = device
        self._sampling_grids = None  # Cached sampling grids
    
    @property
    @abstractmethod
    def num_faces(self) -> int:
        """Number of projection faces."""
        pass
    
    @property
    @abstractmethod
    def face_directions(self) -> List[np.ndarray]:
        """List of face center directions (unit vectors)."""
        pass
    
    @property
    @abstractmethod
    def face_fov(self) -> float:
        """Field of view per face in radians."""
        pass
    
    @abstractmethod
    def project_erp_to_faces(self, erp_image: np.ndarray) -> List[np.ndarray]:
        """
        Project ERP image to face images.
        
        Args:
            erp_image: [H, W, 3] ERP image (numpy uint8)
        
        Returns:
            List of [face_size, face_size, 3] face images
        """
        pass
    
    @abstractmethod
    def reproject_to_erp(
        self, 
        face_features: List[torch.Tensor],
        erp_h: int,
        erp_w: int
    ) -> torch.Tensor:
        """
        Reproject face features back to ERP.
        
        Args:
            face_features: List of [H, W, C] feature tensors
            erp_h, erp_w: Output ERP dimensions
        
        Returns:
            [erp_h, erp_w, C] blended ERP features
        """
        pass
    
    def get_blend_weights(self, erp_h: int, erp_w: int) -> torch.Tensor:
        """
        Get per-pixel blend weights for each face.
        
        Returns:
            [num_faces, erp_h, erp_w] weight tensor
        """
        # Default: uniform weights based on which face covers each pixel
        # Subclasses can override for distance-based blending
        pass


class CubemapProjector(BaseProjector):
    """6-face cubemap projection (90° FOV per face)."""
    
    # Standard cubemap face directions (OpenGL convention)
    FACE_DIRS = [
        np.array([1, 0, 0]),   # +X (right)
        np.array([-1, 0, 0]),  # -X (left)
        np.array([0, 1, 0]),   # +Y (top)
        np.array([0, -1, 0]),  # -Y (bottom)
        np.array([0, 0, 1]),   # +Z (front)
        np.array([0, 0, -1]), # -Z (back)
    ]
    
    FACE_UPS = [
        np.array([0, 1, 0]),   # +X
        np.array([0, 1, 0]),   # -X
        np.array([0, 0, -1]),  # +Y
        np.array([0, 0, 1]),   # -Y
        np.array([0, 1, 0]),   # +Z
        np.array([0, 1, 0]),   # -Z
    ]
    
    @property
    def num_faces(self) -> int:
        return 6
    
    @property
    def face_directions(self) -> List[np.ndarray]:
        return self.FACE_DIRS
    
    @property
    def face_fov(self) -> float:
        return math.pi / 2  # 90°
    
    def project_erp_to_faces(self, erp_image: np.ndarray) -> List[np.ndarray]:
        """Use existing Equirec2Cube for cubemap projection."""
        Equirec2Cube, _ = _load_dap_projection_utils()
        
        H, W = erp_image.shape[:2]
        e2c = Equirec2Cube(H, W, self.face_size)
        
        # Returns [face_size, 6*face_size, 3] horizontal strip
        full_cubemap = e2c.run(erp_image, None)
        
        # Split into 6 faces
        faces = np.split(full_cubemap, 6, axis=1)
        return faces
    
    def reproject_to_erp(
        self, 
        face_features: List[torch.Tensor],
        erp_h: int,
        erp_w: int
    ) -> torch.Tensor:
        """Use existing Cube2Equirec for reprojection."""
        _, Cube2Equirec = _load_dap_projection_utils()
        
        # Stack faces horizontally [C, face_size, 6*face_size]
        C = face_features[0].shape[-1] if face_features[0].dim() == 3 else 1
        
        stacked = []
        for f in face_features:
            if f.dim() == 2:
                f = f.unsqueeze(-1)
            stacked.append(f)
        
        # [H, 6*H, C]
        horizontal = torch.cat(stacked, dim=1)
        
        # [1, C, H, 6*H] for Cube2Equirec
        horizontal = horizontal.permute(2, 0, 1).unsqueeze(0)
        
        c2e = Cube2Equirec(self.face_size, erp_h, erp_w).to(self.device)
        erp_feat = c2e(horizontal)
        
        # [H, W, C]
        result = erp_feat.squeeze(0).permute(1, 2, 0)
        if C == 1:
            result = result.squeeze(-1)
        
        return result


class IcosahedralProjector(BaseProjector):
    """
    20-face icosahedral tangent plane projection.
    
    Uses overlapping square tangent planes centered at icosahedron
    face centroids. Larger overlap enables smooth blending at seams.
    """
    
    def __init__(self, face_size: int, device: torch.device, overlap_ratio: float = 0.3):
        super().__init__(face_size, device)
        self.overlap_ratio = overlap_ratio
        self._face_dirs = None
        self._face_ups = None
        self._compute_icosahedron_geometry()
    
    def _compute_icosahedron_geometry(self):
        """Compute icosahedron face centers and orientations."""
        # Golden ratio
        phi = (1 + math.sqrt(5)) / 2
        
        # 12 vertices of regular icosahedron
        vertices = np.array([
            [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
            [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
            [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
        ])
        # Normalize to unit sphere
        vertices = vertices / np.linalg.norm(vertices[0])
        
        # 20 triangular faces (vertex indices)
        faces = [
            (0, 1, 8), (0, 8, 4), (0, 4, 5), (0, 5, 9), (0, 9, 1),
            (3, 2, 10), (3, 10, 6), (3, 6, 7), (3, 7, 11), (3, 11, 2),
            (1, 6, 8), (8, 6, 10), (8, 10, 4), (4, 10, 2), (4, 2, 5),
            (5, 2, 11), (5, 11, 9), (9, 11, 7), (9, 7, 1), (1, 7, 6)
        ]
        
        # Compute face centers (centroids)
        self._face_dirs = []
        self._face_ups = []
        
        for f in faces:
            center = (vertices[f[0]] + vertices[f[1]] + vertices[f[2]]) / 3
            center = center / np.linalg.norm(center)  # Normalize
            self._face_dirs.append(center)
            
            # Up vector: perpendicular to center, pointing roughly toward +Y
            world_up = np.array([0, 1, 0])
            right = np.cross(world_up, center)
            if np.linalg.norm(right) < 0.01:  # Near poles
                right = np.cross(np.array([1, 0, 0]), center)
            right = right / np.linalg.norm(right)
            up = np.cross(center, right)
            up = up / np.linalg.norm(up)
            self._face_ups.append(up)
    
    @property
    def num_faces(self) -> int:
        return 20
    
    @property
    def face_directions(self) -> List[np.ndarray]:
        return self._face_dirs
    
    @property
    def face_fov(self) -> float:
        # ~72° base + overlap
        base_fov = 2 * math.atan(1 / math.sqrt(5))  # ~63.4° for inscribed
        return base_fov * (1 + self.overlap_ratio)
    
    def project_erp_to_faces(self, erp_image: np.ndarray) -> List[np.ndarray]:
        """Project ERP to 20 tangent plane images."""
        H, W = erp_image.shape[:2]
        erp_tensor = torch.from_numpy(erp_image).float().to(self.device)
        
        faces = []
        for i in range(self.num_faces):
            face = self._sample_tangent_plane(
                erp_tensor, 
                self._face_dirs[i], 
                self._face_ups[i],
                H, W
            )
            faces.append(face.cpu().numpy().astype(np.uint8))
        
        return faces
    
    def _sample_tangent_plane(
        self,
        erp: torch.Tensor,
        center_dir: np.ndarray,
        up_dir: np.ndarray,
        erp_h: int,
        erp_w: int
    ) -> torch.Tensor:
        """
        Sample a tangent plane from ERP using gnomonic projection.
        
        Args:
            erp: [H, W, 3] ERP image
            center_dir: Unit vector pointing to face center
            up_dir: Unit vector for face "up" direction
        """
        # Build tangent plane coordinate system
        forward = center_dir / np.linalg.norm(center_dir)
        up = up_dir / np.linalg.norm(up_dir)
        right = np.cross(forward, up)
        right = right / np.linalg.norm(right)
        up = np.cross(right, forward)
        
        # Create sampling grid in tangent plane coordinates
        half_fov = self.face_fov / 2
        tan_half = math.tan(half_fov)
        
        # Grid in [-tan_half, tan_half]
        u = torch.linspace(-tan_half, tan_half, self.face_size, device=self.device)
        v = torch.linspace(-tan_half, tan_half, self.face_size, device=self.device)
        uu, vv = torch.meshgrid(u, v, indexing='xy')
        
        # 3D directions in tangent plane
        right_t = torch.from_numpy(right).float().to(self.device)
        up_t = torch.from_numpy(up).float().to(self.device)
        forward_t = torch.from_numpy(forward).float().to(self.device)
        
        # Ray directions: forward + u*right + v*up
        dirs = (forward_t.view(1, 1, 3) + 
                uu.unsqueeze(-1) * right_t.view(1, 1, 3) + 
                vv.unsqueeze(-1) * up_t.view(1, 1, 3))
        
        # Normalize to unit sphere
        dirs = F.normalize(dirs, dim=-1)
        
        # Convert to spherical (theta, phi) then to ERP pixel coords
        # theta = azimuth [0, 2π], phi = elevation [0, π]
        x, y, z = dirs[..., 0], dirs[..., 1], dirs[..., 2]
        
        theta = torch.atan2(-z, x)  # [-π, π] -> [0, 2π]
        theta = (theta + 2 * math.pi) % (2 * math.pi)
        
        phi = torch.acos(y.clamp(-1, 1))  # [0, π]
        
        # To normalized coords [-1, 1] for grid_sample
        u_erp = (theta / (2 * math.pi)) * 2 - 1  # [0, 2π] -> [-1, 1]
        v_erp = (phi / math.pi) * 2 - 1          # [0, π] -> [-1, 1]
        
        grid = torch.stack([u_erp, v_erp], dim=-1).unsqueeze(0)
        
        # Sample ERP
        erp_chw = erp.permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
        sampled = F.grid_sample(
            erp_chw, grid, 
            mode='bilinear', 
            padding_mode='border',
            align_corners=True
        )
        
        # [H, W, 3]
        return sampled.squeeze(0).permute(1, 2, 0)
    
    def reproject_to_erp(
        self, 
        face_features: List[torch.Tensor],
        erp_h: int,
        erp_w: int
    ) -> torch.Tensor:
        """
        Reproject 20 face features to ERP with distance-based blending.
        """
        C = face_features[0].shape[-1] if face_features[0].dim() == 3 else 1
        
        # Accumulate weighted features
        result = torch.zeros(erp_h, erp_w, C, device=self.device)
        weights = torch.zeros(erp_h, erp_w, device=self.device)
        
        for i in range(self.num_faces):
            feat = face_features[i]
            if feat.dim() == 2:
                feat = feat.unsqueeze(-1)
            
            # Sample this face's contribution at each ERP pixel
            contribution, weight = self._sample_face_to_erp(
                feat,
                self._face_dirs[i],
                self._face_ups[i],
                erp_h, erp_w
            )
            
            result += contribution * weight.unsqueeze(-1)
            weights += weight
        
        # Normalize by total weight
        result = result / weights.unsqueeze(-1).clamp(min=1e-6)
        
        if C == 1:
            result = result.squeeze(-1)
        
        return result
    
    def _sample_face_to_erp(
        self,
        face_feat: torch.Tensor,
        center_dir: np.ndarray,
        up_dir: np.ndarray,
        erp_h: int,
        erp_w: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample face features onto ERP grid with angular weights.
        
        Returns:
            (contribution, weight) tensors of shape [erp_h, erp_w, C] and [erp_h, erp_w]
        """
        C = face_feat.shape[-1]
        
        # Build tangent plane coordinate system
        forward = center_dir / np.linalg.norm(center_dir)
        up = up_dir / np.linalg.norm(up_dir)
        right = np.cross(forward, up)
        right = right / np.linalg.norm(right)
        up = np.cross(right, forward)
        
        # Create ERP pixel directions
        u_erp = torch.linspace(0, 1, erp_w, device=self.device)
        v_erp = torch.linspace(0, 1, erp_h, device=self.device)
        uu, vv = torch.meshgrid(u_erp, v_erp, indexing='xy')
        
        theta = uu * 2 * math.pi  # [0, 2π]
        phi = vv * math.pi        # [0, π]
        
        # Spherical to Cartesian
        x = torch.sin(phi) * torch.cos(theta)
        y = torch.cos(phi)
        z = -torch.sin(phi) * torch.sin(theta)
        
        dirs = torch.stack([x, y, z], dim=-1)  # [W, H, 3]
        dirs = dirs.permute(1, 0, 2)  # [H, W, 3]
        
        # Project onto tangent plane
        forward_t = torch.from_numpy(forward).float().to(self.device)
        right_t = torch.from_numpy(right).float().to(self.device)
        up_t = torch.from_numpy(up).float().to(self.device)
        
        # Dot with forward to get depth
        depth = (dirs * forward_t).sum(dim=-1)  # [H, W]
        
        # Only valid where depth > 0 (in front of tangent plane)
        valid = depth > 0.1
        
        # Project to tangent plane coords
        proj_right = (dirs * right_t).sum(dim=-1) / depth.clamp(min=0.1)
        proj_up = (dirs * up_t).sum(dim=-1) / depth.clamp(min=0.1)
        
        # Convert to normalized coords for grid_sample
        half_fov = self.face_fov / 2
        tan_half = math.tan(half_fov)
        
        u_face = proj_right / tan_half  # [-1, 1]
        v_face = proj_up / tan_half     # [-1, 1]
        
        # Valid if within face bounds
        in_bounds = (u_face.abs() <= 1) & (v_face.abs() <= 1) & valid
        
        # Weight based on angular distance from face center
        # (smooth falloff toward edges)
        angular_dist = torch.acos((dirs * forward_t).sum(dim=-1).clamp(-1, 1))
        weight = torch.exp(-angular_dist**2 / (self.face_fov**2 / 4))
        weight = weight * in_bounds.float()
        
        # Sample face features
        grid = torch.stack([u_face, v_face], dim=-1).unsqueeze(0)
        face_chw = face_feat.permute(2, 0, 1).unsqueeze(0)  # [1, C, H, W]
        
        sampled = F.grid_sample(
            face_chw, grid,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=True
        )
        
        contribution = sampled.squeeze(0).permute(1, 2, 0)  # [H, W, C]
        
        return contribution, weight


def get_projector(
    mode: Literal["cubemap", "icosahedral"],
    face_size: int,
    device: torch.device
) -> BaseProjector:
    """Factory function to get a projector by mode name."""
    if mode == "cubemap":
        return CubemapProjector(face_size, device)
    elif mode == "icosahedral":
        return IcosahedralProjector(face_size, device)
    else:
        raise ValueError(f"Unknown projection mode: {mode}")
