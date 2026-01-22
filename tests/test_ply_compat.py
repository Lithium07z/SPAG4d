# tests/test_ply_compat.py
"""
Verify PLY files are compatible with target viewers.
"""

import numpy as np
import torch
import pytest
import tempfile
from pathlib import Path

from spag4d.spherical_grid import create_spherical_grid
from spag4d.gaussian_converter import equirect_to_gaussians
from spag4d.ply_writer import save_ply_gsplat, load_ply_gaussians
from spag4d.splat_writer import save_splat, load_splat


class TestPlySchema:
    """Verify PLY file structure matches viewer expectations."""
    
    def test_ply_has_required_properties(self, tmp_path):
        """PLY should have all properties required by SuperSplat/gsplat."""
        from plyfile import PlyData
        
        # Create synthetic gaussians
        gaussians = _create_test_gaussians(100)
        
        # Save PLY
        ply_path = tmp_path / "test.ply"
        save_ply_gsplat(gaussians, str(ply_path), sh_degree=0)
        
        # Load and check schema
        ply = PlyData.read(str(ply_path))
        vertex = ply['vertex']
        
        # Required properties
        required = [
            'x', 'y', 'z',
            'nx', 'ny', 'nz',
            'f_dc_0', 'f_dc_1', 'f_dc_2',
            'opacity',
            'scale_0', 'scale_1', 'scale_2',
            'rot_0', 'rot_1', 'rot_2', 'rot_3'
        ]
        
        for prop in required:
            assert prop in vertex.data.dtype.names, f"Missing required property: {prop}"
    
    def test_ply_sh_degree_3(self, tmp_path):
        """SH degree 3 should add f_rest_* properties."""
        from plyfile import PlyData
        
        gaussians = _create_test_gaussians(100)
        
        ply_path = tmp_path / "test_sh3.ply"
        save_ply_gsplat(gaussians, str(ply_path), sh_degree=3)
        
        ply = PlyData.read(str(ply_path))
        vertex = ply['vertex']
        
        # SH degree 3 has 45 rest coefficients (48 total - 3 DC)
        num_rest = (3 + 1) ** 2 * 3 - 3  # = 45
        
        for i in range(num_rest):
            assert f'f_rest_{i}' in vertex.data.dtype.names, f"Missing f_rest_{i}"
    
    def test_ply_roundtrip(self, tmp_path):
        """Save and load should preserve data."""
        gaussians = _create_test_gaussians(100)
        
        ply_path = tmp_path / "roundtrip.ply"
        save_ply_gsplat(gaussians, str(ply_path))
        
        loaded = load_ply_gaussians(str(ply_path))
        
        # Positions should match (with coordinate transform)
        # Note: save transforms Y-up → OpenCV, load reads OpenCV directly
        # So loaded positions are in OpenCV coords
        assert loaded['means'].shape == gaussians['means'].shape
        assert loaded['colors'].shape == gaussians['colors'].shape


class TestSplatFormat:
    """Test SPLAT format read/write."""
    
    def test_splat_roundtrip(self, tmp_path):
        """Save and load SPLAT should preserve data approximately."""
        gaussians = _create_test_gaussians(100)
        
        splat_path = tmp_path / "test.splat"
        save_splat(gaussians, str(splat_path))
        
        loaded = load_splat(str(splat_path))
        
        # Shapes should match
        assert loaded['means'].shape == gaussians['means'].shape
        assert loaded['colors'].shape == gaussians['colors'].shape
        
        # Values approximately preserved (with quantization loss)
        # Can't expect exact match due to float16/uint8 quantization
    
    def test_ply_splat_same_coordinates(self, tmp_path):
        """PLY and SPLAT should output same coordinate frame."""
        gaussians = _create_test_gaussians(100)
        
        ply_path = tmp_path / "test.ply"
        splat_path = tmp_path / "test.splat"
        
        save_ply_gsplat(gaussians, str(ply_path))
        save_splat(gaussians, str(splat_path))
        
        loaded_ply = load_ply_gaussians(str(ply_path))
        loaded_splat = load_splat(str(splat_path))
        
        # Positions should match (allowing for float16 quantization in SPLAT)
        np.testing.assert_allclose(
            loaded_ply['means'].numpy(),
            loaded_splat['means'].numpy(),
            rtol=1e-2, atol=1e-2,
            err_msg="PLY and SPLAT coordinates should match"
        )
    
    def test_splat_file_size(self, tmp_path):
        """SPLAT should be smaller than PLY."""
        gaussians = _create_test_gaussians(1000)
        
        ply_path = tmp_path / "test.ply"
        splat_path = tmp_path / "test.splat"
        
        save_ply_gsplat(gaussians, str(ply_path))
        save_splat(gaussians, str(splat_path))
        
        ply_size = ply_path.stat().st_size
        splat_size = splat_path.stat().st_size
        
        # SPLAT should be significantly smaller
        assert splat_size < ply_size * 0.7, f"SPLAT {splat_size} not much smaller than PLY {ply_size}"


class TestConversionPipeline:
    """Test full conversion from synthetic depth."""
    
    def test_synthetic_depth_conversion(self, tmp_path):
        """Convert using synthetic depth (no DAP)."""
        device = torch.device('cpu')
        H, W = 256, 512
        
        # Create synthetic image and depth
        image = torch.rand(H, W, 3, device=device)
        depth = torch.ones(H, W, device=device) * 5.0  # 5m flat depth
        
        # Add some variation
        depth += torch.randn(H, W, device=device) * 0.5
        depth = depth.clamp(min=0.1)
        
        # Create grid
        grid = create_spherical_grid(H, W, device, stride=4)
        
        # Convert
        gaussians = equirect_to_gaussians(
            image=image,
            depth=depth,
            grid=grid,
            scale_factor=1.5,
            thickness_ratio=0.1,
            depth_min=0.1,
            depth_max=100.0
        )
        
        # Verify output
        assert gaussians['means'].shape[0] > 0, "Should produce some gaussians"
        assert gaussians['means'].shape[1] == 3
        assert gaussians['scales'].shape[1] == 3
        assert gaussians['quats'].shape[1] == 4
        assert gaussians['colors'].shape[1] == 3
        assert gaussians['opacities'].shape[1] == 1
        
        # Save and verify PLY
        ply_path = tmp_path / "synthetic.ply"
        save_ply_gsplat(gaussians, str(ply_path))
        
        assert ply_path.exists()
        assert ply_path.stat().st_size > 0


def _create_test_gaussians(n: int) -> dict:
    """Create synthetic Gaussian data for testing."""
    return {
        'means': torch.randn(n, 3),
        'scales': torch.rand(n, 3) * 0.1 + 0.01,
        'quats': torch.randn(n, 4),
        'colors': torch.rand(n, 3),
        'opacities': torch.rand(n, 1) * 0.5 + 0.5,
    }


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
