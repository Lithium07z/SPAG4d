
import torch
import numpy as np
from spag4d.spherical_grid import create_spherical_grid

try:
    import scipy
    print("Scipy imported successfully")
except ImportError:
    print("Scipy import failed!")

def test_coords():
    H, W = 180, 360
    device = torch.device('cpu')
    grid = create_spherical_grid(H, W, device, stride=1)
    
    # Test North Pole (Top)
    # Expected: Y=1
    top_y = grid.rhat[0, W//2, 1].item()
    print(f"Top Y: {top_y:.4f} (Expected > 0.95)")
    
    # Test South Pole (Bottom)
    # Expected: Y=-1
    bot_y = grid.rhat[H-1, W//2, 1].item()
    print(f"Bottom Y: {bot_y:.4f} (Expected < -0.95)")
    
    if top_y > 0.95 and bot_y < -0.95:
        print("PASS: Coordinate system is Y-up")
    else:
        print("FAIL: Coordinate system check failed")

if __name__ == "__main__":
    test_coords()
