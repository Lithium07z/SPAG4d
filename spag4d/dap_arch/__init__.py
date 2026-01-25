# spag4d/dap_arch/__init__.py
"""
DAP (Depth Any Panoramas) model architecture.

Wraps the DAP model from Insta360 Research.
https://github.com/Insta360-Research-Team/DAP
"""

import sys
import os
from pathlib import Path

# Get the DAP directory path
DAP_DIR = Path(__file__).parent / "DAP"

# Add DAP subdirectory and its subdirectories to path for imports
if DAP_DIR.exists():
    # Add main DAP dir
    if str(DAP_DIR) not in sys.path:
        sys.path.insert(0, str(DAP_DIR))
    
    # Add depth_anything_v2_metric
    depth_metric_dir = DAP_DIR / "depth_anything_v2_metric"
    if depth_metric_dir.exists() and str(depth_metric_dir) not in sys.path:
        sys.path.insert(0, str(depth_metric_dir))
    
    # Change working directory temporarily to DAP_DIR for relative path imports
    _original_cwd = os.getcwd()


def build_dap_model(max_depth: float = 100.0):
    """
    Build DAP model architecture.
    
    Args:
        max_depth: Maximum depth in meters for metric output
    
    Returns:
        nn.Module: DAP model ready for weight loading
    """
    import os
    
    # Save original working directory
    original_cwd = os.getcwd()
    
    try:
        # Change to DAP directory so relative paths work
        os.chdir(str(DAP_DIR))
        
        from argparse import Namespace
        
        # Import DAP components
        from networks.dap import DAP
        
        args = Namespace()
        args.midas_model_type = 'vitl'
        args.fine_tune_type = 'none'
        args.min_depth = 0.001
        args.max_depth = max_depth
        args.train_decoder = False
        
        model = DAP(args)
        return model
        
    except ImportError as e:
        raise ImportError(
            f"Failed to import DAP model: {e}\n\n"
            "Make sure the DAP repository is properly set up:\n"
            "1. Initialize the DAP submodule: git submodule update --init --recursive\n"
            "   (or clone https://github.com/Insta360-Research-Team/DAP into spag4d/dap_arch/DAP)\n"
            "2. Install DAP dependencies: pip install einops opencv-python\n"
            "3. Ensure depth_anything_v2_metric is available in DAP/\n"
        )
    finally:
        # Restore original working directory
        os.chdir(original_cwd)


def is_dap_available() -> bool:
    """Check if DAP architecture is available."""
    try:
        build_dap_model()
        return True
    except (ImportError, Exception):
        return False
