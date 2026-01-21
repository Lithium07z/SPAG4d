
import sys
print(f"Python: {sys.version}")
try:
    import cv2
    print(f"OpenCV: {cv2.__version__}")
except ImportError as e:
    print(f"OpenCV Import Failed: {e}")

try:
    import scipy
    print(f"Scipy: {scipy.__version__}")
except ImportError as e:
    print(f"Scipy Import Failed: {e}")

try:
    import spag4d.core
    print("SPAG4D Core imported")
except Exception as e:
    print(f"SPAG4D Import Failed: {e}")
