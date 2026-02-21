import torch
from spag4d.da3_model import DA3Model
import numpy as np

def test_da3():
    print("Loading DA3 model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    da3 = DA3Model.load(device=device)
    
    # Create fake ERP image 1024x512
    img = np.zeros((512, 1024, 3), dtype=np.uint8)
    # White box in center
    img[200:300, 400:600] = 255
    
    img_tensor = torch.from_numpy(img).to(device)

    print("\nTesting Equirectangular...")
    depth_eq, _ = da3.predict(img_tensor, projection_mode="equirectangular")
    print(f"Eq output shape: {depth_eq.shape}")

    print("\nTesting Cubemap...")
    depth_cube, _ = da3.predict(img_tensor, projection_mode="cubemap")
    print(f"Cube output shape: {depth_cube.shape}")

    print("\nTesting Icosahedral...")
    depth_ico, _ = da3.predict(img_tensor, projection_mode="icosahedral")
    print(f"Ico output shape: {depth_ico.shape}")

    print("\nSuccess!")

if __name__ == "__main__":
    test_da3()
