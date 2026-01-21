# spag4d/visual_odometry.py
"""
Spherical Visual Odometry for 360° video stabilization.

Estimates camera rotation between consecutive equirectangular frames
using feature matching and spherical geometry.
"""

import cv2
import numpy as np
import torch
from typing import Optional, Tuple
from scipy.spatial.transform import Rotation


def erp_to_sphere(u: np.ndarray, v: np.ndarray, W: int, H: int) -> np.ndarray:
    """
    Convert equirectangular pixel coordinates to 3D unit sphere coordinates.
    
    Args:
        u: Horizontal pixel coordinates (0 to W-1)
        v: Vertical pixel coordinates (0 to H-1)
        W: Image width
        H: Image height
    
    Returns:
        Points on unit sphere [N, 3] in Y-up coordinate system
    """
    # Convert to spherical coordinates
    # Match spherical_grid: θ decreases from 2π to 0 as u increases from 0 to W
    # theta = (1 - u / W) * 2 * np.pi
    # However, for simple matching, we just need consistent relative rotation.
    # Let's match the grid exactly: 
    #   u=0 -> theta=2pi, u=W -> theta=0.
    #   v=0 -> phi=0, v=H -> phi=pi.
    
    # Note: adding small epsilon to W to avoid wrap-around issues if needed, strictly:
    theta = (1.0 - u / W) * 2 * np.pi
    phi = (v / H) * np.pi
    
    # Convert to Cartesian (Y-up) matching spherical_grid.py
    # rhat = [sin(φ)cos(θ), cos(φ), -sin(φ)sin(θ)]
    x = np.sin(phi) * np.cos(theta)
    y = np.cos(phi)
    z = -np.sin(phi) * np.sin(theta)
    
    return np.stack([x, y, z], axis=-1)


def estimate_rotation_ransac(
    pts1_3d: np.ndarray, 
    pts2_3d: np.ndarray,
    n_iterations: int = 100,
    threshold: float = 0.05
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate rotation matrix between two sets of 3D points using RANSAC.
    
    Uses Kabsch algorithm (SVD-based) for rotation estimation.
    
    Args:
        pts1_3d: Points in frame 1 [N, 3]
        pts2_3d: Corresponding points in frame 2 [N, 3]
        n_iterations: RANSAC iterations
        threshold: Inlier threshold (angular distance)
    
    Returns:
        R: 3x3 rotation matrix
        inlier_mask: Boolean mask of inliers
    """
    best_R = np.eye(3)
    best_inliers = np.zeros(len(pts1_3d), dtype=bool)
    best_count = 0
    
    n_points = len(pts1_3d)
    if n_points < 3:
        return best_R, best_inliers
    
    for _ in range(n_iterations):
        # Sample 3 random correspondences
        idx = np.random.choice(n_points, min(3, n_points), replace=False)
        p1 = pts1_3d[idx]
        p2 = pts2_3d[idx]
        
        # Kabsch algorithm: find optimal rotation
        H = p1.T @ p2
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        
        # Ensure proper rotation (det = 1)
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        
        # Count inliers
        pts2_transformed = (R @ pts1_3d.T).T
        errors = np.linalg.norm(pts2_transformed - pts2_3d, axis=1)
        inliers = errors < threshold
        count = inliers.sum()
        
        if count > best_count:
            best_count = count
            best_inliers = inliers
            best_R = R
    
    # Refine with all inliers
    if best_count >= 3:
        p1 = pts1_3d[best_inliers]
        p2 = pts2_3d[best_inliers]
        H = p1.T @ p2
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        best_R = R
    
    return best_R, best_inliers


class SphericalVisualOdometry:
    """
    Visual Odometry for 360° equirectangular video.
    
    Tracks camera rotation across frames using ORB feature matching
    and spherical geometry.
    """
    
    def __init__(self, n_features: int = 1000):
        """
        Initialize the visual odometry system.
        
        Args:
            n_features: Number of ORB features to detect
        """
        self.orb = cv2.ORB_create(nfeatures=n_features)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
        self.prev_keypoints = None
        self.prev_descriptors = None
        self.prev_shape = None
        
        # Accumulated rotation (world -> camera)
        self.R_accumulated = np.eye(3)
    
    def reset(self):
        """Reset the odometry state."""
        self.prev_keypoints = None
        self.prev_descriptors = None
        self.prev_shape = None
        self.R_accumulated = np.eye(3)
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Process a frame and return the accumulated rotation.
        
        Args:
            frame: BGR or RGB image as numpy array [H, W, 3]
        
        Returns:
            R_accumulated: 3x3 rotation matrix (world -> camera)
        """
        H, W = frame.shape[:2]
        
        # Convert to grayscale for feature detection
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        
        # Detect features
        keypoints, descriptors = self.orb.detectAndCompute(gray, None)
        
        if self.prev_keypoints is None or descriptors is None or len(keypoints) < 10:
            # First frame or not enough features
            self.prev_keypoints = keypoints
            self.prev_descriptors = descriptors
            self.prev_shape = (H, W)
            return self.R_accumulated.copy()
        
        if self.prev_descriptors is None or len(self.prev_keypoints) < 10:
            self.prev_keypoints = keypoints
            self.prev_descriptors = descriptors
            self.prev_shape = (H, W)
            return self.R_accumulated.copy()
        
        # Match features
        try:
            matches = self.bf.match(self.prev_descriptors, descriptors)
        except cv2.error:
            self.prev_keypoints = keypoints
            self.prev_descriptors = descriptors
            self.prev_shape = (H, W)
            return self.R_accumulated.copy()
        
        if len(matches) < 10:
            self.prev_keypoints = keypoints
            self.prev_descriptors = descriptors
            self.prev_shape = (H, W)
            return self.R_accumulated.copy()
        
        # Sort by distance and take best matches
        matches = sorted(matches, key=lambda x: x.distance)[:min(200, len(matches))]
        
        # Extract matched points
        pts1 = np.array([self.prev_keypoints[m.queryIdx].pt for m in matches])
        pts2 = np.array([keypoints[m.trainIdx].pt for m in matches])
        
        # Convert to 3D sphere coordinates
        prev_H, prev_W = self.prev_shape
        pts1_3d = erp_to_sphere(pts1[:, 0], pts1[:, 1], prev_W, prev_H)
        pts2_3d = erp_to_sphere(pts2[:, 0], pts2[:, 1], W, H)
        
        # Estimate rotation
        R_rel, inliers = estimate_rotation_ransac(pts1_3d, pts2_3d)
        
        # Accumulate rotation
        self.R_accumulated = R_rel @ self.R_accumulated
        
        # Update state
        self.prev_keypoints = keypoints
        self.prev_descriptors = descriptors
        self.prev_shape = (H, W)
        
        return self.R_accumulated.copy()
    
    def get_stabilization_rotation(self) -> np.ndarray:
        """
        Get the rotation to apply to splats for stabilization.
        
        Returns:
            R_inv: 3x3 rotation matrix (camera -> world), i.e., inverse of accumulated
        """
        return self.R_accumulated.T  # Transpose = inverse for rotation matrices


def transform_gaussians(
    gaussians: dict,
    rotation_matrix: np.ndarray
) -> dict:
    """
    Transform Gaussian splats by a rotation matrix.
    
    Args:
        gaussians: Dict with 'means', 'scales', 'quats', 'colors', 'opacities'
        rotation_matrix: 3x3 rotation matrix
    
    Returns:
        Transformed gaussians dict
    """
    R = torch.from_numpy(rotation_matrix).float().to(gaussians['means'].device)
    
    # Rotate positions
    means = gaussians['means'] @ R.T  # [N, 3] @ [3, 3]
    
    # Convert rotation matrix to quaternion for composing with existing rotations
    rot = Rotation.from_matrix(rotation_matrix)
    q_rot = rot.as_quat()  # [x, y, z, w] scipy convention
    q_rot = torch.tensor([q_rot[0], q_rot[1], q_rot[2], q_rot[3]], 
                         dtype=torch.float32, device=gaussians['quats'].device)
    
    # Compose quaternions: q_new = q_rot * q_existing
    # Hamilton product
    quats = gaussians['quats']  # [N, 4] in XYZW
    
    # Extract components
    x1, y1, z1, w1 = q_rot[0], q_rot[1], q_rot[2], q_rot[3]
    x2, y2, z2, w2 = quats[:, 0], quats[:, 1], quats[:, 2], quats[:, 3]
    
    # Hamilton product
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    
    new_quats = torch.stack([x, y, z, w], dim=-1)
    
    # Normalize quaternions
    new_quats = new_quats / new_quats.norm(dim=-1, keepdim=True)
    
    return {
        'means': means,
        'scales': gaussians['scales'],  # Scales unchanged
        'quats': new_quats,
        'colors': gaussians['colors'],
        'opacities': gaussians['opacities']
    }
