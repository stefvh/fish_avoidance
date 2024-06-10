
import numpy as np

def batch_positions_directions(batch_size: int, poses: np.ndarray):
    batch_poses = np.full_like(poses, np.nan)
    for i in range(batch_size, poses.shape[0]):
        batch_poses[i] = np.nanmean(poses[i - batch_size : i], axis=0)

    batch_directions = np.full((poses.shape[0], 2), np.nan)
    for i in range(batch_size, poses.shape[0] - batch_size):
        p = poses[i]
        q = poses[i + batch_size]
        batch_directions[i] = q[:2] - p[:2]
    
    return batch_poses, batch_directions