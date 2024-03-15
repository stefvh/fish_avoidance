
import numpy as np


def transform_corner_data(corners,
                          width_cm,
                          height_cm):
    new_corners = np.zeros((len(corners), 2))
    for i, corner in enumerate(corners):
        new_corners[i] = np.array([corner['x'], corner['y']])
    new_corners[:, 0] *= width_cm
    new_corners[:, 1] *= height_cm

    return new_corners