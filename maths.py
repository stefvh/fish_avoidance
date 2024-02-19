import numpy as np

def angle_between_vectors(a, b):
    inner = np.sum(a[:]*b[:])  # inner = np.inner(a, b)
    norms = np.linalg.norm(a) * np.linalg.norm(b)

    cos = inner / norms
    rad = np.arccos(max(min(cos, 1.0), -1.0))  # rad = np.arccos(np.clip(cos, -1.0, 1.0))

    return np.pi - rad


def angles_absolute_difference(angle1, angle2):
    # is angle1 in the margin of angle2 ?
    # return (2 * np.pi) - (abs(angle1 - angle2) % (2 * np.pi)) < margin or abs(angle1 - angle2) % (
    #             2 * np.pi) < margin
    return abs(np.arctan2(np.sin(angle1 - angle2), np.cos(angle1 - angle2)))