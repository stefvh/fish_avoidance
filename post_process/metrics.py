import numpy as np

# # COPY FROM robofish-io


def normalize_series(x: np.ndarray) -> np.ndarray:
    """Normalize series.

    Given a series of vectors, return a series of normalized vectors.
    Null vectors are mapped to `NaN` vectors.
    Args:
        x(np.ndarray): The series of vectors.
    Returns:
        np.ndarray: The normalized series.
    """
    return (x.T / np.linalg.norm(x, axis=-1)).T


def calculate_follow(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Calculate the follow metric.

    Given two series of poses - with X and Y coordinates of their
    positions as the first two elements
    return the follow metric from the first to the second series.
    Args:
        a(np.ndarray): The first series of poses.
        b(np.ndarray): The second series of poses.
    Returns:
        np.ndarray: The follow metric.
    """
    a_v = a[1:, :2] - a[:-1, :2]
    b_p = normalize_series(b[:-1, :2] - a[:-1, :2])
    return (a_v * b_p).sum(axis=-1)
