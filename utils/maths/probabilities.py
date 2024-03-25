import numpy as np


def compute_joint_conditional_probability(
    x: np.ndarray, y: np.ndarray, z: np.ndarray, x_bins=None, y_bins=None
):
    """
    x: np.ndarray (1D), shape (n_samples,) continuous data
    y: np.ndarray (1D), shape (n_samples,) continuous data
    z: np.ndarray (1D), shape (n_samples,) discrete data
    Compute Pr(X, Y | Z)
    """
    # Discretize x and y
    if x_bins is None:
        x_min = np.min(x)
        x_max = np.max(x)
        x_bins = np.linspace(x_min, x_max, 10)
    if y_bins is None:
        y_min = np.min(y)
        y_max = np.max(y)
        y_bins = np.linspace(y_min, y_max, 10)
    x = np.digitize(x, bins=x_bins)
    y = np.digitize(y, bins=y_bins)

    x_states = np.unique(x)
    y_states = np.unique(y)
    z_states = np.unique(z)

    x_y_z = np.zeros((len(z_states), len(x_states), len(y_states)))

    for z_i in z_states:
        z_indeces = np.where(z == z_i)[0]
        x_z = x[z_indeces]
        y_z = y[z_indeces]

        n_samples = len(z_indeces)

        for x_i in x_states:
            for y_i in y_states:
                x_y_z[z_i, x_i - 1, y_i - 1] = (
                    np.sum((x_z == x_i) & (y_z == y_i)) / n_samples
                )

    return x_y_z


def compute_conditional_occurences(
    x: np.ndarray,
    z: np.ndarray,
    x_bins=None,
):
    """
    x: np.ndarray (1D), shape (n_samples,) continuous data
    z: np.ndarray (1D), shape (n_samples,) discrete data
    """
    # Discretize x and y
    if x_bins is None:
        x_min = np.min(x)
        x_max = np.max(x)
        x_bins = np.linspace(x_min, x_max, 10)
    x = np.digitize(x, bins=x_bins)

    x_states = np.unique(x)
    z_states = np.unique(z)

    q = []

    for z_i in z_states:
        z_indeces = np.where(z == z_i)[0]
        x_z = x[z_indeces]

        n_samples = len(z_indeces)

        q_x = np.zeros(len(x_states))

        for x_i in x_states:
            q_x[x_i - 1] = np.sum((x_z == x_i))

        q.append({"N_Z": n_samples, "Q_X": q_x})

    return q