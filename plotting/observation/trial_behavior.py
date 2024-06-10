import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from core.experiment import Experiment, get_experiment
from utils.parser import parse_args
from utils.plotter import plot_multipdf


def _get_data_point(frame_number, trial_id, fish_velocity, fish_turn, fish_direction):
    if np.isnan(fish_direction).any():
        return None
    fish_orientation = np.arctan2(fish_direction[1], fish_direction[0])
    if fish_velocity == fish_velocity and fish_turn == fish_turn:
        return {
            "frame_number": frame_number,
            "trial_id": trial_id,
            "fish_velocity": fish_velocity,
            "fish_turn": fish_turn,
            "fish_orientation": fish_orientation,
        }
    else:
        return None


def get_data_point(df):
    fish_velocity = df["fish_velocity_batch"]
    fish_turn = df["fish_turn_batch"]
    fish_direction = df["fish_direction_batch"]

    return _get_data_point(
        df["frame_number"],
        df["trial_id"],
        fish_velocity,
        fish_turn,
        fish_direction,
    )


def plot_sep(name, data, ax=None):
    x = [d["fish_velocity"] for d in data]
    y = [d["fish_orientation"] for d in data]

    z = [d["trial_id"] for d in data]

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(x, y)
    # ax.scatter(x, y, c=z, cmap="viridis")
    # cbar = plt.colorbar(ax.scatter(x, y, c=z, cmap="viridis"))
    for i, trial_id in enumerate(z):
        ax.annotate(trial_id, (x[i], y[i]))

    ax.set_xlabel("Fish Velocity")
    ax.set_ylabel("Fish Orientation")

    if ax is None:
        plt.savefig(f"{name}_data.pdf", bbox_inches="tight")


def plot_all(ax, X, Y, Z, A_data, B_data, C_data):
    ax.scatter(X, Y)
    for i, trial_id in enumerate(Z):
        ax.annotate(trial_id, (X[i], Y[i]))
    df_a = pd.DataFrame(A_data)
    df_b = pd.DataFrame(B_data)
    df_c = pd.DataFrame(C_data)

    unique_ids = list(set(Z))
    # not all dataframes have entries for each trial_id
    for trial_id in unique_ids:
        a = df_a[df_a["trial_id"] == trial_id]
        b = df_b[df_b["trial_id"] == trial_id]
        c = df_c[df_c["trial_id"] == trial_id]
        if len(a) == 0 or len(b) == 0 or len(c) == 0:
            continue
        a_fv = a["fish_velocity"].values[0]
        a_fo = a["fish_orientation"].values[0]
        b_fv = b["fish_velocity"].values[0]
        b_fo = b["fish_orientation"].values[0]
        c_fv = c["fish_velocity"].values[0]
        c_fo = c["fish_orientation"].values[0]
        ax.quiver(
            a_fv,
            a_fo,
            b_fv - a_fv,
            b_fo - a_fo,
            angles="xy",
            scale_units="xy",
            scale=1,
            color="r",
        )
        ax.quiver(
            b_fv,
            b_fo,
            c_fv - b_fv,
            c_fo - b_fo,
            angles="xy",
            scale_units="xy",
            scale=1,
            color="r",
        )

    return ax


def plot(A_data, B_data, C_data):
    data = A_data + B_data + C_data
    X = [d["fish_velocity"] for d in data]
    Y = [d["fish_orientation"] for d in data]
    Z = [d["trial_id"] for d in data]

    x_lim = (min(X), max(X))
    y_lim = (min(Y), max(Y))

    fig, axes = plt.subplots(ncols=5, nrows=1, figsize=(40, 8))
    for i, data in enumerate([A_data, B_data, C_data]):
        ax = axes[i]
        plot_sep(f"Data {i}", data, ax)
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)

    ax = axes[3]

    # ax = plot_all(ax, X, Y, Z, A_data, B_data, C_data)

    x = [d["fish_velocity"] for d in B_data]
    y = [d["fish_orientation"] for d in B_data]
    z = [d["trial_id"] for d in B_data]

    ax.scatter(x, y)
    for i, trial_id in enumerate(z):
        ax.annotate(trial_id, (x[i], y[i]))

    df_a = pd.DataFrame(A_data)
    df_b = pd.DataFrame(B_data)
    # quiver from A to B
    unique_ids = list(set(Z))
    for trial_id in unique_ids:
        a = df_a[df_a["trial_id"] == trial_id]
        b = df_b[df_b["trial_id"] == trial_id]
        if len(a) == 0 or len(b) == 0:
            continue
        ax.scatter(a["fish_velocity"], a["fish_orientation"], color="blue", alpha=0.3)
        a_fv = a["fish_velocity"].values[0]
        a_fo = a["fish_orientation"].values[0]
        b_fv = b["fish_velocity"].values[0]
        b_fo = b["fish_orientation"].values[0]
        ax.quiver(
            a_fv,
            a_fo,
            b_fv - a_fv,
            b_fo - a_fo,
            angles="xy",
            scale_units="xy",
            scale=1,
            linestyle="dashed",
            facecolor="blue",
            # facecolor="none",
            alpha=0.25,
            linewidth=0.5,
            # width=0.0001,
            # headwidth=300,
            # headlength=500,
        )

    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)

    ax = axes[4]

    x = [d["fish_velocity"] for d in C_data]
    y = [d["fish_orientation"] for d in C_data]
    z = [d["trial_id"] for d in C_data]

    ax.scatter(x, y)
    for i, trial_id in enumerate(z):
        ax.annotate(trial_id, (x[i], y[i]))

    df_b = pd.DataFrame(B_data)
    df_c = pd.DataFrame(C_data)

    unique_ids = list(set(Z))
    for trial_id in unique_ids:
        b = df_b[df_b["trial_id"] == trial_id]
        c = df_c[df_c["trial_id"] == trial_id]
        if len(b) == 0 or len(c) == 0:
            continue
        ax.scatter(b["fish_velocity"], b["fish_orientation"], color="blue", alpha=0.3)
        b_fv = b["fish_velocity"].values[0]
        b_fo = b["fish_orientation"].values[0]
        c_fv = c["fish_velocity"].values[0]
        c_fo = c["fish_orientation"].values[0]
        ax.quiver(
            b_fv,
            b_fo,
            c_fv - b_fv,
            c_fo - b_fo,
            angles="xy",
            scale_units="xy",
            scale=1,
            linestyle="dashed",
            facecolor="blue",
            alpha=0.25,
            linewidth=0.5,
            # width=0.0001,
            # headwidth=300,
            # headlength=500,
        )
    
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)

    plt.savefig("all_data.pdf", bbox_inches="tight")


def run(experiment):
    results = experiment.results
    results = results[results["robot_distance"] > 0]

    # for every trial, find the frame_number with minimum robot_distance
    min_robot_distance_frame = results.groupby("trial_id")["robot_distance"].idxmin()
    df = results.loc[min_robot_distance_frame]
    df = df.filter(
        [
            "trial_id",
            "frame_number",
            "robot_distance",
            "fish_velocity_batch",
            "fish_turn_batch",
            "fish_direction_batch",
        ]
    )

    min_frame_number = results.groupby("trial_id")["frame_number"].min()
    max_frame_number = results.groupby("trial_id")["frame_number"].max()

    A_data, B_data, C_data = [], [], []
    for row, trial in df.iterrows():
        frame_number = trial["frame_number"]
        trial_id = trial["trial_id"]
        # A-point
        pre_offset = 25
        pre_df = results.loc[
            (results["trial_id"] == trial_id)
            & (results["frame_number"] >= frame_number - pre_offset)
        ]
        if not pre_df.empty:
            pre_df = pre_df.iloc[0]
            A_data_point = get_data_point(pre_df)
            if A_data_point is not None:
                A_data.append(A_data_point)
        # B-point
        B_data_point = get_data_point(trial)
        if B_data_point is not None:
            B_data.append(B_data_point)
        # C-point
        post_offset = 25
        post_df = results.loc[
            (results["trial_id"] == trial_id)
            & (results["frame_number"] >= frame_number + post_offset)
        ]
        if not post_df.empty:
            post_df = post_df.iloc[0]
            C_data_point = get_data_point(post_df)
            if C_data_point is not None:
                C_data.append(C_data_point)

    plot(A_data, B_data, C_data)
    # plot_sep("A", A_data)
    # plot_sep("B", B_data)
    # plot_sep("C", C_data)


if __name__ == "__main__":
    args = parse_args()
    variables = vars(args)

    experiment = get_experiment(**variables)
    run(experiment)
