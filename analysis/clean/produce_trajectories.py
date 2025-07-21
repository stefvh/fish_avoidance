import os
from matplotlib import patches, pyplot as plt
import pandas as pd
import numpy as np
import mpltern 
from scipy.stats import gaussian_kde, lognorm

import matplotlib.colors as mcolors

from analysis.clean.avoidance_mechanisms import get_data
from analysis.one.wrapper_mixed import run_lmer_model


def find_trajectories(
    df: pd.DataFrame, v_app_threshold: float = 2, reaction_delay: int = 8, min_length: int = 10
):
    """
    Find robot approach trajectories based on v_app > threshold, and compute metrics for the fish's reaction.

    Parameters:
    df (pd.DataFrame): Input dataframe containing fish and robot metrics for each frame.
    v_app_threshold (float): Threshold for v_app to define robot approach.
    frame_delay (int): Number of additional frames after the trajectory ends to account for fish reaction.

    Returns:
    pd.DataFrame: A dataframe with one row per trajectory and calculated metrics.
    """

    # Identify frames where v_app > threshold
    above_threshold = df["v_app"] > v_app_threshold

    # Find the start and end indices of each trajectory (consecutive frames where v_app > threshold)
    df["trajectory_id"] = (
        above_threshold != above_threshold.shift(1)
    ).cumsum() * above_threshold
    trajectories = df[df["trajectory_id"] > 0]["trajectory_id"].unique()

    # Prepare output list for storing trajectory data
    output_rows = []

    # Loop over each trajectory to compute metrics
    for traj_id in trajectories:
        traj_df = df[df["trajectory_id"] == traj_id]

        # Check if the trajectory meets the minimum length and consecutive frame requirements based on "frame" column
        if len(traj_df) < min_length or not (traj_df["frame"].diff().iloc[1:] == 1).all():
            continue  # Skip this trajectory if it's too short or not consecutive


        # Extend the trajectory by frame_delay
        last_frame = traj_df.index[-1]
        post_traj_df = df[
            (df.index > last_frame) & (df.index <= last_frame + reaction_delay)
        ]

        full_traj_df = pd.concat([traj_df, post_traj_df])

        # Calculate metrics only for the original trajectory
        min_distance = traj_df["distance"].min()
        avg_robot_speed = traj_df["speed_robot"].mean()
        angles_rad = traj_df["angle_rf"].to_numpy()
        mean_sin = np.mean(np.sin(angles_rad))
        mean_cos = np.mean(np.cos(angles_rad))
        avg_angle = np.arctan2(mean_sin, mean_cos)
        avg_angle = np.mod(avg_angle, 2 * np.pi)

        # Calculate metrics for the full trajectory (original + post-trajectory delay)
        avg_acceleration = full_traj_df["acceleration_fish"].mean()
        max_acceleration = full_traj_df["acceleration_fish"].max()
        max_accel_frame = full_traj_df["acceleration_fish"].idxmax()

        max_turn = full_traj_df["turn_heading_fish"].abs().max()
        max_turn_frame = full_traj_df["turn_heading_fish"].idxmax()

        # distance at max acceleration and max turn (-4)
        min_frame = full_traj_df.index[0]
        max_accel_frame_pre = max(min_frame, max_accel_frame - 4)
        max_turn_frame_pre = max(min_frame, max_turn_frame - 4)
        distance_max_accel = full_traj_df.loc[max_accel_frame_pre : max_accel_frame, "distance"].min()
        distance_max_turn = full_traj_df.loc[max_turn_frame_pre : max_turn_frame, "distance"].min()

        # robot speed at max acceleration and max turn (-4)
        speed_robot_max_accel = full_traj_df.loc[max_accel_frame_pre : max_accel_frame, "speed_robot"].max()
        speed_robot_max_turn = full_traj_df.loc[max_turn_frame_pre : max_turn_frame, "speed_robot"].max()

        # angle_rf at max acceleration and max turn
        angle_rf_max_accel = full_traj_df.loc[
            max_accel_frame_pre, "angle_rf"
        ]
        angle_rf_max_turn = full_traj_df.loc[
            max_turn_frame_pre, "angle_rf"
        ]

        # cos_rev at max acceleration and max turn, and delayed frames

        cos_rev_max_accel = df.loc[
            max_accel_frame + 4, "cos_avoid"
        ]
        cos_rev_max_turn = df.loc[
            max_turn_frame + 4, "cos_avoid"
        ]

        # cos_rev at the end of full_traj_df
        cos_rev_end = full_traj_df["cos_avoid"].iloc[-1]

        # start and end angle
        start_angle = traj_df["angle_rf"].iloc[0]
        end_angle = traj_df["angle_rf"].iloc[-1]

        # Build output row
        row = {
            "experiment_id": traj_df["experiment_id"].iloc[0],
            "trajectory_id": traj_id,
            "start_frame": traj_df.index[0],
            "end_frame": traj_df.index[-1],
            "avg_acceleration": avg_acceleration,
            "max_acceleration": max_acceleration,
            "max_accel_frame": max_accel_frame,
            "max_turn": max_turn,
            "max_turn_frame": max_turn_frame,
            "min_distance": min_distance,
            "avg_robot_speed": avg_robot_speed,
            "avg_angle": avg_angle,
            "distance_max_accel": distance_max_accel,
            "distance_max_turn": distance_max_turn,
            "speed_robot_max_accel": speed_robot_max_accel,
            "speed_robot_max_turn": speed_robot_max_turn,
            "angle_rf_max_accel": angle_rf_max_accel,
            "angle_rf_max_turn": angle_rf_max_turn,
            "cos_rev_max_accel": cos_rev_max_accel,
            "cos_rev_max_turn": cos_rev_max_turn,
            "cos_rev_end": cos_rev_end,
            "start_angle": start_angle,
            "end_angle": end_angle,
        }

        # Append the row to the output list
        output_rows.append(row)

    # Convert output rows to a dataframe
    output_df = pd.DataFrame(output_rows)

    return output_df


def plot_scatters(
    df: pd.DataFrame, variable: str, output_filename: str, output_folder: str
):
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    # Get x-axis (distance at time t) and y-axis (robot_speed at time t)
    x_values = df["min_distance"]
    y_values = df["avg_robot_speed"]

    # Set normalization ranges based on the variable
    cos_rev_variable = ""
    if variable in ["max_acceleration", "avg_acceleration"]:
        norm = mcolors.Normalize(vmin=-20, vmax=20)  # Normalizing range from -20 to 20
        cos_rev_variable = "cos_rev_max_accel"
    elif variable in ["max_turn"]:
        norm = mcolors.Normalize(
            vmin=-np.pi, vmax=np.pi
        )  # Normalizing range from -π to π
        cos_rev_variable = "cos_rev_max_turn"

    c_values = df[variable]

    # Create scatter plot
    scatter = ax.scatter(
        x_values, y_values, c=c_values, cmap="coolwarm", norm=norm, edgecolor="k"
    )

    # Add text values for cos_rev values
    for i, txt in enumerate(df[cos_rev_variable]):
        ax.annotate(f"{txt[0]:.2f}", (x_values.iloc[i], y_values.iloc[i]))

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label(f"{variable}")

    # Set labels
    ax.set_xlabel("distance")
    ax.set_ylabel("robot speed")
    # Number of samples
    n_samples = len(x_values)
    ax.set_title(f"{variable} N={n_samples}")

    # Save the plot
    os.makedirs(output_folder, exist_ok=True)  # Ensure the output folder exists
    plt.savefig(os.path.join(output_folder, f"{output_filename}.pdf"))

    plt.close()


def plot_colored_triangles(
    df,
    x_column,
    y_column,
    max_accel_column,
    max_turn_column,
    cos_rev_column,
    output_filename,
    output_folder,
):
    # Define colormaps and normalization for each variable
    cmap_accel = plt.get_cmap("RdBu")
    norm_accel = mcolors.Normalize(vmin=-20, vmax=20)

    cmap_turn = plt.get_cmap("PuOr")
    norm_turn = mcolors.Normalize(vmin=-np.pi, vmax=np.pi)

    cmap_cos_rev = plt.get_cmap("PiYG")
    norm_cos_rev = mcolors.Normalize(vmin=-1, vmax=1)

    #
    size = 1.0
    bottom_x_offset = size * np.sqrt(3) / 2
    bottom_y_offset = size / 2

    fig, ax = plt.subplots(figsize=(10, 10))

    for _, row in df.iterrows():
        x = row[x_column]
        y = row[y_column]

        # Normalize values to get colors
        color_accel = cmap_accel(norm_accel(row[max_accel_column]))
        color_turn = cmap_turn(norm_turn(row[max_turn_column]))
        color_cos_rev = cmap_cos_rev(norm_cos_rev(row[cos_rev_column]))

        # Define vertices for the main equilateral triangle centered at (x, y)
        top_vertex = (x, y + size)
        bottom_left_vertex = (x - bottom_x_offset, y - bottom_y_offset)
        bottom_right_vertex = (x + bottom_x_offset, y - bottom_y_offset)

        # Center of the triangle (centroid of an equilateral triangle)
        center_vertex = (x, y)

        # Create three triangular sections by connecting the center to each edge
        # Section 1 (Top) - max_acceleration
        triangle1 = patches.Polygon(
            [top_vertex, center_vertex, bottom_right_vertex],
            closed=True,
            color=color_accel,
        )

        # Section 2 (Bottom Left) - max_turn
        triangle2 = patches.Polygon(
            [bottom_left_vertex, center_vertex, top_vertex],
            closed=True,
            color=color_turn,
        )

        # Section 3 (Bottom Right) - cos_rev
        triangle3 = patches.Polygon(
            [bottom_right_vertex, center_vertex, bottom_left_vertex],
            closed=True,
            color=color_cos_rev,
        )

        # Add triangles to the plot
        ax.add_patch(triangle1)
        ax.add_patch(triangle2)
        ax.add_patch(triangle3)

        # # Create the full triangle outline
        # outline_triangle = patches.Polygon([top_vertex, bottom_left_vertex, bottom_right_vertex], closed=True,
        #                                 edgecolor='black', facecolor='none', linewidth=0.5)

        # # Add the outline to the plot before adding the inner triangles
        # ax.add_patch(outline_triangle)

        ax.scatter(x, y, color="black", s=1, zorder=0)

    ax.set_aspect("equal")

    # Add colorbars for each variable
    sm_accel = plt.cm.ScalarMappable(cmap=cmap_accel, norm=norm_accel)
    sm_accel.set_array([])
    cbar_accel = plt.colorbar(sm_accel, ax=ax, fraction=0.02, pad=0.04)
    cbar_accel.set_label(max_accel_column)

    sm_turn = plt.cm.ScalarMappable(cmap=cmap_turn, norm=norm_turn)
    sm_turn.set_array([])
    cbar_turn = plt.colorbar(sm_turn, ax=ax, fraction=0.02, pad=0.08)
    cbar_turn.set_label(max_turn_column)

    sm_cos_rev = plt.cm.ScalarMappable(cmap=cmap_cos_rev, norm=norm_cos_rev)
    sm_cos_rev.set_array([])
    cbar_cos_rev = plt.colorbar(sm_cos_rev, ax=ax, fraction=0.02, pad=0.12)
    cbar_cos_rev.set_label(cos_rev_column)

    # Set labels
    ax.set_xlabel("distance")
    ax.set_ylabel("robot speed")
    # Number of samples
    n_samples = df.shape[0]
    ax.set_title(f"N={n_samples}")

    # Save the plot
    os.makedirs(output_folder, exist_ok=True)  # Ensure the output folder exists
    fig.savefig(os.path.join(output_folder, f"{output_filename}.pdf"))
    plt.close()


def filter_angle_df(df_angle, angle_fr, delimiter, angle_var):
    angle_fr_upper = angle_fr + np.deg2rad(delimiter)
    angle_fr_upper = angle_fr + np.deg2rad(45)
    
    if angle_fr_upper > 2 * np.pi:
        # If angle_fr + 45 degrees exceeds 360, wrap around
        df_angle = df_angle[(df_angle[angle_var] >= angle_fr) | (df_angle[angle_var] < (angle_fr_upper - 2 * np.pi))]
    else:
        # Otherwise, straightforward range filter
        df_angle = df_angle[(df_angle[angle_var] >= angle_fr) & (df_angle[angle_var] < angle_fr_upper)]
    
    return df_angle


def run_analysis():
    df = get_data()

    output_sub_folder = "analysis/clean/figs/avoidance_trajectories"
    if not os.path.exists(output_sub_folder):
        os.makedirs(output_sub_folder)

    list_exp_id = df["experiment_id"].unique().tolist()
    list_exp_id.append("all")
    list_variables = ["max_acceleration", "max_turn", "avg_acceleration"]

    delimiter = 45
    list_angles_deg = [i for i in range(0, 360, delimiter)]
    list_angles_rad = [np.deg2rad(i) for i in list_angles_deg]

    # Loop over experiment IDs
    for k, exp_id in enumerate(list_exp_id):
        if k < len(list_exp_id) - 1:
            df_exp = df[df["experiment_id"] == exp_id].copy()
        else:
            df_exp = df.copy()
        df_traj = find_trajectories(df_exp)

        for angle_deg, angle_fr in zip(list_angles_deg, list_angles_rad):
            output_folder = (
                f"{output_sub_folder}/angle_rf={angle_deg}-{angle_deg + delimiter}deg"
            )
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)

            # scatter_variables_separately(
            #     df_traj, exp_id, list_variables, angle_fr, delimiter, output_folder
            # )

            df_angle = df_traj.copy()
            df_angle = filter_angle_df(df_angle, angle_fr, delimiter, "avg_angle")
            output_filename = f"comb_exp={exp_id}"
            plot_colored_triangles(
                df_angle,
                "min_distance",
                "avg_robot_speed",
                "max_acceleration",
                "max_turn",
                "cos_rev_end",
                output_filename,
                output_folder,
            )


def scatter_variables_separately(df_traj, exp_id, list_variables, angle_fr, delimiter, output_folder):
    for variable in list_variables:
        df_angle = df_traj.copy()  # Work on the filtered exp_id data

        if variable in ["max_acceleration"]:
            angle_var = "angle_rf_max_accel"
        elif variable in ["max_turn"]:
            angle_var = "angle_rf_max_turn"
        elif variable in ["avg_acceleration"]:
            angle_var = "avg_angle"

        # Handle circular angles (if angle_fr + delimiter exceeds 2*pi, wrap around)
        df_angle = filter_angle_df(df_angle, angle_fr, delimiter, angle_var)

        # # Debugging: Check how many points are in this angle range
        # print(f"Angle {angle_deg}-{angle_deg + delimiter}deg: {len(df_angle)} points")

        # Use degrees in the folder name for better readability
        output_filename = f"var={variable}_exp={exp_id}"
        plot_scatters(df_angle, variable, output_filename, output_folder)


def plot_ternary_density(
    df, 
    left_axis="max_acceleration", 
    right_axis="max_turn", 
    bottom_axis="cos_rev_end", 
    output_filename="ternary_density", 
    output_folder="."
):
    """
    Create a ternary scatter plot with density overlay.

    Parameters:
    - df (pd.DataFrame): Input DataFrame.
    - distance_threshold (float): Threshold to discretize distance.
    - speed_threshold (float): Threshold to discretize avg_robot_speed.
    - left_axis (str): Column name for the left axis in the ternary plot.
    - right_axis (str): Column name for the right axis in the ternary plot.
    - bottom_axis (str): Column name for the bottom axis in the ternary plot.
    - output_filename (str): Name of the output file.
    - output_folder (str): Folder to save the plot.
    """
    # Prepare ternary data by normalizing each row to sum up to 1 for ternary plotting
    ternary_data = df[[left_axis, right_axis, bottom_axis]]
    ternary_data = ternary_data.div(ternary_data.sum(axis=1), axis=0)

    # Extract ternary components
    left_values = ternary_data[left_axis].values
    right_values = ternary_data[right_axis].values
    bottom_values = ternary_data[bottom_axis].values

    # Calculate density for ternary plot points
    # xy = np.vstack([left_values, right_values, bottom_values])
    # density = gaussian_kde(xy)(xy)  # Kernel density estimation for density color overlay

    # Plot setup
    fig, ax = plt.subplots(subplot_kw=dict(projection='ternary'), figsize=(10, 10))
    sc = ax.scatter(left_values, right_values, bottom_values)
    
    # Overlay scatter points in black for visibility
    ax.scatter(left_values, right_values, bottom_values, color="black", s=10, alpha=0.6)

    # Add colorbar to show density levels
    # cbar = plt.colorbar(sc, ax=ax, shrink=0.75, pad=0.1)
    # cbar.set_label("Density")

    # Set axis labels based on selected columns
    ax.set_tlabel(left_axis)   # Top label
    ax.set_llabel(right_axis)  # Left label
    ax.set_rlabel(bottom_axis) # Right label

    # Give each axis a color, and color the ticks and label respectively
    axis_colors = {
        'left': 'blue',   # Color for left axis (max_acceleration)
        'right': 'green', # Color for right axis (max_turn)
        'bottom': 'purple' # Color for bottom axis (cos_rev_end)
    }
    
    # Customize axis label colors
    ax.set_tlabel(left_axis, color='C0')
    ax.set_llabel(right_axis, color='C1')
    ax.set_rlabel(bottom_axis, color='C2')

    # Adjust label position by setting pad values (negative to move inward)
    ax.taxis.set_tick_params(tick2On=True, colors='C0',)
    ax.laxis.set_tick_params(tick2On=True, colors='C1',)
    ax.raxis.set_tick_params(tick2On=True, colors='C2',)

    # Save plot
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, f"{output_filename}.pdf")
    plt.savefig(output_path)
    plt.close()
    print(f"Ternary plot saved at: {output_path}")


def categorize_strategy(df):
    # Define the threshold categories
    acc_conditions = [
        (df["max_acceleration"] < -2, 'A'),
        (df["max_acceleration"].between(-2, 2), 'B'),
        (df["max_acceleration"] > 2, 'C')
    ]
    turn_conditions = [
        (df["max_turn"].abs() <= np.deg2rad(15), 'D'),
        (df["max_turn"].abs() > np.deg2rad(15), 'E')
    ]
    cos_conditions = [
        (df["cos_rev_end"] < -0.5, 'F'),
        (df["cos_rev_end"].between(-0.5, 0.5), 'G'),
        (df["cos_rev_end"] > 0.5, 'H')
    ]
    
    # Initialize an empty column for strategy labels
    df["strategy_label"] = ""
    
    # Generate all combinations (3 x 2 x 3 = 18 combinations)
    strategy_labels = []
    combination_index = 1
    for acc_cond, acc_label in acc_conditions:
        for turn_cond, turn_label in turn_conditions:
            for cos_cond, cos_label in cos_conditions:
                label = f"Strategy_{combination_index}"
                strategy_labels.append((label, acc_label, turn_label, cos_label))
                
                # Assign label to dataframe based on the combined conditions
                df.loc[acc_cond & turn_cond & cos_cond, "strategy_label"] = label
                combination_index += 1
    
    # Print each strategy label with the corresponding conditions
    print("Strategy Labels Mapping:")
    for label, acc_label, turn_label, cos_label in strategy_labels:
        print(f"{label}: max_acceleration={acc_label}, max_turn={turn_label}, cos_rev_end={cos_label}")
    
    return df

def plot_density_by_strategy(df, delimiter=45, title="", output_folder="."):
    delimiter = 45
    list_angles_deg = [i for i in range(0, 360, delimiter)]
    list_angles_rad = [np.deg2rad(i) for i in list_angles_deg]

    fig, axes = plt.subplots(len(list_angles_deg), 1, figsize=(8, 2 * len(list_angles_deg)), sharex=True)
    fig.subplots_adjust(hspace=0)  # Remove vertical spacing between subplots

    for idx, angle_deg in enumerate(list_angles_deg):
        # Filter data for the specific angle range
        angle_min = np.deg2rad(angle_deg)
        angle_max = np.deg2rad(angle_deg + delimiter)
        df_angle = df[(df["avg_angle"] >= angle_min) & (df["avg_angle"] < angle_max)]
        
        ax = axes[idx]
 
        # Define the complete list of strategy labels in the correct sequence
        all_strategies = [f"Strategy_{i}" for i in range(1, 19)]
        
        # Count occurrences of each strategy within the angle range, ensuring all strategies are included
        strategy_counts = df_angle["strategy_label"].value_counts().reindex(all_strategies, fill_value=0)
        
        # Plot histogram with strategy labels on the x-axis and counts on the y-axis
        ax.bar(strategy_counts.index, strategy_counts.values, alpha=0.7, color='blue')
        
        ax.set_ylabel(f"{angle_deg}-{angle_deg + delimiter}°")
        ax.set_xticks(range(len(all_strategies)))  # Set x-ticks to the indices of all strategies
        ax.set_xticklabels(all_strategies, rotation=45)  # Label x-ticks with strategy labels, rotated for readability

        if idx == 0:
            ax.legend(["Strategy Counts"])

        ax.set_ylabel(f"{angle_deg}-{angle_deg + delimiter}°")
    
    # Set common x-axis label
    axes[-1].set_xlabel("Average Angle (Radians)")

    # Set common title for the entire figure
    fig.suptitle(title, fontsize=16)
    
    # Save the figure to the PDF
    fig.savefig(os.path.join(output_folder, "density_by_strategy.pdf"))
    plt.close(fig)
    print(f"Density plots saved")


def run_strategy_analysis():
    df = get_data()
    df_traj = find_trajectories(df)

    output_sub_folder = "analysis/clean/figs/avoidance_trajectories"
    if not os.path.exists(output_sub_folder):
        os.makedirs(output_sub_folder)

    delimiter = 45
    list_angles_deg = [i for i in range(0, 360, delimiter)]
    list_angles_rad = [np.deg2rad(i) for i in list_angles_deg]

    threshold_distance = 20
    threshold_robot_speed = 10

    # Define conditions and folder naming
    conditions = [
        {"distance": "<", "speed": "<", "folder_name": f"distance_<{threshold_distance}_speed_<{threshold_robot_speed}"},
        {"distance": "<", "speed": ">=", "folder_name": f"distance_<{threshold_distance}_speed_>={threshold_robot_speed}"},
        {"distance": ">=", "speed": "<", "folder_name": f"distance_>={threshold_distance}_speed_<{threshold_robot_speed}"},
        {"distance": ">=", "speed": ">=", "folder_name": f"distance_>={threshold_distance}_speed_>={threshold_robot_speed}"},
    ]

    # Loop over conditions and create filtered DataFrames
    for condition in conditions:
        # Create output folder for each distance/speed combination
        condition_folder = os.path.join(output_sub_folder, condition["folder_name"])
        os.makedirs(condition_folder, exist_ok=True)

        # Filter based on distance and speed thresholds
        if condition["distance"] == "<":
            df_distance_filtered = df_traj[df_traj["min_distance"] < threshold_distance]
        else:
            df_distance_filtered = df_traj[df_traj["min_distance"] >= threshold_distance]
        
        if condition["speed"] == "<":
            df_filtered = df_distance_filtered[df_distance_filtered["avg_robot_speed"] < threshold_robot_speed]
        else:
            df_filtered = df_distance_filtered[df_distance_filtered["avg_robot_speed"] >= threshold_robot_speed]

        df_cat = categorize_strategy(df_filtered.copy())
        plot_density_by_strategy(df_cat, delimiter=45, title=condition["folder_name"], output_folder=condition_folder)


def run_continuous_correlation_analysis(df, delimiter=45):
    # Standardize the relevant columns
    df = standardize_columns(df, ["max_acceleration", "max_turn", "cos_rev_end"])

    list_angles_deg = [i for i in range(0, 360, delimiter)]
    correlation_results = {}

    for angle_deg in list_angles_deg:
        # Filter data for each angle range
        angle_min = np.deg2rad(angle_deg)
        angle_max = np.deg2rad(angle_deg + delimiter)
        df_angle = df[(df["avg_angle"] >= angle_min) & (df["avg_angle"] < angle_max)]

        # Calculate correlation within this angle range using continuous data
        if len(df_angle) > 1:  # Ensure enough data points
            corr_matrix = df_angle[["max_acceleration", "max_turn", "cos_rev_end"]].corr()
            correlation_results[f"{angle_deg}-{angle_deg + delimiter}°"] = corr_matrix

    return correlation_results

def standardize_columns(df, columns):
    # Standardize specified columns in the DataFrame
    df[columns] = df[columns].apply(lambda x: (x - x.mean()) / x.std())
    return df

def spearman_correlation_analysis(df, delimiter=45, high_acceleration_only=False):
    list_angles_deg = [i for i in range(0, 360, delimiter)]
    correlation_results = {}

    for angle_deg in list_angles_deg:
        # Filter data for each angle range
        angle_min = np.deg2rad(angle_deg)
        angle_max = np.deg2rad(angle_deg + delimiter)
        df_angle = df[(df["avg_angle"] >= angle_min) & (df["avg_angle"] < angle_max)]

        # Optional: Focus on high acceleration cases only
        if high_acceleration_only:
            acc_threshold = df["max_acceleration"].quantile(0.75)  # e.g., top 25% acceleration values
            df_angle = df_angle[df_angle["max_acceleration"] > acc_threshold]

        # Calculate Spearman correlation within this angle range
        if len(df_angle) > 1:  # Ensure enough data points
            corr_matrix = df_angle[["max_acceleration", "max_turn", "cos_rev_end"]].corr(method="spearman")
            correlation_results[f"{angle_deg}-{angle_deg + delimiter}°"] = corr_matrix

    return correlation_results


def plot_max_turn_density(df):
    # Assuming df["max_turn"] contains the data you want to plot
    data = df["max_turn"]

    # Calculate the density
    density = gaussian_kde(data)
    x_vals = np.linspace(min(data), max(data), 1000)  # Define x-axis range for smooth plotting
    y_vals = density(x_vals)

    # Plot with matplotlib
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.fill_between(x_vals, y_vals, color="blue", alpha=0.5)
    ax.plot(x_vals, y_vals, color="blue")

    # Labeling the axes
    ax.set_xlabel("Max Turn")
    ax.set_ylabel("Density")
    ax.set_title("Density Plot of Max Turn Values")

    percentile = np.percentile(data, 87.5)
    print(f"The 75th percentile (start of the tail) is at max turn = {percentile:.2f}")
    ax.axvline(percentile, color="black", linestyle="-", linewidth=1, alpha=0.5)

    # Step 1: Fit distributions to the data
    # Log-normal fit
    shape, loc, scale = lognorm.fit(data)
    # Indicate tail start points
    mean_lognorm, std_dev_lognorm = lognorm.stats(shape, loc=loc, scale=scale, moments="mv")
    tail_start_lognorm = mean_lognorm + 0.5 * std_dev_lognorm # Start of the tail (2 std deviations from mean)
    ax.axvline(tail_start_lognorm, color="purple", linestyle="--", label="Tail Start (2 std devs - Log-normal)")

    # Save the plot
    output_folder = "analysis/clean/figs/avoidance_trajectories"
    os.makedirs(output_folder, exist_ok=True)
    fig.savefig(os.path.join(output_folder, "max_turn_density.pdf"))
    plt.close()
    print("Density plot saved")

    # Find the x value at the peak of the density
    max_density_index = np.argmax(y_vals)
    max_density_x = x_vals[max_density_index]
    max_density_y = y_vals[max_density_index]

    # Define a range around the max density point to count samples (e.g., within ±5% of the peak x value)
    tolerance = 0.05 * (x_vals[-1] - x_vals[0])
    sample_count = ((data >= max_density_x - tolerance) & (data <= max_density_x + tolerance)).sum()

    print(f"Max density occurs at x = {max_density_x:.2f} with density = {max_density_y:.2f}")
    print(f"Number of samples within ±{tolerance:.2f} of max density x value: {sample_count}")


def plot_scatter_max_turn_function_of_var(df, variable):
    # x : variable
    # y : max_turn

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    x_values = df[variable].values
    y_values = df["max_turn"].values

    ax.scatter(x_values, y_values, alpha=0.5)

    ax.set_xlabel(variable)
    ax.set_ylabel("max_turn")

    # Save the plot
    output_folder = "analysis/clean/figs/avoidance_trajectories"
    os.makedirs(output_folder, exist_ok=True)
    fig.savefig(os.path.join(output_folder, f"max_turn_vs_{variable}.pdf"))
    plt.close()

def plot_threed_scatter_max_turn(df):
    # Create a 3D scatter plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    x = df["distance_max_turn"]
    y = df["speed_robot_max_turn"]
    z = df["max_turn"]

    # Plotting the scatter
    scatter = ax.scatter(x, y, z, c=z, cmap='viridis', alpha=0.8)
    ax.set_xlabel("Distance")
    ax.set_ylabel("Robot Speed")
    ax.set_zlabel("Max Turn")
    ax.set_title("3D Scatter Plot of Max Turn, Distance, and Robot Speed")

    # Add color bar to indicate max_turn values
    cbar = plt.colorbar(scatter, ax=ax, pad=0.1)
    cbar.set_label("Max Turn")

    # Print number of points where z > 1.0
    n_points = (z > 1.0).sum()
    print(f"Number of points with max_turn > 1.0: {n_points}")
    n_total_points = len(z)
    print(f"Total number of points: {n_total_points}")

    # Save the plot
    output_folder = "analysis/clean/figs/avoidance_trajectories"
    os.makedirs(output_folder, exist_ok=True)
    fig.savefig(os.path.join(output_folder, f"max_turn_3d.pdf"))
    plt.close()


def plot_start_end_angles(df):
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.scatter(df["start_angle"], df["end_angle"])
    # labels
    ax.set_xlabel("Start angle")
    ax.set_ylabel("End angle")

    ax.axvline(np.pi, color="black", linestyle="--")
    ax.axhline(np.pi, color="black", linestyle="--")

    # do a y = x line
    ax.plot([0, 2 * np.pi], [0, 2 * np.pi], color="black", linestyle="-")

    output_folder = "analysis/clean/figs/avoidance_trajectories"
    os.makedirs(output_folder, exist_ok=True)
    fig.savefig(os.path.join(output_folder, f"traj_start-end_angles.pdf"))
    plt.close()

    # add subplots for marginal density of start angle and also for end angle
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    # start angle
    ax.hist(df["start_angle"], bins=20, density=True, color="blue", alpha=0.5, label="Start angle")
    ax.set_xlabel("Angle")
    ax.set_ylabel("Density")
    ax.hist(df["end_angle"], bins=20, density=True, color="orange", alpha=0.5, label="End angle")

    ax.legend()

    fig.savefig(os.path.join(output_folder, f"traj_start-end_angles_hist.pdf"))
    plt.close()


if __name__ == "__main__":
    df = get_data()
    df_traj = find_trajectories(df)

    plot_start_end_angles(df_traj)

    # # Only keep trajectories where distance_max_turn < 20
    # df_traj = df_traj[df_traj["distance_max_turn"] < 40]
    # df_traj = df_traj[df_traj["max_turn"] > 1.0]

    # # plot_threed_scatter_max_turn(df_traj)

    # output = run_lmer_model(
    #     df_traj, 
    #     response_var="max_turn",
    #     fixed_effects=["distance_max_turn", "speed_robot_max_turn", "angle_rf_max_turn"],
    #     random_effects=["experiment_id"]
    # )
    # print(output)



    # variables = ["distance_max_turn", "speed_robot_max_turn", "angle_rf_max_turn", "cos_rev_max_turn"]
    # for variable in variables:
    #     plot_scatter_max_turn_function_of_var(df_traj, variable)