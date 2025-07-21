import os
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

import matplotlib.colors as mcolors

def get_data():
    df = pd.read_csv("analysis/clean/measurements.csv")

    # Calculate relative position of the robot to the fish
    df["x_rf"] = df["x_robot"] - df["x_fish"]
    df["y_rf"] = df["y_robot"] - df["y_fish"]

    # Calculate relative angle and adjust by fish's heading
    df["angle_rf"] = np.arctan2(df["y_rf"], df["x_rf"]) - df["heading_fish"]
    df["angle_rf"] = np.mod(df["angle_rf"], 2 * np.pi)

    # Calculate relative angle (fr)
    df["x_fr"] = df["x_fish"] - df["x_robot"]
    df["y_fr"] = df["y_fish"] - df["y_robot"]
    df["angle_fr"] = np.arctan2(df["y_fr"], df["x_fr"]) - df["heading_robot"]
    df["angle_fr"] = np.mod(df["angle_fr"], 2 * np.pi)

    # Test
    # Fish avoidance speed(projection of fish's velocity onto the relative position vector)
    df["v_test"] = df["dx_fish"] * df["x_rf"] + df["dy_fish"] * df["y_rf"]
    df["t_test"] = np.arccos(df["v_test"] / (df["speed_fish"] * df["distance"]))

    # Abs turn
    df["turn_fish"] = np.abs(df["turn_heading_fish"])

    return df


def plot_scatters(df: pd.DataFrame, variable: str, output_filename: str, output_folder: str):
    delta = 1

    # Filter points where cos_app > 0
    filtered_df = df[df['cos_app'] > 0]

    # Get x-axis (distance at time t) and y-axis (robot_speed at time t)
    x_values = filtered_df['distance']
    y_values = filtered_df['speed_robot']

    # Color: variable at t+1
    variable_at_t1 = filtered_df[variable].shift(-delta)

    # Create the scatter plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    # Set normalization ranges based on the variable
    if variable in ["acceleration_fish", "a_avoid"]:
        norm = mcolors.Normalize(vmin=-20, vmax=20)  # Normalizing range from -20 to 20
    elif variable == "turn_heading_fish":
        norm = mcolors.Normalize(vmin=-np.pi, vmax=np.pi)  # Normalizing range from -π to π

    # Create scatter plot
    scatter = ax.scatter(x_values, y_values, c=variable_at_t1, cmap='coolwarm', norm=norm, edgecolor='k')

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label(f'{variable}')

    # Set labels
    ax.set_xlabel('distance')
    ax.set_ylabel('robot Speed')
    # Number of samples
    n_samples = len(x_values)
    ax.set_title(f'{variable} N={n_samples}')

    # Save the plot
    os.makedirs(output_folder, exist_ok=True)  # Ensure the output folder exists
    plt.savefig(os.path.join(output_folder, f'{output_filename}.pdf'))

    plt.close()


if __name__ == "__main__":
    output_sub_folder = "analysis/clean/figs/avoidance_mechanisms"
    if not os.path.exists(output_sub_folder):
        os.makedirs(output_sub_folder)

    delimiter = 45
    list_angles_deg = [i for i in range(0, 360, delimiter)]
    list_angles_rad = [np.deg2rad(i) for i in list_angles_deg]

    df = get_data()
    list_exp_id = df["experiment_id"].unique()

    list_variables = ["acceleration_fish", "a_avoid", "turn_heading_fish"]

    # Loop over experiment IDs first to avoid re-filtering by exp_id for each angle
    for exp_id in list_exp_id:
        df_exp = df[df["experiment_id"] == exp_id].copy()

        for angle_deg, angle_fr in zip(list_angles_deg, list_angles_rad):
            df_angle = df_exp.copy()  # Work on the filtered exp_id data
            
            # Handle circular angles (if angle_fr + delimiter exceeds 2*pi, wrap around)
            angle_fr_upper = angle_fr + np.deg2rad(delimiter)
            if angle_fr_upper > 2 * np.pi:
                # Combine two conditions: from angle_fr to 2*pi and from 0 to the remainder
                df_angle = df_angle[((df_angle["angle_rf"] >= angle_fr) & (df_angle["angle_rf"] < 2 * np.pi)) |
                                    ((df_angle["angle_rf"] >= 0) & (df_angle["angle_rf"] < (angle_fr_upper - 2 * np.pi)))]
            else:
                df_angle = df_angle[(df_angle["angle_rf"] >= angle_fr) & (df_angle["angle_rf"] < angle_fr_upper)]

            # Debugging: Check how many points are in this angle range
            print(f"Angle {angle_deg}-{angle_deg + delimiter}deg: {len(df_angle)} points")

            # Use degrees in the folder name for better readability
            output_folder = f"{output_sub_folder}/angle_rf={angle_deg}-{angle_deg + delimiter}deg"
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)

            for variable in list_variables:
                output_filename = f"var={variable}_exp={exp_id}"
                plot_scatters(df_angle, variable, output_filename, output_folder)
