import os
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

from core.experiment import Experiment
from core.study import study


def get_single(df, column, frame, f_t):
    return df[column].values[frame + f_t]


def get_mean(df, column, frame, f_t):
    return df[column].values[frame : frame + f_t].mean()


def get_max(df, column, frame, f_t):
    return df[column].values[frame : frame + f_t].max()


def get_data(experiments: list[Experiment], speed_extract_operator, fa_sr, fd_sf):
    data = {
        "experiment_id": [],
        "speed_robot": [],
        "speed_fish": [],
        "distance": [],
        "cos_apv": [],
    }

    for experiment in experiments:
        rel_mq = experiment.relative_motion_quantities

        for df in rel_mq.df_segments:
            df = df.dropna(subset=["speed_robot", "speed_fish"])
            df.reset_index(drop=True, inplace=True)
            for index, row in df.iterrows():
                if index - fd_sf < 0 or index + fd_sf >= len(df):
                    pass
                else:
                    distance = row["distance"]
                    cos_apv = row["cos_apv"]
                    speed_robot = speed_extract_operator(
                        df, "speed_robot", index, fa_sr
                    )
                    speed_fish = speed_extract_operator(
                        df, "speed_fish", index, fd_sf
                    )

                    data["experiment_id"].append(experiment.id)
                    data["speed_robot"].append(speed_robot)
                    data["speed_fish"].append(speed_fish)
                    data["distance"].append(distance)
                    data["cos_apv"].append(cos_apv)

    return pd.DataFrame(data)


def plot_df(df, seo, fa_sr, fd_sf):
    experiment_ids = [0] # df["experiment_id"].unique()
    for experiment_id in experiment_ids:
        data = df[df["experiment_id"] == experiment_id]
        fig_3d, axes_3d = plt.subplots(
            1, 2, figsize=(12, 6), subplot_kw={"projection": "3d"}
        )
        fig_2d, axes_2d = plt.subplots(1, 2, figsize=(12, 6))

        fig_3d.suptitle(
            f"3D Speed Measurement: {seo.__name__}, Frame advance speed robot: {fa_sr}, Frame delay speed fish: {fd_sf}"
        )
        fig_2d.suptitle(
            f"2D Colormap Speed Measurement: {seo.__name__}, Frame advance speed robot: {fa_sr}, Frame delay speed fish: {fd_sf}"
        )

        c_max = data["speed_fish"].max()
        c_min = data["speed_fish"].min()

        for i, cos_apv in enumerate([0, 1]):
            ax_3d = axes_3d[i]
            ax_2d = axes_2d[i]

            data_filtered = (
                data[data["cos_apv"] > 0] if cos_apv else data[data["cos_apv"] < 0]
            )

            x_var = "speed_robot"
            y_var = "distance"
            z_var = "speed_fish"

            x_values = data_filtered[x_var]
            y_values = data_filtered[y_var]
            z_values = data_filtered[z_var]

            c_values = z_values
            c_values = (c_values - c_min) / (c_max - c_min)

            ax_3d.scatter(
                x_values,
                y_values,
                z_values,
                c=c_values,
                cmap="viridis",
                alpha=0.5,
            )
            scat = ax_2d.scatter(
                x_values,
                y_values,
                c=c_values,
                cmap="viridis",
                alpha=0.5,
                vmin=0,
                vmax=1,
            )
            plt.colorbar(
                scat
            )

            ax_3d.set_title(f"cos_apv {'>' if cos_apv else '<'} 0")
            ax_3d.set_xlabel(x_var)
            ax_3d.set_ylabel(y_var)
            ax_3d.set_zlabel(z_var)

            ax_2d.set_title(f"cos_apv {'>' if cos_apv else '<'} 0")
            ax_2d.set_xlabel(x_var)
            ax_2d.set_ylabel(y_var)

        filepath = f"{study.plot_output_folder}/scatter_by_vars/"
        if not os.path.exists(filepath):
            os.makedirs(filepath)
        suffix = f"{seo.__name__}_a_{fa_sr}_d_{fd_sf}.pdf"
        fig_3d.savefig(
            f"{filepath}/E{experiment_id}_3D_{suffix}.pdf",
            dpi=300,
        )
        fig_2d.savefig(
            f"{filepath}/E{experiment_id}_2D_{suffix}.pdf",
            dpi=300,
        )

        plt.close(fig_3d)
        plt.close(fig_2d)


def run():
    experiments = study.list_experiments

    frame_advances_speed_robot = [1]  # [-25, -20, -15, -10, -5, 0]
    frame_delays_speed_fish = [1, 5, 10, 15, 20, 25]

    speed_extract_operators = [
        # get_single,
        # get_mean,
        get_max,
    ]

    for speed_extract_operator in speed_extract_operators:
        for fa_sr in frame_advances_speed_robot:
            for fd_sf in frame_delays_speed_fish:
                print("Extracting data...")
                df = get_data(experiments, speed_extract_operator, fa_sr, fd_sf)
                print("Plotting data...")
                plot_df(df, speed_extract_operator, fa_sr, fd_sf)


if __name__ == "__main__":
    run()
