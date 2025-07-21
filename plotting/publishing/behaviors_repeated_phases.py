import os
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

from analysis.clean.freezing_conditions import compute_related_freezing
from core.experiment import Experiment
from core.study import study

import networkx as nx

import utils.plotter as plotter


def get_trials():
    remove_exp_trials = [[10, 21], [2, 11, 12, 13, 15], [8], [], []]

    data = study.data
    data = compute_related_freezing(data)

    exp_trials = []

    for exp in study.list_experiments:
        exp_id = exp.id
        experiment_datetime = exp.experiment_datetime

        trials = []
        remove_trials = remove_exp_trials[exp_id]
        df = pd.read_csv(f"pre_process/manual_trials/{experiment_datetime}.csv")
        for i, row in df.iterrows():
            if row["trial_id"] in remove_trials:
                continue
            trial_data = data[
                (data["experiment_id"] == exp_id)
                & (data["frame"] >= row["start_frame"])
                & (data["frame"] <= row["end_frame"])
            ]
            if trial_data.shape[0] != row["end_frame"] - row["start_frame"] + 1:
                print("Missing frames in trial data")
                print(f"Experiment ID: {exp_id}, Trial ID: {i}")
                print(
                    f"Expected: {row['end_frame'] - row['start_frame'] + 1}, Actual: {trial_data.shape[0]}"
                )
            else:
                trial = {
                    "experiment_id": exp_id,
                    "trial_id": i,
                    "m_trial_id": row["trial_id"],
                    "start_frame": row["start_frame"],
                    "end_frame": row["end_frame"],
                    "data": trial_data,
                }
                trials.append(trial)
        exp_trials.append(trials)

    print("-------------")

    return exp_trials


def mapping(rev, speed_fish):
    if speed_fish < 2:
        return 1
    if rev >= 0:
        return 2
    else:
        return 0


def get_map(exp_trials):
    prob_map = np.zeros(27)
    map_dict = {}
    freezing_encounters = []

    for exp_id, trials in enumerate(exp_trials):
        n_trials = 0
        freezing_a = 0
        freezing_b = 0

        map_dict[exp_id] = []
        for trial in trials:
            z_values = trial["data"]["distance"].to_numpy()
            rev_values = trial["data"]["rev"].to_numpy()
            speed_fish_values = trial["data"]["speed_fish"].to_numpy()

            # Measurements
            ef = np.argmin(z_values)
            max_t = z_values.shape[0]
            td = 10
            ph1 = (0, max(0, ef - td))
            ph2 = (max(0, ef - td), ef + td + 1)
            ph3 = (ef + td + 1, max_t)
            phases = [ph1, ph2, ph3]
            revs_by_avg = []
            speeds_by_avg = []
            for l, phase in enumerate(phases):
                avg_rev = np.mean(rev_values[phase[0] : phase[1]])
                revs_by_avg.append(avg_rev)
                avg_speed = np.mean(speed_fish_values[phase[0] : phase[1]])
                speeds_by_avg.append(avg_speed)

            k = mapping(revs_by_avg[0], speeds_by_avg[0])
            l = mapping(revs_by_avg[1], speeds_by_avg[1])
            m = mapping(revs_by_avg[2], speeds_by_avg[2])

            prob_map[k * 9 + l * 3 + m] += 1
            map_dict[exp_id].append((k, l, m))

            # Freezing
            phase = ph2
            freezing_spd_values = trial["data"]["freezing_spd"].to_numpy()
            freezing_dist_values = trial["data"]["freezing_dist"].to_numpy()

            freezing_spd = freezing_spd_values[phase[0]]  # 0.0415
            freezing_dist = freezing_dist_values[phase[0]]  # 0.028

            if freezing_spd:
                freezing_a += 1
            if freezing_dist:
                freezing_b += 1
            n_trials += 1

        freezing_encounters.append([freezing_a / n_trials, freezing_b / n_trials])

    prob_map = prob_map / np.sum(prob_map)

    return prob_map, map_dict, freezing_encounters


def plot_repeated(probability_map, map_df, freezing_encounters):
    # Labels for behaviors and colors for behaviors
    behavior_labels = ["Attraction", "Slow", "Avoidance"]
    behavior_colors = {
        "Attraction": "#cc99ff",
        "Slow": "#ff9999",
        "Avoidance": "#99ccff",
        "Freezing": "#ffcc66"
    }

    # Creating the decision tree graph using networkx
    G = nx.DiGraph()

    # Step 1: Compute Phase 1 Probabilities
    phase_1_probs = [
        np.sum(probability_map[0:9]),  # Sum of rows 0-8 for Exploration
        np.sum(probability_map[9:18]),  # Sum of rows 9-17 for Freezing
        np.sum(probability_map[18:27]),  # Sum of rows 18-26 for Avoidance
    ]

    # Add Phase 1 nodes with the computed probabilities (with unique identifier)
    for i, behavior in enumerate(behavior_labels):
        node_id = f"Phase 1: {behavior}"  # Unique identifier (hash) for the node
        label = f"{behavior}\n" + r"$p=$ " + f"{phase_1_probs[i]:.2f}"
        G.add_node(node_id, label=label, subset=0, color=behavior_colors[behavior])

    phase_2_probs = [
        np.sum([probability_map[i * 9 : i * 9 + 3] for i in range(3)]),
        np.sum([probability_map[i * 9 + 3 : i * 9 + 6] for i in range(3)]),
        np.sum([probability_map[i * 9 + 6 : i * 9 + 9] for i in range(3)]),
    ]

    for i, behavior in enumerate(behavior_labels):
        node_id = f"Phase 2: {behavior}"
        label = f"{behavior}\n" + r"$p=$ " + f"{phase_2_probs[i]:.2f}"
        G.add_node(node_id, label=label, subset=1, color=behavior_colors[behavior])

    phase_3_probs = [
        np.sum([probability_map[i * 3] for i in range(9)]),
        np.sum([probability_map[i * 3 + 1] for i in range(9)]),
        np.sum([probability_map[i * 3 + 2] for i in range(9)]),
    ]

    for i, behavior in enumerate(behavior_labels):
        node_id = f"Phase 3: {behavior}"
        label = f"{behavior}\n" + r"$p=$ " + f"{phase_3_probs[i]:.2f}"
        G.add_node(node_id, label=label, subset=2, color=behavior_colors[behavior])

    # Phase 2 and Phase 3 probabilities based on transitions
    for i, phase_1_behavior in enumerate(behavior_labels):
        node_id_1 = f"Phase 1: {phase_1_behavior}"
        probs_f1t2 = [
            np.sum(probability_map[i * 9 + j * 3 : i * 9 + j * 3 + 3]) for j in range(3)
        ]
        probs_f1t2 /= np.sum(probs_f1t2)
        for j, phase_2_behavior in enumerate(behavior_labels):
            prob_f1t2 = probs_f1t2[j]
            if prob_f1t2 > 0:
                node_id_2 = f"Phase 2: {phase_2_behavior}"
                G.add_edge(node_id_1, node_id_2, weight=prob_f1t2)

    for i, phase_2_behavior in enumerate(behavior_labels):
        node_id_2 = f"Phase 2: {phase_2_behavior}"
        probs_f2t3 = [
            np.sum([probability_map[i * 3 + j * 9 + k] for j in range(3)])
            for k in range(3)
        ]
        probs_f2t3 /= np.sum(probs_f2t3)
        for j, phase_3_behavior in enumerate(behavior_labels):
            prob_f2t3 = probs_f2t3[j]
            if prob_f2t3 > 0:
                node_id_3 = f"Phase 3: {phase_3_behavior}"
                G.add_edge(node_id_2, node_id_3, weight=prob_f2t3)

    ### FIGURE
    fig, axes = plt.subplots(2, 1, figsize=(6, 8), height_ratios=[0.6, 1])
    # height space
    plt.subplots_adjust(hspace=5.0)

    # Pre-encounter avoidance over trials
    ax = axes[0]

    df = map_df
    for exp_id in df["experiment_id"].unique():
        # Filter the dataframe for each subject (experiment_id)
        subject_data = df[df["experiment_id"] == exp_id]

        # Calculate the cumulative sum of avoidance and normalize by the number of trials
        subject_data["cumulative_avoidance"] = subject_data["avoidance"].cumsum() / (
            subject_data["trial_id"].rank(method="first")
        )

        # Plot the cumulative avoidance percentage for each subject
        ax.plot(
            subject_data["trial_id"],
            subject_data["cumulative_avoidance"],
            label=f"{exp_id+1}",
            color=plotter.colors[exp_id * 2],
        )

    fontsize = 16
    tick_fontsize = 10
    ax.tick_params(axis="both", which="major", labelsize=tick_fontsize)

    ax.set_xlabel("trial number", fontsize=fontsize)
    ax.set_ylabel("cum. prop. pre-avoidance", fontsize=fontsize)
    legend = ax.legend(
        title="Subjects",
        loc="lower right",
        # bbox_to_anchor=(0.65, 0.6),
        edgecolor="white",
        fontsize=8,
    )
    legend.get_title().set_fontsize(10)

    # Phases
    # Draw the graph with multipartite layout (using the subset key)
    ax = axes[1]

    pos = nx.multipartite_layout(
        G, subset_key="subset"
    )  # Arrange nodes by phase (subset)

    # Node attributes
    node_labels = nx.get_node_attributes(G, "label")  # Display 'label' as the node text
    node_colors = [
        G.nodes[n]["color"] for n in G.nodes()
    ]  # Use 'color' attribute for node color

    # Draw the nodes with custom colors and labels
    nx.draw(
        G,
        pos,
        labels=node_labels,
        with_labels=True,
        node_size=2500,
        node_color=node_colors,
        font_size=10,
        font_weight="bold",
        edge_color="gray",
        edgecolors="black",
        arrows=True,
    )

    # Add edge labels with weights
    edge_labels = {(u, v): f'{G[u][v]["weight"]:.2f}' for u, v in G.edges()}
    nx.draw_networkx_edge_labels(
        G,
        pos,
        edge_labels=edge_labels,
        font_size=10,
        font_weight="bold",
        label_pos=0.625,
    )

    # # Get the current axis limits
    # # ax = plt.gca()
    # xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # # Adjust the ylim to create more space at the top
    ax.set_ylim(
        ylim[0] - 0.2, ylim[1] + 0.3
    )  # Add 0.5 units of space above the top nodes

    # Adjust axis for subplot B and add text annotations for phases
    # ax.set_xlim(-0.4, 1.4)
    # ax.set_ylim(-0.2, 1.3)  # Adjusting for more space

    # Add the text at the top of the subplot, relative to the subplot coordinates
    fontsize = 16
    ax.text(
        0.02,
        0.95,
        "pre-encounter",
        horizontalalignment="left",
        fontsize=fontsize,
        fontweight="bold",
        transform=ax.transAxes,
    )
    ax.text(
        0.5,
        0.95,
        "encounter",
        horizontalalignment="center",
        fontsize=fontsize,
        fontweight="bold",
        transform=ax.transAxes,
    )
    ax.text(
        0.98,
        0.95,
        "post-encounter",
        horizontalalignment="right",
        fontsize=fontsize,
        fontweight="bold",
        transform=ax.transAxes,
    )

    plt.axvspan(-0.3, 0.3, color="gray", alpha=0.2)

    # Freezing
    freezing_encounters = np.array(freezing_encounters)
    freezing_pair = np.mean(freezing_encounters, axis=0)
    print(freezing_pair)
    freezing = np.mean(freezing_pair)
    label = "Freezing\n" + r"$p=$ " + f"{freezing:.2f}"
    color = behavior_colors["Freezing"]
    G2 = nx.DiGraph()
    G2.add_node("Freezing", label=label, subset=0, color=color)
    pos2 = {"Freezing": np.array([0.145, 0.35])}
    nx.draw(
        G2,
        pos2,
        labels=nx.get_node_attributes(G2, "label"),
        with_labels=True,
        node_size=2000,
        node_color=color,
        font_size=10,
        font_weight="bold",
        edge_color="gray",
        edgecolors="black",
        arrows=True,
    )

    # # Adjust figure layout
    # plt.subplots_adjust(wspace=0.3)

    # Add edges surrounding the subplot B, using rectangle patch
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    rect = plt.Rectangle(
        (xlim[0], ylim[0]),
        xlim[1] - xlim[0],
        ylim[1] - ylim[0],
        linewidth=1.5,
        edgecolor="black",
        facecolor="none",
    )
    ax.add_patch(rect)

    # Annotate axes with bold A and bold B
    fontsize = 18
    axes[0].text(
        -0.05,
        1.05,
        r"$\textbf{A.}$",
        horizontalalignment="left",
        fontsize=fontsize,
        fontweight="bold",
        transform=axes[0].transAxes,
    )
    axes[1].text(
        -0.05,
        1.03,
        r"\textbf{B.}",
        horizontalalignment="left",
        fontsize=fontsize,
        fontweight="bold",
        transform=axes[1].transAxes,
    )

    plt.tight_layout()

    # # Add edge color around subplot B (not the entire figure)
    # for spine in ax.spines.values():
    #     spine.set_edgecolor('black')
    #     spine.set_linewidth(2)

    plt.savefig(f"{study.plot_output_folder}/PUBLISH/behavior_repeated.pdf")


def create_map_df(exp_trials, map_dict):
    data = {
        "experiment_id": [],
        "trial_id": [],
        "time": [],
        "avoidance": [],
    }

    for exp_id, trials in enumerate(exp_trials):
        for i, trial in enumerate(trials):
            m_trial_id = trial["m_trial_id"]
            start_frame = trial["start_frame"]
            k, l, m = map_dict[exp_id][i]
            data["experiment_id"].append(exp_id)
            data["trial_id"].append(m_trial_id)
            data["time"].append(start_frame)
            avoidance = 1 if k > 1 else 0
            data["avoidance"].append(avoidance)

    df = pd.DataFrame(data)

    return df


if __name__ == "__main__":
    if os.path.exists(f"{study.plot_output_folder}/PUBLISH/map_df.csv"):
        prob_map = np.load(f"{study.plot_output_folder}/PUBLISH/prob_map.npy")
        map_df = pd.read_csv(f"{study.plot_output_folder}/PUBLISH/map_df.csv")
        freez_list = np.load(f"{study.plot_output_folder}/PUBLISH/freez_list.npy")
    else:
        exp_trials = get_trials()
        prob_map, map_dict, freez_list = get_map(exp_trials)
        map_df = create_map_df(exp_trials, map_dict)
        np.save(f"{study.plot_output_folder}/PUBLISH/prob_map.npy", prob_map)
        np.save(f"{study.plot_output_folder}/PUBLISH/freez_list.npy", freez_list)
        map_df.to_csv(f"{study.plot_output_folder}/PUBLISH/map_df.csv", index=False)
    plot_repeated(prob_map, map_df, freez_list)
