from matplotlib import pyplot as plt
from matplotlib.patches import FancyArrowPatch
import numpy as np
import pandas as pd

from core.experiment import Experiment
from core.study import study
from utils.plotter import plot_multipdf

import seaborn as sns
import networkx as nx

def get_trials():
    remove_exp_trials = [
        [10, 21],
        [2, 11, 12, 13, 15],
        [8],
        [],
        []
    ]
    data = study.data

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


def plot_trial(ax_row, i, j, trial, columns):
    colors = [
        "blue",
        "orange",
        "green",
        "red",
        "purple",
    ]
    ylims = [
        (0, 40),
        (0, 20),
        (0, 20),
        (0, 25),
        (-np.pi, np.pi),
    ]

    for k, column in enumerate(columns):
        ax = ax_row[k]

        frames = trial["data"]["frame"].to_numpy()
        y_values = trial["data"][column].to_numpy()

        if column == "distance":
            ef = np.argmin(y_values)

        if column == "apv":
            ax.plot(frames, y_values, color="black")
            begin = 5
            ax.vlines(
                frames[begin],
                ymin=0,
                ymax=40,
                color="red",
                linestyle="--",
            )
            # find the first frame where apv < 1.0
            ef = begin
            for i, apv in enumerate(y_values[begin:]):
                if apv < 1.0:
                    ef = begin + i
                    break
            y_values[ef:] = 0

        if column in ["distance", "apv"]:
            # Measurements
            max_t = y_values.shape[0]
            td = 10
            ph1 = (0, max(0, ef - td))
            ph2 = (max(0, ef - td), ef + td + 1)
            ph3 = (ef + td + 1, max_t)
            phases = [ph1, ph2, ph3]
            rev_values = trial["data"]["rev"].to_numpy()
            for l, phase in enumerate(phases):
                max_rev = np.max(rev_values[phase[0]:phase[1]])
                avg_rev = np.mean(rev_values[phase[0]:phase[1]])
                ax.text(
                    0.5,
                    0.2 + l * 0.15,
                    f"{l}; Max: {max_rev:.1f}, Avg: {avg_rev:.1f}",
                    horizontalalignment="center",
                    verticalalignment="center",
                    transform=ax.transAxes  # This ensures the text positioning is relative to the axes
                )
        
        if column == "turn_fish":
            distances = trial["data"]["distance"].to_numpy()
            ef = np.argmin(distances)
            td = 10
            ph2 = (max(0, ef - td), ef + td + 1)
            # overlay with gray area this ph2
            start_frame = frames[ph2[0]]
            end_frame = frames[ph2[-1]]
            ax.axvspan(start_frame, end_frame, color='gray', alpha=0.2)

        ax.plot(frames, y_values, color=colors[k])
        if j == 0:
            ax.set_title(column)
        ax.set_ylabel(trial["m_trial_id"])
        ax.set_ylim(ylims[k])


def plot_trials_per_exp(exp_trials, output_folder=None, columns=None):
    if output_folder is None:
        output_folder = study.plot_output_folder
    if columns is None:
        columns = [
            "distance",
            "speed_fish",
            "rev",
            "apv",
            "turn_fish",
        ]
    
    for exp_id, trials in enumerate(exp_trials):
        num_trials = len(trials)
        plot_multipdf(
            num_trials,
            5,
            5,
            lambda ax_row, i, j: plot_trial(
                ax_row=ax_row,
                i=i,
                j=j,
                trial=trials[i + j],
                columns=columns,
            ),
            f"{output_folder}/trials_overview_{exp_id}.pdf",
        )


def mapping(rev):
    if rev < -0.2:
        return 0
    elif -0.2 < rev < 2:
        return 1
    else:
        return 2
    
def mapping_r_sf(rev, speed_fish):
    if speed_fish < 2:
        return 1
    if rev >= 0:
        return 2
    else:
        return 0

def mapping_names(rev):
    if rev < 0:
        return "EXPLORATION"
    elif 0 < rev < 2:
        return "FREEZING"
    else:
        return "ESCAPE"

def behavior_maps_by_z(exp_trials, z_name="distance", op="max"):
    prob_map = np.zeros(27)

    for exp_id, trials in enumerate(exp_trials):
        for trial in trials:
            z_values = trial["data"][z_name].to_numpy()
            rev_values = trial["data"]["rev"].to_numpy()
            speed_fish_values = trial["data"]["speed_fish"].to_numpy()

            if z_name == "distance":
                ef = np.argmin(z_values)
            if z_name == "apv":
                begin = 5
                ef = begin
                for i, apv in enumerate(z_values[begin:]):
                    if apv < 1.0:
                        ef = begin + i
                        break

            # Measurements
            max_t = z_values.shape[0]
            td = 10
            ph1 = (0, max(0, ef - td))
            ph2 = (max(0, ef - td), ef + td + 1)
            ph3 = (ef + td + 1, max_t)
            phases = [ph1, ph2, ph3]
            revs_by_max = []
            revs_by_avg = []
            speeds_by_avg = []
            for l, phase in enumerate(phases):
                max_rev = np.max(rev_values[phase[0]:phase[1]])
                avg_rev = np.mean(rev_values[phase[0]:phase[1]])
                revs_by_max.append(max_rev)
                revs_by_avg.append(avg_rev)
                avg_speed = np.mean(speed_fish_values[phase[0]:phase[1]])
                speeds_by_avg.append(avg_speed)
            
            # Assign to map
            if op == "max":
                k = mapping_r_sf(revs_by_max[0], speeds_by_avg[0])
                l = mapping_r_sf(revs_by_max[1], speeds_by_avg[1])
                m = mapping_r_sf(revs_by_max[2], speeds_by_avg[2])
            elif op == "avg":
                k = mapping_r_sf(revs_by_avg[0], speeds_by_avg[0])
                l = mapping_r_sf(revs_by_avg[1], speeds_by_avg[1])
                m = mapping_r_sf(revs_by_avg[2], speeds_by_avg[2])
            else:
                raise ValueError("Invalid operation")
            
            if l == 2 and m == 0:
                print(f"Exp ID: {exp_id}, Trial ID: {trial['m_trial_id']}")

            prob_map[k * 9 + l * 3 + m] += 1

    prob_map = prob_map / np.sum(prob_map)

    return prob_map

def draw_curved_edges(G, pos, edge_labels):
    ax = plt.gca()
    for (u, v), label in edge_labels.items():
        # Adjust curvature
        rad = 0.2 if G[u][v]['weight'] != 1 else -0.2
        edge = FancyArrowPatch(posA=pos[u], posB=pos[v], connectionstyle=f'arc3,rad={rad}', 
                               arrowstyle='->', color='gray', lw=2)
        ax.add_patch(edge)
        # Position edge labels along the curve taking into account the rad when calculating mid_x and mid_y
        mid_x = (pos[u][0] + pos[v][0]) / 2 + (pos[v][1] - pos[u][1]) * rad
        mid_y = (pos[u][1] + pos[v][1]) / 2 - (pos[v][0] - pos[u][0]) * rad
        ax.text(mid_x, mid_y, label, color='black', fontsize=8, fontweight='bold', ha='center', va='center')


# Function to draw non-overlapping edge labels with slight offset
def draw_non_overlapping_edge_labels(G, pos, edge_labels):
    for (u, v), label in edge_labels.items():
        # Find the midpoint of the edge
        mid_x = (pos[u][0] + pos[v][0]) / 2
        mid_y = (pos[u][1] + pos[v][1]) / 2
        # Offset label position slightly to avoid overlap
        offset_x = 0.05 * (pos[v][1] - pos[u][1])  # Small offset along x
        offset_y = 0.05 * (pos[u][0] - pos[v][0])  # Small offset along y
        plt.text(mid_x + offset_x, mid_y + offset_y, label, color="black", fontsize=10)


def plot_prob_map(probability_map, z="distance", op="max"):
    # Labels for behaviors and colors for behaviors
    behavior_labels = ['Exploration', 'Freezing', 'Escape']
    behavior_colors = {'Exploration': '#cc99ff', 'Freezing': '#ff9999', 'Escape': '#99ccff'}

    # Creating the decision tree graph using networkx
    G = nx.DiGraph()

    # Step 1: Compute Phase 1 Probabilities
    phase_1_probs = [
        np.sum(probability_map[0:9]),   # Sum of rows 0-8 for Non-Escape
        np.sum(probability_map[9:18]),  # Sum of rows 9-17 for Freezing
        np.sum(probability_map[18:27])  # Sum of rows 18-26 for Escape
    ]

    # Add Phase 1 nodes with the computed probabilities (with unique identifier)
    for i, behavior in enumerate(behavior_labels):
        node_id = f'Phase 1: {behavior}'  # Unique identifier (hash) for the node
        G.add_node(node_id, label=f'{behavior}\n(P: {phase_1_probs[i]:.2f})', subset=0, color=behavior_colors[behavior])
    
    phase_2_probs = [
        np.sum([probability_map[i * 9: i * 9 + 3] for i in range(3)]),
        np.sum([probability_map[i * 9 + 3: i * 9 + 6] for i in range(3)]),
        np.sum([probability_map[i * 9 + 6: i * 9 + 9] for i in range(3)]),
    ]

    for i, behavior in enumerate(behavior_labels):
        node_id = f'Phase 2: {behavior}'
        G.add_node(node_id, label=f'{behavior}\n(P: {phase_2_probs[i]:.2f})', subset=1, color=behavior_colors[behavior])
    
    phase_3_probs = [
        np.sum([probability_map[i * 3] for i in range(9)]),
        np.sum([probability_map[i * 3 + 1] for i in range(9)]),
        np.sum([probability_map[i * 3 + 2] for i in range(9)]),
    ]

    print(phase_3_probs)

    for i, behavior in enumerate(behavior_labels):
        node_id = f'Phase 3: {behavior}'
        G.add_node(node_id, label=f'{behavior}\n(P: {phase_3_probs[i]:.2f})', subset=2, color=behavior_colors[behavior])

    # Phase 2 and Phase 3 probabilities based on transitions
    for i, phase_1_behavior in enumerate(behavior_labels):
        node_id_1 = f'Phase 1: {phase_1_behavior}'
        probs_f1t2 = [np.sum(probability_map[i * 9 + j * 3: i * 9 + j * 3 + 3]) for j in range(3)]
        probs_f1t2 /= np.sum(probs_f1t2)
        for j, phase_2_behavior in enumerate(behavior_labels):
            prob_f1t2 = probs_f1t2[j]
            if prob_f1t2 > 0:
                node_id_2 = f'Phase 2: {phase_2_behavior}'
                G.add_edge(node_id_1, node_id_2, weight=prob_f1t2)

    for i, phase_2_behavior in enumerate(behavior_labels):
        node_id_2 = f'Phase 2: {phase_2_behavior}'
        probs_f2t3 = [np.sum([probability_map[i * 3 + j * 9 + k]for j in range(3)]) for k in range(3)]
        probs_f2t3 /= np.sum(probs_f2t3)
        for j, phase_3_behavior in enumerate(behavior_labels):
            prob_f2t3 = probs_f2t3[j]
            if prob_f2t3 > 0:
                node_id_3 = f'Phase 3: {phase_3_behavior}'
                G.add_edge(node_id_2, node_id_3, weight=prob_f2t3)

    # Draw the graph with multipartite layout (using the subset key)
    plt.figure(figsize=(6, 5))
    pos = nx.multipartite_layout(G, subset_key="subset")  # Arrange nodes by phase (subset)

    # Node attributes
    node_labels = nx.get_node_attributes(G, 'label')  # Display 'label' as the node text
    node_colors = [G.nodes[n]['color'] for n in G.nodes()]  # Use 'color' attribute for node color

    # Draw the nodes with custom colors and labels
    nx.draw(G, pos, labels=node_labels, with_labels=True, node_size=3000, node_color=node_colors, font_size=8, font_weight='bold', edge_color='gray', edgecolors='black', arrows=True)

    # Add edge labels with weights
    # edge_labels = {(u, v): f'{G[u][v]["weight"]:.2f}' for u, v in G.edges()}
    # nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    # Curved edges with weights
    # Curved edges with weights
    edge_labels = {(u, v): f'{G[u][v]["weight"]:.2f}' for u, v in G.edges()}
    # nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8, font_weight='bold', label_pos=0.625)

    # # Add phase annotations on top of the nodes
    # phase_positions = {0: (0.5, 1.05), 1: (0.5, 0.7), 2: (0.5, 0.35)}  # Position for phase labels
    # for phase, coords in phase_positions.items():
    #     plt.text(coords[0], coords[1], f'Phase {phase + 1}', horizontalalignment='center', verticalalignment='center', fontsize=12, fontweight='bold', transform=plt.gca().transAxes)

    # Get the current axis limits
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # Adjust the ylim to create more space at the top
    ax.set_ylim(ylim[0], ylim[1] + 0.25)  # Add 0.5 units of space above the top nodes

    # Add the text at the top of the figure, in figure coordinates
    plt.text(0.05, 0.95, 'Phase 1', horizontalalignment='left', fontsize=12, fontweight='bold', transform=plt.gcf().transFigure)
    plt.text(0.5, 0.95, 'Phase 2', horizontalalignment='center', fontsize=12, fontweight='bold', transform=plt.gcf().transFigure)
    plt.text(0.95, 0.95, 'Phase 3', horizontalalignment='right', fontsize=12, fontweight='bold', transform=plt.gcf().transFigure)

    plt.title("Hierarchical Tree Visualization of Behaviors and Phases")
    plt.savefig(f"{study.plot_output_folder}/behavior_tree_{z}_{op}.pdf")

if __name__ == "__main__":
    exp_trials = get_trials()
    plot_trials_per_exp(exp_trials)

    # # print number of trials
    # num_trials = 0
    # for trials in exp_trials:
    #     num_trials += len(trials)
    # print(f"Total number of trials: {num_trials}")

    # z = "distance"
    # op = "avg"
    # prob_map = behavior_maps_by_z(exp_trials, z, op)
    # print(prob_map)
    # plot_prob_map(prob_map, z, op)