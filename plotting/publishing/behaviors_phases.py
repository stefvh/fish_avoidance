import os
from matplotlib import pyplot as plt
from matplotlib.patches import Circle
import numpy as np
import pandas as pd

from core.experiment import Experiment
from core.study import study

import networkx as nx

import utils.plotter as plotter

from analysis.clean.freezing_conditions import compute_related_freezing


def get_trials():
    remove_exp_trials = [
        [10, 21],
        [2, 11, 12, 13, 15],
        [8],
        [],
        []
    ]
    data = study.data

    # TODO
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


def get_prob_map(exp_trials):
    prob_map = np.zeros(27)
    freezing_encounters = []

    for exp_id, trials in enumerate(exp_trials):
        freezing_a, freezing_b = 0, 0
        n_trials = 0

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
                avg_rev = np.mean(rev_values[phase[0]:phase[1]])
                revs_by_avg.append(avg_rev)
                avg_speed = np.mean(speed_fish_values[phase[0]:phase[1]])
                speeds_by_avg.append(avg_speed)
            
            k = mapping(revs_by_avg[0], speeds_by_avg[0])
            l = mapping(revs_by_avg[1], speeds_by_avg[1])
            m = mapping(revs_by_avg[2], speeds_by_avg[2])

            prob_map[k * 9 + l * 3 + m] += 1

            # Freezing
            phase = ph2

            freezing_spd_values = trial["data"]["freezing_spd"].to_numpy()
            freezing_dist_values = trial["data"]["freezing_dist"].to_numpy()

            # freezing_spd = np.any(freezing_spd_values[phase[0]:phase[1]])   #  0.059
            # freezing_dist = np.any(freezing_dist_values[phase[0]:phase[1]]) # 0.028

            freezing_spd = freezing_spd_values[phase[0]] # 0.0415
            freezing_dist = freezing_dist_values[phase[0]] # 0.028

            # freezing_spd = np.any(freezing_spd_values[phase[0]:phase[1] - td])   #  0.0415
            # freezing_dist = np.any(freezing_dist_values[phase[0]:phase[1] - td]) # 0.028

            # freezing_spd = np.any(freezing_spd_values[phase[0] + td:phase[1]])   #  0.036
            # freezing_dist = np.any(freezing_dist_values[phase[0] + td:phase[1]]) # 0.009

            # freezing_spd = np.mean(freezing_spd_values[phase[0]:phase[1]]) > 0.5  # 0.018
            # freezing_dist = np.mean(freezing_dist_values[phase[0]:phase[1]]) > 0.5 # 0.009

            if freezing_spd:
                freezing_a += 1
            if freezing_dist:
                freezing_b += 1
            n_trials += 1
        
        freezing_encounters.append([freezing_a / n_trials, freezing_b / n_trials])

    prob_map = prob_map / np.sum(prob_map)

    return prob_map, freezing_encounters


def plot_prob_map(probability_map, freezing_encounters):
    # Labels for behaviors and colors for behaviors
    behavior_labels = ['Attraction', 'Slow motion', 'Avoidance']
    behavior_colors = {'Attraction': '#cc99ff', 'Slow motion': '#ff9999', 'Avoidance': '#99ccff'}

    # Creating the decision tree graph using networkx
    G = nx.DiGraph()

    # Step 1: Compute Phase 1 Probabilities
    phase_1_probs = [
        np.sum(probability_map[0:9]),   # Sum of rows 0-8 for Exploration
        np.sum(probability_map[9:18]),  # Sum of rows 9-17 for Freezing
        np.sum(probability_map[18:27])  # Sum of rows 18-26 for Avoidance
    ]

    # Add Phase 1 nodes with the computed probabilities (with unique identifier)
    for i, behavior in enumerate(behavior_labels):
        node_id = f'Phase 1: {behavior}'  # Unique identifier (hash) for the node
        label = f'{behavior}\n' + r'$p=$ ' + f'{phase_1_probs[i]:.2f}'
        G.add_node(node_id, label=label, subset=0, color=behavior_colors[behavior])
    
    phase_2_probs = [
        np.sum([probability_map[i * 9: i * 9 + 3] for i in range(3)]),
        np.sum([probability_map[i * 9 + 3: i * 9 + 6] for i in range(3)]),
        np.sum([probability_map[i * 9 + 6: i * 9 + 9] for i in range(3)]),
    ]

    for i, behavior in enumerate(behavior_labels):
        node_id = f'Phase 2: {behavior}'
        label = f'{behavior}\n' + r'$p=$ ' + f'{phase_2_probs[i]:.2f}'
        G.add_node(node_id, label=label, subset=1, color=behavior_colors[behavior])
    
    phase_3_probs = [
        np.sum([probability_map[i * 3] for i in range(9)]),
        np.sum([probability_map[i * 3 + 1] for i in range(9)]),
        np.sum([probability_map[i * 3 + 2] for i in range(9)]),
    ]

    for i, behavior in enumerate(behavior_labels):
        node_id = f'Phase 3: {behavior}'
        label = f'{behavior}\n' + r'$p=$ ' + f'{phase_3_probs[i]:.2f}'
        G.add_node(node_id, label=label, subset=2, color=behavior_colors[behavior])

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
    plt.figure(figsize=(5, 3))
    pos = nx.multipartite_layout(G, subset_key="subset")  # Arrange nodes by phase (subset)

    # Node attributes
    node_labels = nx.get_node_attributes(G, 'label')  # Display 'label' as the node text
    node_colors = [G.nodes[n]['color'] for n in G.nodes()]  # Use 'color' attribute for node color

    # Draw the nodes with custom colors and labels
    nx.draw(G, pos, labels=node_labels, with_labels=True, node_size=2000, node_color=node_colors, font_size=6, font_weight='bold', edge_color='gray', edgecolors='black', arrows=True)

    # Add edge labels with weights
    edge_labels = {(u, v): f'{G[u][v]["weight"]:.2f}' for u, v in G.edges()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=6, font_weight='bold', label_pos=0.625)

    # TODO: Add freezing
    freezing_encounters = np.array(freezing_encounters)
    freezing_pair = np.mean(freezing_encounters, axis=0)
    print(freezing_pair)
    freezing = np.mean(freezing_pair)
    label = "Freezing\n" + r'$p\approx$ ' + f'{freezing:.2f}'
    color = '#e06666'
    G2 = nx.DiGraph()
    G2.add_node('Freezing', label=label, subset=0, color=color)
    pos2 = {'Freezing': np.array([0.12, 0.35])}
    nx.draw(G2, pos2, labels=nx.get_node_attributes(G2, 'label'), with_labels=True, node_size=1250, node_color=color, font_size=6, font_weight='bold', edge_color='gray', edgecolors='black', arrows=True)


    ax = plt.gca()
    # Get the current axis limits
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # Adjust the ylim to create more space at the top
    ax.set_ylim(ylim[0] - 0.2, ylim[1] + 0.3)  # Add 0.5 units of space above the top nodes

    # Add the text at the top of the figure, in figure coordinates
    plt.text(0.02, 0.95, 'pre-encounter', horizontalalignment='left', fontsize=10, fontweight='bold', transform=plt.gcf().transFigure)
    plt.text(0.5, 0.95, 'encounter', horizontalalignment='center', fontsize=10, fontweight='bold', transform=plt.gcf().transFigure)
    plt.text(0.98, 0.95, 'post-encounter', horizontalalignment='right', fontsize=10, fontweight='bold', transform=plt.gcf().transFigure)

    plt.axvspan(-0.3, 0.3, color='gray', alpha=0.2)

    # Add a black border around the figure
    plt.gcf().patch.set_edgecolor('black')  # Get current figure and set edge color to black
    plt.gcf().patch.set_linewidth(1)        # Set the border width

    plt.savefig(f"{study.plot_output_folder}/PUBLISH/behavior_phases.pdf")

if __name__ == "__main__":
    # # if prob_map freez_list exist, load it otherwise compute it
    if os.path.exists(f"{study.plot_output_folder}/PUBLISH/prob_map.npy"):
        prob_map = np.load(f"{study.plot_output_folder}/PUBLISH/prob_map.npy")
        freez_list = np.load(f"{study.plot_output_folder}/PUBLISH/freez_list.npy")
    else:
        exp_trials = get_trials()
        prob_map, freez_list = get_prob_map(exp_trials)
        np.save(f"{study.plot_output_folder}/PUBLISH/prob_map.npy", prob_map)
        np.save(f"{study.plot_output_folder}/PUBLISH/freez_list.npy", freez_list)
    plot_prob_map(prob_map, freez_list)