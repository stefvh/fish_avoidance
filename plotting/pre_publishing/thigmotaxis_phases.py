from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

from core.experiment import Experiment
from core.study import study

import networkx as nx

import utils.plotter as plotter

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


def mapping(d_fw):
    if d_fw < 10:
        return 0
    else:
        return 1

def get_prob_map(exp_trials):
    prob_map = np.zeros(8)

    exp_phase_avgs = {}

    for exp_id, trials in enumerate(exp_trials):
        phase_avgs = [
            [], []
        ]
        for trial in trials:
            z_values = trial["data"]["distance"].to_numpy()
            dfw_values = trial["data"]["min_wall_distance_fish"].to_numpy()

            # Measurements
            ef = np.argmin(z_values)
            max_t = z_values.shape[0]
            td = 10
            ph1 = (0, max(0, ef - td))
            ph2 = (max(0, ef - td), ef + td + 1)
            ph3 = (ef + td + 1, max_t)
            phases = [ph1, ph2, ph3]
            dfws_by_avg = []
            for l, phase in enumerate(phases):
                avg_dfw = np.mean(dfw_values[phase[0]:phase[1]])
                dfws_by_avg.append(avg_dfw)
            phase_avgs[0].append(dfws_by_avg[1] - dfws_by_avg[0])
            phase_avgs[1].append(dfws_by_avg[2] - dfws_by_avg[1])
            
            k = mapping(dfws_by_avg[0])
            l = mapping(dfws_by_avg[1])
            m = mapping(dfws_by_avg[2])

            prob_map[k * 4 + l * 2 + m] += 1
        phase_avgs = [np.mean(phase) for phase in phase_avgs]
        exp_phase_avgs[exp_id] = phase_avgs

    prob_map = prob_map / np.sum(prob_map)

    print(exp_phase_avgs)

    return prob_map


def plot_prob_map(probability_map):
    # Labels for behaviors and colors for behaviors
    behavior_labels = ['Open-field', 'Thigomatxis']
    behavior_colors = {'Open-field': '#ffd699', 'Thigomatxis': '#99ff99',}

    # Creating the decision tree graph using networkx
    G = nx.DiGraph()

    # Step 1: Compute Phase 1 Probabilities
    phase_1_probs = [
        np.sum(probability_map[0:4]),   # Sum of rows 0-8 for Open-field
        np.sum(probability_map[4:]),  # Sum of rows 9-17 for Thigomatxis
    ]

    # Add Phase 1 nodes with the computed probabilities (with unique identifier)
    for i, behavior in enumerate(behavior_labels):
        node_id = f'Phase 1: {behavior}'  # Unique identifier (hash) for the node
        label = f'{behavior}\n' + r'$p=$ ' + f'{phase_1_probs[i]:.2f}'
        G.add_node(node_id, label=label, subset=0, color=behavior_colors[behavior])
    
    phase_2_probs = [
        np.sum([probability_map[i * 4: i * 4 + 2] for i in range(2)]),
        np.sum([probability_map[i * 4 + 2: i * 4 + 4] for i in range(2)]),
    ]

    for i, behavior in enumerate(behavior_labels):
        node_id = f'Phase 2: {behavior}'
        label = f'{behavior}\n' + r'$p=$ ' + f'{phase_2_probs[i]:.2f}'
        G.add_node(node_id, label=label, subset=1, color=behavior_colors[behavior])
    
    phase_3_probs = [
        np.sum([probability_map[i * 2] for i in range(4)]),
        np.sum([probability_map[i * 2 + 1] for i in range(4)]),
    ]

    for i, behavior in enumerate(behavior_labels):
        node_id = f'Phase 3: {behavior}'
        label = f'{behavior}\n' + r'$p=$ ' + f'{phase_3_probs[i]:.2f}'
        G.add_node(node_id, label=label, subset=2, color=behavior_colors[behavior])

    # Phase 2 and Phase 3 probabilities based on transitions
    for i, phase_1_behavior in enumerate(behavior_labels):
        node_id_1 = f'Phase 1: {phase_1_behavior}'
        probs_f1t2 = [np.sum(probability_map[i * 4 + j * 2: i * 4 + j * 2 + 2]) for j in range(2)]
        probs_f1t2 /= np.sum(probs_f1t2)
        for j, phase_2_behavior in enumerate(behavior_labels):
            prob_f1t2 = probs_f1t2[j]
            if prob_f1t2 > 0:
                node_id_2 = f'Phase 2: {phase_2_behavior}'
                G.add_edge(node_id_1, node_id_2, weight=prob_f1t2)

    for i, phase_2_behavior in enumerate(behavior_labels):
        node_id_2 = f'Phase 2: {phase_2_behavior}'
        probs_f2t3 = [np.sum([probability_map[i * 2 + j * 4 + k]for j in range(2)]) for k in range(2)]
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

    # Get the current axis limits
    ax = plt.gca()
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

    plt.savefig(f"{study.plot_output_folder}/TEST-PUBLISH/thigomatxis_phases.pdf")

if __name__ == "__main__":
    exp_trials = get_trials()
    prob_map = get_prob_map(exp_trials)
    plot_prob_map(prob_map)