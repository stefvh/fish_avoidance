from matplotlib import pyplot as plt
import numpy as np

from pre_process import environment
import utils.maths as maths

XLIMS = [0, 10]
YLIMS = [0, 10]
WALL_CORNERS = np.array(
    [
        [XLIMS[0], YLIMS[1]],
        [XLIMS[1], YLIMS[1]],
        [XLIMS[1], YLIMS[0]],
        [XLIMS[0], YLIMS[0]],
    ]
)

U_WALL_DISTANCE = 2.0
U_WALL_ALIGNMENT = np.pi / 6


def is_fish_wall_obstructed(wall_distances: list):
    return any([d < U_WALL_DISTANCE for d in wall_distances])


def get_robot_wall_alignments(wall_sorted_indexes: list, robot_vector: np.ndarray):
    robot_wall_aligments = []
    for wall_index in wall_sorted_indexes:
        wall_vector = environment.get_wall_vector_by_index(WALL_CORNERS, wall_index)
        robot_wall_aligments.append(
            abs(maths.angle_between_vectors(robot_vector, wall_vector))
            < U_WALL_ALIGNMENT
        )
    return robot_wall_aligments


def hypothesized_potential(position, robot_position):
    potential = 0.0
    # repulsion from robot
    distance = np.linalg.norm(position - robot_position)
    if distance < 2:
        potential += min(5.0, 1 / distance)
    # repulsion from wall
    wall_distances, _ = environment.get_wall_distances_sorted(WALL_CORNERS, position)
    for d in wall_distances:
        if d < 1:
            potential += min(10.0, 1 / d)
    # alignment with close walls
    for d in wall_distances:
        if 1 < d < 2:
            potential -= 10
    return potential


def hypothesized_potential_2(position, fish_position, fish_orientation, robot_position):
    potential = 0.0
    max_velocity = 5.0

    # move in the current direction
    angle = np.arctan2(
        position[1] - fish_position[1],
        position[0] - fish_position[0],
    )
    angle_diff = abs(maths.diff_angles(angle, fish_orientation))
    if angle_diff < np.pi / 6:
        # smaller angle difference, higher potential
        p_f = 1 - angle_diff / (np.pi / 6)
    else:
        p_f = 0
    f_distance = np.linalg.norm(position - fish_position)
    if f_distance < max_velocity:
        # smaller distance, higher potential
        p_f *= 1 - f_distance / max_velocity
    else:
        p_f = 0

    # move away from robot
    rf_angle = np.arctan2(
        fish_position[1] - robot_position[1],
        fish_position[0] - robot_position[0],
    )
    angle_diff = abs(maths.diff_angles(angle, rf_angle))
    if angle_diff < np.pi / 6:
        # smaller angle difference, higher potential
        p_r = 1 - angle_diff / (np.pi / 6)
    else:
        p_r = 0
    if f_distance < max_velocity:
        # smaller distance, higher potential
        p_r *= 1 - f_distance / max_velocity
    else:
        p_r = 0

    # align with close walls
    wall_distances, wall_indexes = environment.get_wall_distances_sorted(
        WALL_CORNERS, position
    )
    p_w = 0.0
    for d, wall_index in zip(wall_distances, wall_indexes):
        if d < 2:
            wall_vector = environment.get_wall_vector_by_index(WALL_CORNERS, wall_index)
            wall_angle = np.arctan2(wall_vector[1], wall_vector[0])
            for w_a in [wall_angle, wall_angle + np.pi]:
                angle_diff = abs(maths.diff_angles(angle, w_a))
                _p_w = 0.0
                if angle_diff < np.pi / 6:
                    # smaller angle difference, higher potential
                    _p_w += 1 - angle_diff / (np.pi / 6)
                if f_distance < max_velocity:
                    # smaller distance, higher potential
                    _p_w *= 1 - f_distance / max_velocity
                else:
                    _p_w = 0
                p_w += _p_w

    w_f, w_r, w_w = 1.0, 1.0, 1.0
    potential = w_f * p_f + w_r * p_r + w_w * p_w

    # remove potential around robot
    angle_diff = abs(maths.diff_angles(angle, rf_angle + np.pi))
    if angle_diff < np.pi / 6:
        potential = 0

    # remove potential towards wall
    max_d = environment.get_max_possible_trajectory_distance(
        WALL_CORNERS, position, angle, 2
    )
    max_d += f_distance
    if max_d < 0.99 * 2:
        potential = 0

    return potential


def compute_probability_forward_and_turn_movements_from_potential_grid(
    entity_position, entity_orientation, potential_grid
):
    """
    The entity moves by a forward length and a turn angle
    Based on the potential grid, compute the probability of forward and turn movements
    The output is a 3d array, where the x-axis is the forward length and the y-axis is the turn angle
    The z-axis is the probability of forward and turn movements.

    The potential grid is divided as the X, Y meshgrid
    """
    x = np.linspace(XLIMS[0], XLIMS[1], 100)
    y = np.linspace(YLIMS[0], YLIMS[1], 100)
    X, Y = np.meshgrid(x, y)

    # potential_grid = potential_grid.T

    forward_lengths = np.arange(0.25, 3.0, 0.25)
    turn_angles = np.arange(-np.pi, np.pi, np.pi / 18)
    probabilities = np.zeros((len(forward_lengths), len(turn_angles)))
    for i, forward_length in enumerate(forward_lengths):
        for j, turn_angle in enumerate(turn_angles):
            new_position = entity_position + forward_length * np.array(
                [
                    np.cos(entity_orientation + turn_angle),
                    np.sin(entity_orientation + turn_angle),
                ]
            )
            if (
                new_position[0] < XLIMS[0]
                or new_position[0] > XLIMS[1]
                or new_position[1] < YLIMS[0]
                or new_position[1] > YLIMS[1]
            ):
                probabilities[i, j] = 0
            else:
                x_index = np.argmin(np.abs(x - new_position[0]))
                y_index = np.argmin(np.abs(y - new_position[1]))
                probabilities[i, j] = potential_grid[x_index, y_index]
    return probabilities, forward_lengths, turn_angles


def test_wall_scenarios():
    scenario_1 = {
        "fish_position": np.array(
            [
                5.0,
                5.0,
            ],
        ),
        "fish_orientation": np.pi,
        "robot_positions": np.array(
            [
                [
                    7.5,
                    9.0,
                ],
                [6.0, 6.5],
            ]
        ),
    }
    scenario_2 = {
        "fish_position": np.array(
            [
                5.0,
                1.5,
            ],
        ),
        "fish_orientation": 0.0,
        "robot_positions": np.array(
            [
                [
                    7.0,
                    6.0,
                ],
                [5.5, 3.0],
            ]
        ),
    }
    scenario_3 = {
        "fish_position": np.array(
            [
                5.0,
                1.5,
            ],
        ),
        "fish_orientation": np.pi,
        "robot_positions": np.array(
            [
                [7.5, 1.0],
                [6, 1.2],
            ]
        ),
    }
    scenario_4 = {
        "fish_position": np.array(
            [
                8.5,
                1.5,
            ],
        ),
        "fish_orientation": np.pi / 2.0,
        "robot_positions": np.array(
            [
                [
                    8.5,
                    5.0,
                ],
                [8.5, 3.0],
            ]
        ),
    }

    scenarios = [scenario_1, scenario_2, scenario_3, scenario_4]

    # plot the scenarios with potential
    potential_grids = []
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for i, scenario in enumerate(scenarios):
        ax = axes[i // 2, i % 2]

        fish_position = scenario["fish_position"]
        robot_positions = scenario["robot_positions"]

        wall_distances, wall_sorted_indexes = environment.get_wall_distances_sorted(
            WALL_CORNERS, fish_position
        )
        fish_wall_obstructed = is_fish_wall_obstructed(wall_distances)

        robot_vector = robot_positions[1] - robot_positions[0]
        robot_walls_alignments = get_robot_wall_alignments(
            wall_sorted_indexes, robot_vector
        )

        wall_obstructions = [d < U_WALL_DISTANCE for d in wall_distances]
        n_obstructed_walls = sum(wall_obstructions)

        interferred = np.logical_and(wall_obstructions, robot_walls_alignments)
        n_interferred_walls = sum(interferred)

        ax.scatter(
            fish_position[0],
            fish_position[1],
            c="blue",
            marker="D",
            s=10,
        )
        # quiver fish
        ax.quiver(
            fish_position[0],
            fish_position[1],
            2 * np.cos(scenario["fish_orientation"]),
            2 * np.sin(scenario["fish_orientation"]),
            angles="xy",
            scale_units="xy",
            scale=1,
        )

        for r_pos in robot_positions:
            ax.scatter(
                r_pos[0],
                r_pos[1],
                c="red",
                marker="o",
                s=10,
            )
        ax.quiver(
            robot_positions[0][0],
            robot_positions[0][1],
            (robot_positions[1][0] - robot_positions[0][0]),
            (robot_positions[1][1] - robot_positions[0][1]),
            angles="xy",
            scale_units="xy",
            scale=1,
        )

        ax.set_xlim(XLIMS)
        ax.set_ylim(YLIMS)
        ax.set_title(rf"$N_w$: {n_obstructed_walls} | $N_i$: {n_interferred_walls}")

        # colormap of potential
        x = np.linspace(XLIMS[0], XLIMS[1], 100)
        y = np.linspace(YLIMS[0], YLIMS[1], 100)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)
        for i in range(len(x)):
            for j in range(len(y)):
                Z[i, j] = hypothesized_potential_2(
                    np.array([x[i], y[j]]),
                    fish_position,
                    scenario["fish_orientation"],
                    robot_positions[1],
                )
                # Z[i, j] = hypothesized_potential(
                #     np.array([x[i], y[j]]), robot_positions[1]
                # )
        Z_T = Z.T
        potential_grids.append(Z)
        ax.contourf(X, Y, Z_T, alpha=0.5)
        # show colorbar
        cbar = plt.colorbar(ax.contourf(X, Y, Z_T, alpha=0.5), ax=ax)

    plt.savefig("testing/test_wall_scenarios.pdf")

    # plot the scenarios with probabilities
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    for i, scenario in enumerate(scenarios):
        ax = axes[i // 2, i % 2]

        # colormap of probabilities
        PR, fl, tl = compute_probability_forward_and_turn_movements_from_potential_grid(
            scenario["fish_position"], scenario["fish_orientation"], potential_grids[i]
        )
        # print(PR)
        X, Y = np.meshgrid(fl, tl)
        PR_T = PR.T
        ax.contourf(X, Y, PR_T, alpha=0.5)
        # show colorbar
        cbar = plt.colorbar(ax.contourf(X, Y, PR_T, alpha=0.5), ax=ax)

        ax.set_xlabel("Forward length")
        ax.set_ylabel("Turn angle")

    plt.savefig("testing/test_wall_scenarios_probabilities.pdf")


if __name__ == "__main__":
    test_wall_scenarios()
