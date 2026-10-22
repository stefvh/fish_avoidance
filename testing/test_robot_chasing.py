from matplotlib import pyplot as plt
import numpy as np

import utils.maths.geometry as geometry

U_ROBOT_VELOCITY = 0.5
L_ROBOT_ANGLE_DIFF = np.pi / 6

def is_robot_chasing(robot_velocity,
                     robot_angle_diff_n):
    return robot_velocity >= U_ROBOT_VELOCITY and abs(robot_angle_diff_n) <= L_ROBOT_ANGLE_DIFF

def test_is_robot_chasing():
    good_scenario = {
        "fish_positions": np.array([
            [8., 8.,],
            [10., 10.],
        ]),
        "robot_positions": np.array([
            [6., 6.,],
            [7.5, 9],
        ]),
    }
    good2_scenario = {
        "fish_positions": np.array([
            [6., 6.,],
            [4., 4.],
        ]),
        "robot_positions": np.array([
            [8., 8.,],
            [6., 5.],
        ]),
    }
    bad_scenario = {
        "fish_positions": np.array([
            [8., 8.,],
            [10., 10.],
        ]),
        "robot_positions": np.array([
            [6., 6.,],
            [9., 6.],
        ]),
    }
    bad2_scenario = {
        "fish_positions": np.array([
            [6., 6.,],
            [4., 4.],
        ]),
        "robot_positions": np.array([
            [8., 8.,],
            [9., 7.],
        ]),
    }
    scenarios = [good_scenario, bad_scenario,
                 good2_scenario, bad2_scenario]

    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    for i, scenario in enumerate(scenarios):
        ax = axes[i // 2, i % 2]

        fish_positions = scenario["fish_positions"]
        robot_positions = scenario["robot_positions"]

        robot_velocity = np.linalg.norm(
            robot_positions[1] - robot_positions[0]
        )

        robot_angle = geometry.vector_angle(
            robot_positions[0], robot_positions[1]
        )
        robot_fish_angle = geometry.vector_angle(
            robot_positions[0], fish_positions[1]
        )
        robot_angle_diff = abs(geometry.diff_angles(robot_fish_angle, robot_angle))

        for r_pos in robot_positions:
            ax.scatter(r_pos[0], r_pos[1], c = 'red', marker = 'o', s = 10,)
        ax.quiver(
            robot_positions[0][0], robot_positions[0][1],
            (robot_positions[1][0] - robot_positions[0][0]),
            (robot_positions[1][1] - robot_positions[0][1]),
            angles='xy', scale_units='xy', scale=1
        )

        ax.scatter(fish_positions[1, 0], fish_positions[1, 1], c = 'blue', marker = 'D', s = 10,)
        ax.quiver(
            robot_positions[0][0], robot_positions[0][1],
            (fish_positions[1][0] - robot_positions[0][0]),
            (fish_positions[1][1] - robot_positions[0][1]),
            angles='xy', scale_units='xy', scale=1
        )

        # From robot_positions[0] : shade the area of the robot's vision
        # The vision is 30 degrees above and below the angle of the robot to the fish
        ax.plot(
            [robot_positions[0, 0], robot_positions[0, 0] + 10 * np.cos(robot_fish_angle + L_ROBOT_ANGLE_DIFF)],
            [robot_positions[0, 1], robot_positions[0, 1] + 10 * np.sin(robot_fish_angle + L_ROBOT_ANGLE_DIFF)],
            c = 'tab:green', linestyle = '--'
        )
        ax.plot(
            [robot_positions[0, 0], robot_positions[0, 0] + 10 * np.cos(robot_fish_angle - L_ROBOT_ANGLE_DIFF)],
            [robot_positions[0, 1], robot_positions[0, 1] + 10 * np.sin(robot_fish_angle - L_ROBOT_ANGLE_DIFF)],
            c = 'tab:green', linestyle = '--'
        )

        ax.set_xlim(0, 12)
        ax.set_ylim(0, 12)

        ax.set_title(fr"$v_R$: {robot_velocity:.2f}  $\alpha$: {robot_angle_diff:.2f} ({is_robot_chasing(robot_velocity, robot_angle_diff)})")

    plt.savefig("testing/test_robot_chasing.pdf", bbox_inches='tight', dpi=300)


if __name__ == "__main__":
    test_is_robot_chasing()
