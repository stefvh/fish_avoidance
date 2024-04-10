from matplotlib import pyplot as plt
import numpy as np

import utils.maths.geometry as geometry

def test_angles():
    scenario_a = {
        "fish_positions": np.array([
            [8., 8.,],
            [6., 7.5]
            # [10., 7.5],
        ]),
        "robot_positions": np.array([
            [6., 6.,],
            [7.5, 9],
        ]),
    }
    scenario_b = {
        "fish_positions": np.array([
            [6., 6.,],
            [6., 4.],
        ]),
        "robot_positions": np.array([
            [8., 8.,],
            [6., 5.],
        ]),
    }
    scenario_c = {
        "fish_positions": np.array([
            [8., 8.,],
            [10., 10.],
        ]),
        "robot_positions": np.array([
            [6., 6.,],
            [4., 4.],
        ]),
    }
    scenario_d = {
        "fish_positions": np.array([
            [6., 6.,],
            [4., 4.],
        ]),
        "robot_positions": np.array([
            [8., 8.,],
            [9., 7.],
        ]),
    }
    scenarios = [scenario_a, scenario_b, scenario_c, scenario_d]

    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    for i, scenario in enumerate(scenarios):
        ax = axes[i // 2, i % 2]

        ax.text(
            1, 1,
            f"Scenario {i + 1}",
            fontsize=5, color='black'
        )

        fish_positions = scenario["fish_positions"]
        robot_positions = scenario["robot_positions"]

        q_f =  fish_positions[1] - fish_positions[0]
        q_fr = fish_positions[0] - robot_positions[0]
        alpha = geometry.vectors_angle(
            q_f, q_fr
        )
        q_r = robot_positions[1] - robot_positions[0]
        q_rf = robot_positions[0] - fish_positions[0]
        beta = geometry.vectors_angle(
            q_r / np.linalg.norm(q_r), q_fr / np.linalg.norm(q_fr)
        )

        ax.scatter(robot_positions[0, 0], robot_positions[0, 1], c = 'red', marker = 'o', s = 10,)
        q_r_n = q_r / np.linalg.norm(q_r)
        ax.quiver(
            robot_positions[0][0], robot_positions[0][1],
            (q_r_n[0]),
            (q_r_n[1]),
            angles='xy', scale_units='xy', scale=1, color="red"
        )
        ax.text(
            robot_positions[0, 0] + 0.2, robot_positions[0, 1] + 0.05,
            r"$q_R$",
            fontsize=8, color='black'
        )

        ax.scatter(fish_positions[0, 0], fish_positions[0, 1], c = 'blue', marker = 'D', s = 10,)
        q_f_n = q_f / np.linalg.norm(q_f)
        ax.quiver(
            fish_positions[0][0], fish_positions[0][1],
            (q_f_n[0]),
            (q_f_n[1]),
            angles='xy', scale_units='xy', scale=1, color="blue"
        )
        ax.text(
            fish_positions[0, 0] + 0.2, fish_positions[0, 1] + 0.05,
            r"$q_F$",
            fontsize=8, color='black'
        )

        ax.quiver(
            robot_positions[0][0], robot_positions[0][1],
            (q_fr[0]),
            (q_fr[1]),
            angles='xy', scale_units='xy', scale=1, color="green"
        )
        pos = robot_positions[0] + q_fr / 2
        ax.text(
            pos[0] + 0.2, pos[1] + 0.05,
            r"$q_{FR}$",
            fontsize=8, color='green'
        )

        # ax.quiver(
        #     fish_positions[0][0], fish_positions[0][1],
        #     (q_rf[0]),
        #     (q_rf[1]),
        #     angles='xy', scale_units='xy', scale=1, color="black"
        # )
        # ax.text(
        #     fish_positions[0, 0] + 0.2, fish_positions[0, 1] + 0.05,
        #     r"$q_{RF}$",
        #     fontsize=8, color='black'
        # )

        ax.text(
            1, 11,
            f"α = {alpha:.2f} rad" + r" $(q_F, q_{FR})$" + f" ({np.degrees(alpha):.2f}°)",
            fontsize=8, color='black'
        )
        ax.text(
            1, 10,
            f"β = {beta:.2f} rad" + r" $(q_R, q_{FR})$" + f" ({np.degrees(beta):.2f}°)",
            fontsize=8, color='black'
        )


        ax.set_xlim(0, 12)
        ax.set_ylim(0, 12)

    plt.savefig("testing/test_angles.pdf", bbox_inches='tight', dpi=300)


if __name__ == "__main__":
    test_angles()