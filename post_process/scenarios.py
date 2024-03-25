import numpy as np
import pandas as pd

from pre_process.environment import Environment

import utils.maths.geometry as geometry

CHASE_ROBOT_VELOCITY_LOWER_LIMIT = 0.5
CHASE_ROBOT_ANGLE_DIFF_UPPER_LIMIT = np.pi / 6

def is_robot_chasing(robot_velocity, robot_angle_diff_n):
    return (
        robot_velocity >= CHASE_ROBOT_VELOCITY_LOWER_LIMIT
        and abs(robot_angle_diff_n) <= CHASE_ROBOT_ANGLE_DIFF_UPPER_LIMIT
    )

OBSTRUCTED_WALL_DISTANCE_UPPER_LIMIT = 10.
ALIGNMENT_WALL_ANGLE_UPPER_LIMIT = np.pi / 6

def get_wall_information(environment: Environment,
                         robot_vector,
                         fish_position):
    wall_distances, wall_sorted_indexes = environment.get_wall_distances_sorted(fish_position)
    wall_obstructions = [d < OBSTRUCTED_WALL_DISTANCE_UPPER_LIMIT for d in wall_distances]
    n_obstructed_walls = sum(wall_obstructions)

    robot_wall_aligments = []
    for wall_index in wall_sorted_indexes:
        wall_vector = environment.get_wall_vector_by_index(wall_index)
        robot_wall_aligments.append(
            abs(geometry.angle_between_vectors(robot_vector, wall_vector))
            < ALIGNMENT_WALL_ANGLE_UPPER_LIMIT
        )

    interferred = np.logical_and(wall_obstructions, robot_wall_aligments)
    n_interferred_walls = sum(interferred)

    return n_obstructed_walls,n_interferred_walls
                         
