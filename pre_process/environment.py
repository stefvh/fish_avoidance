import numpy as np

from utils.maths.geometry import get_min_distance_from_point_to_line, get_intersection_of_two_lines
import numpy as np

class Environment:
    def __init__(self,
                 wall_corners) -> None:
        self.wall_corners = wall_corners
    
    def get_wall_distances_sorted(
        self,
        entity_position,
    ):
        distances = np.array(
            [
                get_min_distance_from_point_to_line(
                    entity_position, self.wall_corners[i], self.wall_corners[(i + 1) % 4]
                )
                for i in range(4)
            ]
        )

        return np.sort(distances), np.argsort(distances)
    
    def get_wall_vector_by_index(self, index):
        return self.wall_corners[(index + 1) % 4] - self.wall_corners[index]
    

def get_wall_distances_sorted(
    wall_corners,
    entity_position,
):
    distances = np.array(
        [
            get_min_distance_from_point_to_line(
                entity_position, wall_corners[i], wall_corners[(i + 1) % 4]
            )
            for i in range(4)
        ]
    )

    return np.sort(distances), np.argsort(distances)

def get_wall_vector_by_index(wall_corners, index):
    return wall_corners[(index + 1) % 4] - wall_corners[index]

def will_entity_crash_into_wall(
    wall_corners,
    entity_position,
    entity_orientation,
    entity_distance
):
    """
    Check if the entity will crash into the wall
    the entity is a point that moves in a straight line with a given orientation
    the entity will crash into the wall if the projected trajectory of the entity intersects the wall
    the projected trajectory of the entity is a line that starts at the entity's position and has the same orientation as the entity
    """
    future_position = entity_position + entity_distance * np.array(
        [np.cos(entity_orientation), np.sin(entity_orientation)]
    )

    # check if future_position is outside of the environment by the wall_corners
    if (
        future_position[0] < min(wall_corners[:, 0])
        or future_position[0] > max(wall_corners[:, 0])
        or future_position[1] < min(wall_corners[:, 1])
        or future_position[1] > max(wall_corners[:, 1])
    ):
        return True


def get_max_possible_trajectory_distance(
    wall_corners,
    entity_position,
    entity_orientation,
    entity_distance
):
    """
    Get the maximum possible distance the entity can travel without crashing into the wall
    the entity is a point that moves in a straight line with a given orientation
    """
    future_position = entity_position + entity_distance * np.array(
        [np.cos(entity_orientation), np.sin(entity_orientation)]
    )
    vector = future_position - entity_position

    # find intersection with each wall
    distances = []
    for i in range(4):
        wall_vector = wall_corners[(i + 1) % 4] - wall_corners[i]
        intersection = get_intersection_of_two_lines(
            entity_position, vector, wall_corners[i], wall_vector
        )
        if intersection is not None:
            distances.append(np.linalg.norm(intersection - entity_position))
    
    if len(distances) == 0:
        return entity_distance
    else:
        return min(distances)

