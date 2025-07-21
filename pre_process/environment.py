import numpy as np

from utils.maths.geometry import get_min_distance_from_point_to_line, get_intersection_of_two_lines
import numpy as np

class Environment:
    def __init__(self,
                 wall_corners) -> None:
        self.wall_corners = wall_corners
    
    def get_wall_distances(
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
        return distances
    
    def get_wall_distances_sorted(
        self,
        entity_position,
    ):
        distances = self.get_wall_distances(entity_position)
        return np.sort(distances), np.argsort(distances)
    
    def get_wall_vector_by_index(self, index):
        return self.wall_corners[(index + 1) % 4] - self.wall_corners[index]

    def get_orientation_to_wall(self, position, orientation, wall_index):
        """
        Compute the orientation angle (gamma) of the fish relative to the closest point on the wall.

        Parameters:
        - position: tuple (x_f, y_f) representing the fish's position.
        - orientation: heading angle theta_f in radians representing the individual's orientation.
        - wall_index: index of the wall segment to consider (0, 1, 2, or 3).

        Returns:
        - gamma: the orientation angle (in radians) between the fish's current moving direction and the closest point on the wall.
        """
        # Fish position (x_f, y_f)
        x_f, y_f = position
        
        # Fish's moving direction vector based on its orientation (theta_f)
        theta_f = orientation
        v_f = np.array([np.cos(theta_f), np.sin(theta_f)])  # Unit vector in the fish's moving direction
        
        # Wall points (x_wall1, y_wall1) and (x_wall2, y_wall2)
        wall_start = self.wall_corners[wall_index]
        wall_end = self.wall_corners[(wall_index + 1) % 4]
        x_wall1, y_wall1 = wall_start
        x_wall2, y_wall2 = wall_end
        
        # Vector representing the wall segment
        wall_vector = np.array([x_wall2 - x_wall1, y_wall2 - y_wall1])
        
        # Vector from the first wall point to the fish's position
        wall_to_fish = np.array([x_f - x_wall1, y_f - y_wall1])
        
        # Project fish's position onto the wall segment to find the closest point
        wall_length_squared = np.dot(wall_vector, wall_vector)
        t = np.dot(wall_to_fish, wall_vector) / wall_length_squared
        
        # Clamp t between 0 and 1 to make sure the closest point is on the segment
        t = max(0, min(1, t))
        
        # Closest point on the wall
        closest_point = np.array([x_wall1, y_wall1]) + t * wall_vector
        
        # Vector from the fish to the closest point on the wall
        vector_to_closest_point = closest_point - np.array([x_f, y_f])
        
        # Compute the angle (gamma) between the fish's moving direction (v_f) and the vector to the closest point
        dot_product = np.dot(v_f, vector_to_closest_point)
        magnitude_v_f = np.linalg.norm(v_f)
        magnitude_closest_point = np.linalg.norm(vector_to_closest_point)
        
        # Avoid division by zero (when fish and wall points are coincident)
        if magnitude_closest_point == 0:
            return 0.0  # If the fish is exactly at the closest point, the angle is 0
        
        # Compute the cosine of the angle between the vectors
        cos_gamma = dot_product / (magnitude_v_f * magnitude_closest_point)
        
        # Clip to avoid precision issues (cosine should always be in [-1, 1])
        cos_gamma = np.clip(cos_gamma, -1.0, 1.0)
        
        # Return the angle gamma in radians
        gamma = np.arccos(cos_gamma)
        
        return gamma
    

    def get_wall_side(self, fish_position, fish_orientation, threshold=10):
        """
        Determine if the fish is swimming along a wall (close < threshold cm) 
        and whether the wall is on the left or right side of the fish.

        Parameters:
            fish_position: tuple (x_f, y_f) representing the fish's position.
            fish_orientation: heading angle theta_f in radians representing the fish's orientation.
            threshold: Distance threshold for determining "close to wall."

        Returns:
            dict: 
                - close_walls: Dictionary of walls that are within the threshold distance.
                - wall_sides: Dictionary indicating whether the wall is on the left or right side of the fish.
        """
        # Get distances to all walls
        distances = self.get_wall_distances(fish_position)

        # Identify walls within the threshold distance
        close_walls = {i: dist for i, dist in enumerate(distances) if dist < threshold}

        # Determine the side (left or right) for each close wall
        wall_sides = {}
        for wall_index in close_walls.keys():
            # Get wall vector
            wall_vector = self.get_wall_vector_by_index(wall_index)

            # Fish heading vector
            fish_heading_vector = np.array([np.cos(fish_orientation), np.sin(fish_orientation)])

            # Compute the cross product to determine the relative position of the wall
            cross_product = np.cross(wall_vector, fish_heading_vector)
            wall_sides[wall_index] = "left" if cross_product < 0 else "right"

        return {"close_walls": close_walls, "wall_sides": wall_sides}


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
    
