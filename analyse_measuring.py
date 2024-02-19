import numpy as np

class Results:
    def __init__(self,
                 list_rel_time_sec,
                 list_fish_velocity,
                 list_fish_orientation,
                 list_robot_velocity,
                 list_robot_fish_distance,
                 list_robot_fish_angle,
                 list_robot_angle_diff,
                 list_robot_fish_repulsion,
                 min_fish_robot_distance,
                 max_increase_repulsion,
                 max_increase_distance,
                 max_increase_fish_ori,
                 hit_robot_velocity,
                 hit_robot_fish_angle,
                 poses_robot,
                 poses_fish,):
        self.list_rel_time_sec = list_rel_time_sec
        self.list_fish_velocity = list_fish_velocity
        self.list_fish_orientation = list_fish_orientation
        self.list_robot_velocity = list_robot_velocity
        self.list_robot_fish_distance = list_robot_fish_distance
        self.list_robot_fish_angle = list_robot_fish_angle
        self.list_robot_angle_diff = list_robot_angle_diff
        self.list_robot_fish_repulsion = list_robot_fish_repulsion
        self.min_fish_robot_distance = min_fish_robot_distance
        self.max_increase_repulsion = max_increase_repulsion
        self.max_increase_distance = max_increase_distance
        self.max_increase_fish_ori = max_increase_fish_ori
        self.hit_robot_velocity = hit_robot_velocity
        self.hit_robot_fish_angle = hit_robot_fish_angle
        self.poses_robot = poses_robot
        self.poses_fish = poses_fish

class MeasuringTool:
    def __init__(self):
        # Initialize variables
        self.reset()

    def reset(self):
        # Reset variables
        self.list_rel_time_sec = []
        self.list_fish_velocity = []
        self.list_fish_acceleration = []
        self.list_robot_state = []
        self.list_robot_velocity = []
        self.list_robot_acceleration = []
        self.list_robot_fish_distance = []
        self.list_robot_fish_angle = []
        self.list_robot_fish_repulsion = []

    def run(self, 
            index,
            batch,
            rel_time_sec,
            positions_robot,
            positions_fish,
            states_robot,):
        self.list_rel_time_sec.append(rel_time_sec)
        self.list_fish_velocity.append(self.fish_velocity(index, positions_fish))
        self.list_fish_acceleration.append(self.fish_acceleration(index, positions_fish))
        self.list_robot_state.append(self.robot_state(index, states_robot))
        self.list_robot_velocity.append(self.robot_velocity(index, positions_robot))
        self.list_robot_acceleration.append(self.robot_acceleration(index))
        self.list_robot_fish_distance.append(self.robot_fish_distance(index, positions_robot, positions_fish))

    def fish_velocity(self, index, batch, positions_fish):
        if len(self.list_rel_time_sec) > 1:
            return np.linalg.norm(positions_fish[index] - positions_fish[index - 1]) / (self.list_rel_time_sec[index] - self.list_rel_time_sec[index - 1])

    def fish_acceleration(self, index):
        if len(self.list_fish_velocity) > 1:
            return (self.list_fish_velocity[index] - self.list_fish_velocity[index - 1]) / (self.list_rel_time_sec[index] - self.list_rel_time_sec[index - 1])

    def robot_state(self, index, states_robot):
        return states_robot[index]

    def robot_velocity(self, index, positions_robot):
        if len(self.list_rel_time_sec) > 1:
            return (positions_robot[index] - positions_robot[index - 1]) / (self.list_rel_time_sec[index] - self.list_rel_time_sec[index - 1])

    def robot_acceleration(self, index):
        if len(self.list_robot_velocity) > 1:
            return (self.list_robot_velocity[index] - self.list_robot_velocity[index - 1]) / (self.list_rel_time_sec[index] - self.list_rel_time_sec[index - 1])

    def robot_fish_distance(self, index, positions_robot, positions_fish):
        return np.linalg.norm(positions_robot[index] - positions_fish[index])

    def robot_fish_angle(self, index, positions_robot, positions_fish):
        return np.arctan2(positions_robot[index] - positions_fish[index])

    def robot_fish_repulsion(self):
        pass
