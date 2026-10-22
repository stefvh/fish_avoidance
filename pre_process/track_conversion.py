import numpy as np


def get_track_pose(frame_data, front_index, center_index):
    world_nodes = frame_data['world_nodes']
    front = world_nodes[front_index]
    center = world_nodes[center_index]
    if front['x'] is None or front['y'] is None or \
        center['x'] is None or center['y'] is None:
        return {
            "center": np.array([np.nan, np.nan, np.nan]),
            "front": np.array([np.nan, np.nan, np.nan]),
        }
    midline = (front['x'] - center['x'], front['y'] - center['y'])
    direction = midline / np.linalg.norm(midline)
    orientation = np.arctan2(direction[0], direction[1]) + np.pi / 2.0
    if np.isnan(orientation):
        orientation = 0.0
    return {
        "center": np.array([center['x'], center['y'], orientation]),
        "front": np.array([front['x'], front['y'], orientation]),
    }

def get_track_pixel(frame_data, front_index, center_index):
    image_nodes = frame_data['image_nodes']
    point = image_nodes[front_index]
    if point['x'] is None or point['y'] is None:
        return np.array([np.nan, np.nan])
    return np.array([point['x'], point['y']])