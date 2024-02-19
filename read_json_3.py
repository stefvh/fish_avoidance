import json
import os

import cv2
import numpy as np

# Read the JSON file
data = json.load(open('full_tracks.json'))
arena = json.load(open('full_tracks.arena.json'))

# print(data.keys())
print(arena)
rectification_corners = arena['rectification_corners']
tracking_area_corners = arena['tracking_area_corners']

print(data['skeleton'])
print(data['original_frame_count'])
print(data['original_track_start'])
# print(data['features'].keys())
# print(data['features']['30'].keys())
# print(data['features']['30']['features'])
# print(data['features']['30']['features'][0])

frame_numbers = data['features'].keys()
frame_numbers = sorted([int(x) for x in frame_numbers])
print(frame_numbers)

# print(frame_numbers)
# # check if duplicates
# print(len(frame_numbers) == len(set(frame_numbers)))


vid = cv2.VideoCapture('full_tracks.avi')

length = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
fps = vid.get(cv2.CAP_PROP_FPS)
print(length, fps)

print("frame_count: ", data['original_frame_count'])
frame_start = data['original_track_start']
print("frame_start: ", frame_start)
if frame_numbers[1] - frame_numbers[0] != 1:
    frame_start = frame_numbers[1]

frame_counter = frame_numbers[1]

front_index = data['skeleton']['front_index']
center_index = data['skeleton']['center_index']

data = data['features']

colors = {'fish': (255, 0, 255), 'robot': (0, 255, 255)}

begin_offset = 0 # 25 * 10
vid.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_start-begin_offset))

width, height = (640, 480)

def get_pose(frame_data):
    world_nodes = frame_data['world_nodes']
    front = world_nodes[front_index]
    center = world_nodes[center_index]
    midline = (front.x - center.x, front.y - center.y)
    direction = midline / np.linalg.norm(midline)
    orientation = np.arctan2(direction[0], direction[1]) + np.pi / 2.0
    if np.isnan(orientation):
        # happens if center == front
        orientation = 0.0
    return center.x, center.y, orientation

def do_stuff(frame, frame_data):
    if frame_data['id'] == 1:
        label = 'fish'
    else:
        label = 'robot'
    
    color = colors[label]
    
    image_nodes = frame_data['image_nodes']

    # Average over image_nodes
    list_x, list_y = [], []
    for e in image_nodes:
        if e['x'] is not None and e['y'] is not None:
            list_x.append(e['x'])
            list_y.append(e['y'])
    mean_x, mean_y = np.mean(list_x), np.mean(list_y)

    # Take center image_node
    center_x = image_nodes[center_index]['x']
    center_y = image_nodes[center_index]['y']

    x, y = center_x, center_y
    x = (x/2048) *  width
    y = (y/2048) * height

    cv2.circle(frame, (int(x), int(y)), 6, color, 1)

    return

offset_counter = 0

while True:
    starttime = cv2.getTickCount()
    ret, frame = vid.read()
    if not ret:
        break

    frame = cv2.resize(frame, (width, height))
    # frame = cv2.resize(frame, (2048, 2048))

    # recompute corners between [0, 1] into frame size
    np_ref_corners_x = np.array([e['x'] * width for e in rectification_corners])
    np_ref_corners_y = np.array([e['y'] * height for e in rectification_corners])
    np_ref_corners = np.array([np_ref_corners_x, np_ref_corners_y], dtype=np.int32).T
    cv2.rectangle(frame, (np_ref_corners[0][0], np_ref_corners[0][1]), (np_ref_corners[2][0], np_ref_corners[2][1]), (0, 255, 0), 2)

    if offset_counter >= begin_offset:
        if frame_counter in frame_numbers:
            if len(data[str(frame_counter)]['features']) == 2:
                do_stuff(frame, data[str(frame_counter)]['features'][0])
                do_stuff(frame, data[str(frame_counter)]['features'][1])

            frame_counter += 1
    
    offset_counter += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    cv2.imshow('frame', frame)


