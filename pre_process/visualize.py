import cv2
import numpy as np
import os
from time import time as timer


tl = [55,60]
tr = [555,60]
bl = [60,435]
br = [560,435]

box = [tl, tr, bl, br]


def rescale_x_y_to_box(x,y):

    width = box[1][0] - box[0][0]
    height = box[2][1] - box[0][1]

    # first we scale the new value to the range of the box 
    x = (x/2048) *  width
    y = (y/2048) * height

    # then we add the top left corner of the box to the new value
    x = x + box[0][0]
    y = y + box[0][1]

    return x, y

def adjust_coo_system(x, y):
    # x = (x + 50)/100 * 2048
    # y = (100-(y + 50))/100 * 2048 
    x = x * 2048/100
    y = (100 - y) * 2048/100
    return x, y

def visualize_position(frame, x, y, color):
    x, y = adjust_coo_system(x, y)
    x, y = rescale_x_y_to_box(x, y)
    cv2.circle(frame, (int(x), int(y)), 6, color, 1)

def visualize(robot_poses,
        fish_poses,
        camera_filename):
    vid = cv2.VideoCapture(camera_filename)

    # get number of frames
    length = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = vid.get(cv2.CAP_PROP_FPS)
    print(length, fps)

    frame_counter = 0
    frame_offset = 38

    # add a slider to the window for the frame_offset
    cv2.namedWindow('frame')
    cv2.createTrackbar('frame_offset', 'frame', frame_offset, 300, lambda x: None)

    frame_skip = 20
    for i in range(frame_skip):
        ret, frame = vid.read()
        # frame_counter += 1

    while True:
        starttime = cv2.getTickCount()
        start = timer()
        ret, frame = vid.read()
        if not ret:
            break

        frame = cv2.resize(frame, (640, 480))
        cv2.rectangle(frame, (int(tl[0]), int(tl[1])), (int(br[0]), int(br[1])), (0, 255, 0), 2)

        frame_offset = cv2.getTrackbarPos('frame_offset', 'frame')

        #fish_poses is a pandas dataframe
        x, y = fish_poses["x"].iloc[frame_counter], fish_poses["y"].iloc[frame_counter]
        visualize_position(frame, x, y, (0, 255, 255))

        x, y, = robot_poses["x"].iloc[frame_counter], robot_poses["y"].iloc[frame_counter]
        visualize_position(frame, x, y, (255, 0, 255))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # diff = timer() - start
        # while  diff < fps / 1000:
        #     diff = timer() - start

        frame_counter += 1

        # show the current fps in the window
        totaltime = (cv2.getTickCount() - starttime) / cv2.getTickFrequency()
        play_fps = 1 / totaltime
        cv2.putText(frame, f'FPS: {play_fps:.2f}', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.imshow('frame', frame)


