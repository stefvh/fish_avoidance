import cv2
import numpy as np
import os

def rescale_x_y_to_box(x,y, box):

    width = box[1][0] - box[0][0]
    height = box[2][1] - box[0][1]

    # first we scale the new value to the range of the box 
    x = (x/2048) *  width
    y = (y/2048) * height

    # then we add the top left corner of the box to the new value
    x = x + box[0][0]
    y = y + box[0][1]

    return x, y

if __name__ == '__main__':
    root = 'TESTING_DATA_BERLIN'

    tl = [55,60]
    tr = [555,60]
    bl = [60,435]
    br = [560,435]

    b = [tl, tr, bl, br]
    # load the npy files
    fish = np.load(os.path.join(root, 'io_fish_poses.npy'))
    robot = np.load(os.path.join(root, 'io_robot_poses.npy'))

    print(fish.shape)

    vid = cv2.VideoCapture(os.path.join(root, 'CameraCapture2023-06-13T18_53_09_3499405.avi'))

    # get number of frames
    length = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = vid.get(cv2.CAP_PROP_FPS)
    print(length, fps)

    frame_counter = 0
    frame_offset = 38

    # add a slider to the window for the frame_offset
    cv2.namedWindow('frame')
    cv2.createTrackbar('frame_offset', 'frame', frame_offset, 300, lambda x: None)

    for i in range(20):
        ret, frame = vid.read()
        # frame_counter += 1


    while True:
        starttime = cv2.getTickCount()
        ret, frame = vid.read()
        if not ret:
            break


        frame_offset = cv2.getTrackbarPos('frame_offset', 'frame')


        x, y , dx, dy = fish[0][frame_counter]
        x = (x + 50)/100 * 2048
        y = (100-(y + 50))/100 * 2048 
        # print(x, y )
        # cv2.circle(frame, (int(x), int(y)), 10, (255, 255, 255), -1)

        x_r, y_r, dx_r, dy_r = robot[0][frame_counter]
        x_r = (x_r + 50)/100 * 2048
        y_r = (100-(y_r + 50))/100 * 2048 

        x_s, y_s = rescale_x_y_to_box(x, y, b)
        x_rs, y_rs = rescale_x_y_to_box(x_r, y_r, b)

        frame = cv2.resize(frame, (640, 480))
        cv2.rectangle(frame, (int(tl[0]), int(tl[1])), (int(br[0]), int(br[1])), (0, 255, 0), 2)
        cv2.circle(frame, (int(x_s), int(y_s)), 6, (255, 0, 255), 1)
        cv2.circle(frame, (int(x_rs), int(y_rs)), 6, (0, 255, 255), 1)

        frame_counter += 1

        # show the current fps in the window
        totaltime = (cv2.getTickCount() - starttime) / cv2.getTickFrequency()
        play_fps = 1 / totaltime
        cv2.putText(frame, f'FPS: {play_fps:.2f}', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


