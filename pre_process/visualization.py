import cv2
import numpy as np

size_input = (width_input, height_input) = (2048, 2048)
size_output = (width_output, height_output) = (640, 480)


def rescale_position(position):
    x, y = position
    x = (x / width_input) * width_output
    y = (y / height_input) * height_output
    return x, y


def rescale_corners(corners):
    corners[:, 0] *= (width_output / 100)
    corners[:, 1] *= (height_output / 100)
    corners = corners.astype(int)
    return corners


def visualize(
    camera_filename,
    output_folder,
    output_filename,
    rectification_corners,
    robot_positions,
    fish_positions,
    trials,
    robot_states,
):
    vid_input = cv2.VideoCapture(camera_filename)

    vid_output = cv2.VideoWriter(
        f"{output_folder}/{output_filename}.mp4", cv2.VideoWriter_fourcc(*"MPEG"), 25, size_output
    )

    corners = rescale_corners(rectification_corners)

    color_default, color_robot, color_fish = (0, 255, 0), (0, 255, 255), (255, 0, 255)
    frame_number = 0
    while True:
        ret, frame = vid_input.read()
        if not ret:
            break

        frame = cv2.resize(frame, size_output)

        draw_y = 30
        draw_i = 1

        # Draw frame number
        cv2.putText(frame,
            "F: " + str(frame_number),
            (10, draw_i * draw_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            color_default,
            2,
            cv2.LINE_AA,
        )
        draw_i += 1

        # Draw trial number
        trial_number = None
        for index, row in trials.iterrows():
            if row["start_frame"] <= frame_number <= row["end_frame"]:
                trial_number = index
                break
        if trial_number is None:
            trial_number = "#"
        cv2.putText(frame,
            "T: " + str(trial_number),
            (10, draw_i * draw_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            color_default,
            2,
            cv2.LINE_AA,
        )
        draw_i += 1

        # Draw robot state. robot_states id pd.Dataframe with frame_number and state as columns
        state = robot_states[robot_states["frame_number"] == frame_number]["state"].values
        if len(state) > 0:
            text_state = state[0]
        else:
            text_state = "#"
        cv2.putText(frame,
            "R: " + text_state,
            (10, draw_i * draw_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            color_default,
            2,
            cv2.LINE_AA,
        )
        draw_i += 1

        # Draw legend of color_robot and color_fish
        cv2.putText(frame,
            "Robot",
            (10, draw_i * draw_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            color_robot,
            2,
            cv2.LINE_AA,
        )
        draw_i += 1
        cv2.putText(frame,
            "Fish",
            (10, draw_i * draw_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            color_fish,
            2,
            cv2.LINE_AA,
        )
        draw_i += 1

        # Draw rectification corners
        cv2.rectangle(
            frame,
            (corners[0][0], corners[0][1]),
            (corners[2][0], corners[2][1]),
            color_default,
            2,
        )

        # Draw robot and fish positions, if positions are not np.nan
        if frame_number < robot_positions.shape[0] and not np.isnan(robot_positions[frame_number]).any():
            x, y = rescale_position(robot_positions[frame_number])
            cv2.circle(frame, (int(x), int(y)), 6, color_robot, 1)
        if frame_number < fish_positions.shape[0] and not np.isnan(fish_positions[frame_number]).any():
            x, y = rescale_position(fish_positions[frame_number])
            cv2.circle(frame, (int(x), int(y)), 6, color_fish, 1)

        vid_output.write(frame)
        frame_number += 1

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        cv2.imshow("frame", frame)

    vid_output.release()
    vid_input.release()
    cv2.destroyAllWindows()
