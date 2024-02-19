import argparse
import datetime
import pickle
import os
import pandas as pd
import robofish.io
import cv2
import glob

import numpy as np

from pre_process.trial import Trial 
from pre_process.visualize import visualize

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent_folder", type=str, default="JUNE 13 (SET 2)")
    parser.add_argument("--experiment_datetime", type=str, default="20230613-165310")

    args = parser.parse_args()
    return args

class Experiment:
    def __init__(self,
                 parent_folder,
                 experiment_datetime,
                 detect_trials=False,
                 render_trajectories=False,) -> None:
        self.input_folder = f"data/RoboFish_Commander_Data/{parent_folder}/{experiment_datetime}"
        self.output_folder = f"{self.input_folder}/output"
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

        filenames = glob.glob(self.input_folder + '/CameraCapture*.avi')
        if len(filenames) == 1:
            self.camera_filename = filenames[0]
        else:
            raise ValueError(f"No or multiple camera files found: {filenames}")

        print("Loading data...")
        self.load_data()

        if not os.path.exists(self.output_folder + "/robot_updated_poses.pkl"):
            print("Updating data...")
            self.robot_poses = self.update_data(self.robot_poses, self.robot_poses["time"])
            with open(self.output_folder + "/robot_updated_poses.pkl", "wb") as f:
                pickle.dump(self.robot_poses, f)
            self.fish_poses = self.update_data(self.fish_poses, self.robot_poses["time"])
            with open(self.output_folder + "/fish_updated_poses.pkl", "wb") as f:
                pickle.dump(self.fish_poses, f)
        else:
            with open(self.output_folder + "/robot_updated_poses.pkl", "rb") as f:
                self.robot_poses = pickle.load(f)
            with open(self.output_folder + "/fish_updated_poses.pkl", "rb") as f:
                self.fish_poses = pickle.load(f)
        
        print("Cropping data...")
        # self.crop_data()

        if detect_trials:
            if not os.path.exists(self.output_folder + "/trials.pkl"):
                print("Detecting trials automatically...")
                self.trials = self.automatic_trial_detection()
                with open(self.output_folder + "/trials.pkl", "wb") as f:
                    pickle.dump(self.trials, f)
            else:
                with open(self.output_folder + "/trials.pkl", "rb") as f:
                    self.trials = pickle.load(f)

        if render_trajectories:
            print("Rendering trajectories...")
            trajectories_file = self.output_folder + "/trajectories.hdf5"
            self.render_trajectories(trajectories_file)
    
    def load_data(self):
        folder = self.input_folder
        with open(folder + "/robot_poses.pkl", "rb") as f:
            self.robot_poses = pickle.load(f)
        with open(folder + "/robot_directions.pkl", "rb") as f:
            self.robot_directions = pickle.load(f)
        with open(folder + "/robot_states.pkl", "rb") as f:
            self.robot_states = pickle.load(f)
        with open(folder + "/fish_poses.pkl", "rb") as f:
            self.fish_poses = pickle.load(f)
        with open(folder + "/parameters.pkl", "rb") as f:
            self.parameters = pickle.load(f)
        
        self.start_datetime = self.robot_poses["time"].iloc[0]
    
    def update_data(self, df_poses, datetime_points):
        ids = np.unique(df_poses['id'])
        print(f"IDs: {ids}, type: {type}")
        if len(ids) == 1:
            return df_poses

        # create new_df_poses based on datetime_points
        new_df_poses = pd.DataFrame()
        new_df_poses["time"] = datetime_points
        new_data = np.zeros((len(datetime_points), 4))

        for i, time in enumerate(datetime_points):
            data_point = np.zeros(4)
            data_points_by_id = {}
            previous_id = None
            for j, identifier in enumerate(ids):
                df = df_poses.loc[
                    (df_poses['time'] == time) & 
                    (df_poses['id'] == identifier)
                ]
                if not df.empty:
                    data_point = [
                        float(identifier),
                        float(df['x'].iloc[0]),
                        float(df['y'].iloc[0]),
                        float(df['z'].iloc[0]),
                    ]
                    data_points_by_id[identifier] = data_point
            if len(data_points_by_id) == 1:
                new_data[i] = data_point
                previous_id = list(data_points_by_id.keys())[0]
            elif len(data_points_by_id) > 1:
                print(f"Multiple IDs at same time point: {time}")
                if previous_id is None:
                    # Choose pose closest to the left corner
                    left_corner = np.array([0.45, 0.45])
                    distances = np.zeros(len(data_points_by_id))
                    for j, identifier in enumerate(data_points_by_id.keys()):
                        distances[j] = np.linalg.norm(left_corner - data_points_by_id[identifier][1:3])
                    previous_id = list(data_points_by_id.keys())[np.argmin(distances)]
                new_data[i] = data_points_by_id[previous_id]
                # raise ValueError("Multiple IDs at same time point")
            else:
                if i == 0:
                    raise ValueError("First pose is NaN")
                else:
                    new_data[i] = new_data[i-1]
        new_df_poses["id"] = new_data[:, 0].astype(int)
        new_df_poses["x"] = new_data[:, 1]
        new_df_poses["y"] = new_data[:, 2]
        new_df_poses["z"] = new_data[:, 3]

        return new_df_poses
    
    def crop_data(self):
        foo = self.camera_filename.split("/")[-1].split(".")[0].split("CameraCapture")[1]
        camera_start_datetime = datetime.datetime.strptime(foo[:-1], 
                                                           "%Y-%m-%dT%H_%M_%S_%f")
        camera_start_datetime -= datetime.timedelta(hours=2)
        camera_start_datetime = camera_start_datetime.replace(tzinfo=datetime.timezone.utc)

        data_start_datetime = self.robot_poses["time"].iloc[0]

        delta_datetime_start = camera_start_datetime - data_start_datetime
        delta_frames_start = delta_datetime_start / datetime.timedelta(seconds=0.04)
        n_frames = abs(int(delta_frames_start))
        print(n_frames)

        if delta_frames_start < 0:
            # cut out poses from data
            self.robot_poses = self.robot_poses.iloc[n_frames:]
            self.fish_poses = self.fish_poses.iloc[n_frames:]
        else:
            # cut out frames from video
            raise NotImplementedError("Cutting out frames from video")
        
        video = cv2.VideoCapture(self.camera_filename)
        frame_count = video.get(cv2.CAP_PROP_FRAME_COUNT)
        fps = video.get(cv2.CAP_PROP_FPS)
        video.release()

        bar = datetime.timedelta(seconds=frame_count / fps)
        camera_end_datetime = camera_start_datetime + bar
        data_end_datetime = self.robot_poses["time"].iloc[-1]

        delta_datetime_end = camera_end_datetime - data_end_datetime
        delta_frames_end = delta_datetime_end / datetime.timedelta(seconds=0.04)
        n_frames = abs(int(delta_frames_end))
        print(n_frames)

        if delta_frames_end < 0:
            # cut out poses from data
            self.robot_poses = self.robot_poses.iloc[:-n_frames]
            self.fish_poses = self.fish_poses.iloc[:-n_frames]

        return
    
    def automatic_trial_detection(self):
        counter = 0
        previous_state = "MILLING"
        reaction_timedelta = datetime.timedelta(
            hours=0,
            minutes=0,
            seconds=0,
            microseconds=0,
        )
        init_datetime = self.robot_states["time"].iloc[0]
        mid_datetime = None
        is_waiting = False
        trials = []
        for datetime_point in self.robot_poses["time"]:
            robot_state = self.robot_states.loc[self.robot_states["time"] == datetime_point]
            robot_state = robot_state["state"].iloc[0]
            if robot_state == "MILLING":
                continue

            if previous_state in ["APPROACHING_SLOW", "APPROACHING_FAST"] and robot_state == "INITIALIZING":
                counter += 1
                mid_datetime = datetime_point
                is_waiting = True
            
            if robot_state == "INITIALIZING":
                robot_direction = self.robot_directions.loc[self.robot_directions["time"] == datetime_point]
                if is_waiting:
                    if abs(robot_direction['dx'].iloc[0]) > 0.01 and abs(robot_direction['dy'].iloc[0]) > 0.01:
                        mid_datetime = datetime_point
                        is_waiting = False

            if previous_state == "INITIALIZING" or previous_state == "MILLING" and robot_state in ["APPROACHING_SLOW", "APPROACHING_FAST"]:
                # New trial started and previous trial ended
                if counter > 0:
                    end_datetime = min(mid_datetime + reaction_timedelta, datetime_point)
                    trial = Trial(
                        id=len(trials),
                        experiment_init_datetime=init_datetime,
                        start_datetime=start_datetime,
                        end_datetime=end_datetime,
                    )
                    trial.filter_data(
                        self.robot_poses,
                        self.robot_directions,
                        self.robot_states,
                        self.fish_poses,
                        self.parameters,
                    )
                    trials.append(trial)

                start_datetime = datetime_point

            previous_state = robot_state
        return trials
    
    def show(self):
        visualize(
            self.robot_poses,
            self.fish_poses,
            self.camera_filename,
        )

    def _format_trajectories(self, df_poses):
        trajectories = np.zeros((1, df_poses.shape[0], 4))
        for index, row in df_poses.iterrows():
            pose = np.float32([np.nan] * 4)
            pose[0] = max(min(float(row['x'] - 50.), 50.45), -50.45)
            pose[1] = max(min(float(row['y'] - 50.), 50.45), -50.45)
            pose[2] = np.cos((row['z']))
            pose[3] = np.sin((row['z']))
            trajectories[0, index] = pose
        return trajectories
    
    def render_trajectories(self, trajectories_file):
        calendar_time_points = self.robot_poses['time']

        io_robot_poses = self._format_trajectories(self.robot_poses)
        io_fish_poses = self._format_trajectories(self.fish_poses)

        f = robofish.io.File(
            trajectories_file,
            mode="w",
            world_size_cm=[100., 100.],
            frequency_hz=25,
            calendar_time_points=calendar_time_points,
        )
        robots = f.create_multiple_entities(category="robot", poses=io_robot_poses)
        fish = f.create_multiple_entities(category="fish", poses=io_fish_poses)

        f.validate()
        f.close()
    
    def draw_trajectories(self):
        cap_file = self.input_folder + "/CameraCapture2023-06-13T18_53_09_3499405.avi"
        cap = cv2.VideoCapture(cap_file)
    
        output = cv2.VideoWriter(self.input_folder + "/output.avi", cv2.VideoWriter_fourcc(*'MPEG'), 25, (2048, 2048))

        index = 0
        runner = True
        while(runner): 
            ret, frame = cap.read() 
            if(ret): 
                # x = self.robot_poses["x"].iloc[index]
                # y = self.robot_poses["y"].iloc[index]
                x, y = 0, 0
                frame = cv2.circle(frame, (x,y), radius=30, color=(0, 0, 255), thickness=3)

                if index >= 25 * 60 * 8 + 20 and index <= 25 * 60 * 8 + 60:
                    cv2.imwrite(f"{self.input_folder}/{index}.png", frame)

                # x = self.fish_poses["x"].iloc[index]
                # y = self.fish_poses["y"].iloc[index]
                # cv2.circle(frame, (x,y), radius=10.0, color=(0, 255, 0))
                
                # writing the new frame in output 
                output.write(frame) 
                if cv2.waitKey(1) & 0xFF == ord('s'): 
                    break
            else: 
                break
            index += 1
            runner = index < 25 * 60 * 8 + 60
  
        cv2.destroyAllWindows() 
        output.release() 
        cap.release() 

    def save(self):
        pickle.dump(self, open(self.output_folder + "/experiment.pkl", "wb"))


if __name__ == "__main__":
    args = parse_args()
    variables = vars(args)

    parent_folder = variables["parent_folder"]
    experiment_datetime = variables["experiment_datetime"]
    input_folder = f"data/RoboFish_Commander_Data/{parent_folder}/{experiment_datetime}"
    output_folder = f"{input_folder}/output"
    experiment = Experiment(**variables)
    experiment.show()
    # if not os.path.exists(output_folder + "/experiment.pkl"):
    #     experiment = Experiment(**variables)
    #     experiment.save()
    # else:
    #     experiment = pickle.load(open(output_folder + "/experiment.pkl", "rb"))
    #     # experiment.render_trajectories(experiment.output_folder + "/trajectories.hdf5")
