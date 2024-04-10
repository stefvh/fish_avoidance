import argparse
import datetime
import json
import pickle
import os
import pandas as pd
import glob

import numpy as np
from priors.foldering import get_data_folders
from pre_process.arena_conversion import transform_corner_data
from pre_process.track_conversion import get_track_pixel, get_track_pose
from pre_process.visualization import visualize
from post_process.computation import compute_results
from utils.parser import get_default_parser
# from pre_process.rendering import render

def parse_args():
    parser = get_default_parser()
    parser.add_argument("--renew_trials", action='store_true')
    parser.add_argument("--renew_tracks", action='store_true')
    parser.add_argument("--renew_results", action='store_true')
    parser.add_argument("--renew_all", action='store_true')
    parser.add_argument("--save_video",action='store_true')

    args = parser.parse_args()
    return args


class Experiment:
    def __init__(self,
                 parent_folder,
                 experiment_datetime,
                 frequency=25,
                 renew_trials=False,
                 renew_tracks=False,
                 renew_results=False,
                 renew_all=False,
                 save_video=False,
                 **kwargs,) -> None:
        self.parent_folder = parent_folder
        self.experiment_datetime = experiment_datetime
        self.input_folder, self.output_folder = get_data_folders(parent_folder, experiment_datetime)
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
        
        self.frequency = frequency
        self.period = 1 / frequency

        self.renew_trials = renew_trials
        self.renew_tracks = renew_tracks
        self.renew_results = renew_results
        if renew_all:
            self.renew_trials = True
            self.renew_tracks = True
            self.renew_results = True

        filenames = glob.glob(self.input_folder + '/CameraCapture*.avi')
        if len(filenames) == 1:
            self.camera_filename = filenames[0]
        else:
            raise ValueError(f"No or multiple camera files found: {filenames}")

        print("Loading data...")
        self.load_data()

        print("Loading properties...")
        self.load_properties()

        print("Loading trials...")
        self.load_trials()

        print("Loading tracks...")
        self.load_tracks()

        if save_video:
            print("Rendering experiment...")
            self.show()
        
        print("Loading results...")
        self.load_results()
        
        # if save_rendering:
        #     print("Rendering tracking...")
        #     self.render_tracking(render_results)
        
        self.save()
    
    def load_data(self):
        folder = self.input_folder
        with open(folder + "/robot_directions.pkl", "rb") as f:
            self.robot_directions = pickle.load(f)
        with open(folder + "/robot_states.pkl", "rb") as f:
            self.robot_states = pickle.load(f)
        with open(folder + "/parameters.pkl", "rb") as f:
            self.parameters = pickle.load(f)
        
        self.experiment_start_datetime = self.robot_states["time"].iloc[0]

        self._extend_data()
    
    def _extend_data(self):
        init_datetime = self.robot_states["time"].iloc[0]

        self.robot_states["relative_time"] = self.robot_states["time"] - init_datetime
        self.robot_states["frame_number"] = (self.robot_states["relative_time"].dt.total_seconds() / self.period).astype(int)
    
    def load_properties(self):
        json_tracks = json.load(open(self.input_folder + '/tracks.json'))
        self.original_frame_count = json_tracks['original_frame_count']
        self.original_track_start = json_tracks['original_track_start']

        json_arena = json.load(open(self.input_folder + '/tracks.arena.json'))
        width_cm = json_arena['width_cm']
        height_cm = json_arena['height_cm']
        rectification_corners = json_arena['rectification_corners']
        self.rectification_corners = transform_corner_data(rectification_corners, width_cm, height_cm)
        tracking_area_corners = json_arena['tracking_area_corners']
        self.tracking_area_corners = transform_corner_data(tracking_area_corners, width_cm, height_cm)

        self.arena_xlim = (-50., 50.)
        self.arena_ylim = (-50., 50.)

        self._align_data_props()
    
    def _align_data_props(self):
        print("commander_frame_count:", self.robot_states["frame_number"].max())
        print("tracking_frame_count:", self.original_frame_count)

        foo = self.camera_filename.split("/")[-1].split(".")[0].split("CameraCapture")[1]
        camera_start_datetime = datetime.datetime.strptime(foo[:-1], 
                                                           "%Y-%m-%dT%H_%M_%S_%f")
        camera_start_datetime -= datetime.timedelta(hours=2)
        camera_start_datetime = camera_start_datetime.replace(tzinfo=datetime.timezone.utc)

        data_start_datetime = self.robot_states["time"].iloc[0]

        delta_datetime_start = camera_start_datetime - data_start_datetime
        delta_frames_start = delta_datetime_start / datetime.timedelta(seconds=0.04)

        delta_frames_start = -38

        delta = abs(int(delta_frames_start))
        print(delta_frames_start)

        if delta_frames_start < 0:
            # extend robot_states with state as NaN, but increasing frame_number for the first delta_frames_start
            new_states = []
            for i in range(delta):
                new_states.append({
                    "time": self.robot_states["time"].iloc[0] - datetime.timedelta(seconds=0.04 * (delta - i + 1)),
                    "state": np.nan,
                    "frame_number": i,
                })
            new_states = pd.DataFrame(new_states)
            self.robot_states["frame_number"] = self.robot_states["frame_number"] + delta
            self.robot_states = pd.concat([new_states, self.robot_states])

            print(self.robot_states.head())
            
        elif delta_frames_start > 0:
            # remove the first delta_frames_start rows from robot_states
            self.robot_states = self.robot_states.iloc[delta:]
        else:
            pass

    def load_tracks(self):
        if self.renew_tracks or \
           not os.path.exists(self.output_folder + "/robot_poses.pkl") or \
           not os.path.exists(self.output_folder + "/fish_poses.pkl") or \
           not os.path.exists(self.output_folder + "/robot_pixels.pkl") or \
           not os.path.exists(self.output_folder + "/fish_pixels.pkl"):
            print("...Starting track conversion")
            self._track_conversion()
            print("...Finished track conversion")
            print("...Starting track identities correction")
            self._track_identities_correction()
            print("...Finished track identities correction")
            print("...Starting track faults correction")
            self._track_faults_correction()
            print("...Finished track faults correction")
            print("...Starting track manual correction")
            self._track_manual_correction()
            print("...Finished track manual correction")
            with open(self.output_folder + "/robot_poses.pkl", "wb") as f:
                pickle.dump(self.robot_poses, f)
            with open(self.output_folder + "/fish_poses.pkl", "wb") as f:
                pickle.dump(self.fish_poses, f)
            with open(self.output_folder + "/robot_pixels.pkl", "wb") as f:
                pickle.dump(self.robot_pixels, f)
            with open(self.output_folder + "/fish_pixels.pkl", "wb") as f:
                pickle.dump(self.fish_pixels, f)
            with open(self.output_folder + "/manual_corrections.pkl", "wb") as f:
                pickle.dump(self.manual_corrections, f)
        else:
            with open(self.output_folder + "/robot_poses.pkl", "rb") as f:
                self.robot_poses = pickle.load(f)
            with open(self.output_folder + "/fish_poses.pkl", "rb") as f:
                self.fish_poses = pickle.load(f)
            with open(self.output_folder + "/robot_pixels.pkl", "rb") as f:
                self.robot_pixels = pickle.load(f)
            with open(self.output_folder + "/fish_pixels.pkl", "rb") as f:
                self.fish_pixels = pickle.load(f)
            with open(self.output_folder + "/manual_corrections.pkl", "rb") as f:
                self.manual_corrections = pickle.load(f)
            self.identities_corrected = os.path.exists(f"pre_process/identity_swaps/{self.experiment_datetime}.csv")
            self.manually_corrected = os.path.exists(f"pre_process/manual_poses/{self.experiment_datetime}.csv")
    
    @property
    def milling_tracks(self):
        sf = self.robot_states.loc[self.robot_states["state"] == "MILLING"]["frame_number"].iloc[0]
        ef = self.robot_states.loc[self.robot_states["state"] == "MILLING"]["frame_number"].iloc[-1]
        return self.robot_poses[sf:ef], self.fish_poses[sf:ef]
    
    @property
    def active_tracks(self):
        sf = self.trials["start_frame"].min()
        ef = self.trials["end_frame"].max()
        return self.robot_poses[sf:ef], self.fish_poses[sf:ef]
    
    def _track_conversion(self):
        json_tracks = json.load(open(self.input_folder + '/tracks.json'))

        robot_poses, fish_poses = np.zeros((self.original_frame_count, 3)), np.zeros((self.original_frame_count, 3))
        robot_pixels, fish_pixels = np.zeros((self.original_frame_count, 2)), np.zeros((self.original_frame_count, 2))
        front_index, center_index = json_tracks['skeleton']['front_index'], json_tracks['skeleton']['center_index']
        json_features = json_tracks['features']
        for frame_number in range(self.original_frame_count):
            if str(frame_number) in json_features:
                frame_datas = json_features[str(frame_number)]['features']
                for frame_data in frame_datas:
                    pose = get_track_pose(frame_data, front_index, center_index)
                    pixel = get_track_pixel(frame_data, front_index, center_index)
                    if frame_data['id'] == 1:   # fish
                        fish_poses[frame_number] = pose["center"]
                        fish_pixels[frame_number] = pixel
                    else:   # robot
                        robot_poses[frame_number] = pose["front"]
                        robot_pixels[frame_number] = pixel
            else:
                robot_poses[frame_number] = np.nan
                fish_poses[frame_number] = np.nan
        
        self.robot_poses = robot_poses
        self.fish_poses = fish_poses

        self.robot_pixels = robot_pixels
        self.fish_pixels = fish_pixels
    
    def _track_identities_correction(self):
        if not os.path.exists(f"pre_process/identity_swaps/{self.experiment_datetime}.csv"):
            self.identities_corrected = False
            raise UserWarning(f"Identity swaps file not found for {self.experiment_datetime} \n" +
                                "Are you sure there are no problems with identities?")
        else:
            cv_input = pd.read_csv(f"pre_process/identity_swaps/{self.experiment_datetime}.csv")
            frame_numbers = list(cv_input["frame_number"])
            for i in range(len(frame_numbers)):
                if i % 2 == 0:
                    start_frame = frame_numbers[i]
                    if i <= len(frame_numbers) - 2:
                        end_frame = frame_numbers[i+1]
                    else:
                        end_frame = self.original_frame_count
                    
                    buffer = np.copy(self.fish_poses[start_frame:end_frame])
                    self.fish_poses[start_frame:end_frame] = np.copy(self.robot_poses[start_frame:end_frame])
                    self.robot_poses[start_frame:end_frame] = buffer

                    buffer = np.copy(self.fish_pixels[start_frame:end_frame])
                    self.fish_pixels[start_frame:end_frame] = np.copy(self.robot_pixels[start_frame:end_frame])
                    self.robot_pixels[start_frame:end_frame] = buffer
            buffer = np.copy(self.fish_poses)
            self.fish_poses = np.copy(self.robot_poses)
            self.robot_poses = buffer

            buffer = np.copy(self.fish_pixels)
            self.fish_pixels = np.copy(self.robot_pixels)
            self.robot_pixels = buffer

            self.identities_corrected = True
    
    def _track_faults_correction(self):
        self.robot_poses, self.fish_poses = self.__track_faults_correction(self.robot_poses, self.fish_poses, 3)
        self.robot_pixels, self.fish_pixels = self.__track_faults_correction(self.robot_pixels, self.fish_pixels, 2)
    
    def __track_faults_correction(self, df_robot, df_fish, dimensions):
        for i in range(len(df_robot)):
            if df_robot[i, 0] == 0 and df_robot[i, 1] == 0:
                df_robot[i] = np.array([np.nan] * dimensions)
            if df_fish[i, 0] == 0 and df_fish[i, 1] == 0:
                df_fish[i] = np.array([np.nan] * dimensions)
        return df_robot, df_fish
    
    def calc_track_accuracy(self, poses):
        return 1 - np.sum(np.isnan(poses[:, 0])) / poses.shape[0]
    
    def _track_manual_correction(self):
        pre_mil_acc = self.calc_track_accuracy(self.milling_tracks[0])
        pre_act_acc = self.calc_track_accuracy(self.active_tracks[0])
        pre_tot_acc = self.calc_track_accuracy(self.robot_poses)

        self.manual_corrections = np.zeros((self.original_frame_count, 1), dtype=bool)
        if not os.path.exists(f"pre_process/manual_poses/{self.experiment_datetime}.csv"):
            self.manually_corrected = False
            raise UserWarning(f"Manual poses file not found for {self.experiment_datetime} \n" +
                                "Are you sure there are no positions missing? \n" +
                                "Note: Active tracking accuracy is {pre_accuracy:.2f}%.")
        else:
            cv_input = pd.read_csv(f"pre_process/manual_poses/{self.experiment_datetime}.csv")
            for _, row in cv_input.iterrows():
                frame_number = row["frame_number"]
                type = row["type"]
                x, y, z = row["x"], row["y"], row["z"]

                if type == "robot":
                    self.robot_poses[frame_number] = np.array([x, y, z])
                if type == "fish":
                    self.fish_poses[frame_number] = np.array([x, y, z])
                
                self.manual_corrections[frame_number] = True

            self.manually_corrected = True
        
        post_mil_acc = self.calc_track_accuracy(self.milling_tracks[0])
        post_act_acc = self.calc_track_accuracy(self.active_tracks[0])
        post_tot_acc = self.calc_track_accuracy(self.robot_poses)
        
        print(f"---> Total tracking accuracy improved from {pre_tot_acc:.4f}% to {post_tot_acc:.4f}%.")
        print(f"---> Active tracking accuracy improved from {pre_act_acc:.4f}% to {post_act_acc:.4f}%.")
        print(f"---> Milling tracking accuracy improved from {pre_mil_acc:.4f}% to {post_mil_acc:.4f}%.")
    
    def load_trials(self):
        if self.renew_trials or not os.path.exists(self.output_folder + "/trials.pkl"):
            print("...Starting automatic trial detection")
            self._automatic_trial_detection()
            print("...Finished automatic trial detection")
            print("...Starting trial validation")
            self._trial_validation()
            print("...Finished trial validation")
            print("...Starting trial extension")
            self._trial_extension()
            print("...Finished trial extension")
            with open(self.output_folder + "/trials.pkl", "wb") as f:
                pickle.dump(self.trials, f)
        else:
            with open(self.output_folder + "/trials.pkl", "rb") as f:
                self.trials = pickle.load(f)
        if self.renew_trials or not os.path.exists(self.output_folder + "/trials.csv"):
            self.trials.to_csv(self.output_folder + "/trials.csv")
        # print(self.trials)

    def _automatic_trial_detection(self):
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

        list_trial_start_datetime = []
        list_trial_end_datetime = []
        for datetime_point in self.robot_states["time"]:
            robot_state = self.robot_states.loc[self.robot_states["time"] == datetime_point]
            robot_state = robot_state["state"].iloc[0]
            if robot_state == "MILLING":
                continue

            if previous_state in ["APPROACHING_SLOW", "APPROACHING_FAST", "WALL_AVOIDANCE"] and robot_state == "INITIALIZING":
                counter += 1
                mid_datetime = datetime_point
                is_waiting = True
            
            if robot_state == "INITIALIZING":
                robot_direction = self.robot_directions.loc[self.robot_directions["time"] == datetime_point]
                if is_waiting:
                    if abs(robot_direction['dx'].iloc[0]) > 0.01 and abs(robot_direction['dy'].iloc[0]) > 0.01:
                        mid_datetime = datetime_point
                        is_waiting = False

            if (previous_state == "INITIALIZING" or previous_state == "MILLING") and robot_state in ["APPROACHING_SLOW", "APPROACHING_FAST"]:
                # New trial started and previous trial ended
                if counter > 0:
                    end_datetime = min(mid_datetime + reaction_timedelta, datetime_point)
                    list_trial_start_datetime.append(start_datetime)
                    list_trial_end_datetime.append(end_datetime)

                start_datetime = datetime_point

            previous_state = robot_state
        
        df_trials = pd.DataFrame({
            "start_datetime": list_trial_start_datetime,
            "end_datetime": list_trial_end_datetime,
        })
        # compute relative start_datetime and end_datetime
        df_trials["start_relative_time"] = df_trials["start_datetime"] - init_datetime
        df_trials["end_relative_time"] = df_trials["end_datetime"] - init_datetime
        # compute start_frame and end_frame
        df_trials["start_frame"] = (df_trials["start_relative_time"].dt.total_seconds() / self.period).astype(int)
        df_trials["end_frame"] = (df_trials["end_relative_time"].dt.total_seconds() / self.period).astype(int)

        df_trials.insert(0, "trial_id", range(len(df_trials)))

        self.trials = df_trials
    
    def _trial_validation(self):
        if not os.path.exists(f"pre_process/invalid_trials/{self.experiment_datetime}.csv"):
            self.trials_validated = False
            raise UserWarning(f"Invalid trials file not found for {self.experiment_datetime} \n" +
                               "Are you sure there are no trials to be removed?")
        else:
            cv_input = pd.read_csv(f"pre_process/invalid_trials/{self.experiment_datetime}.csv")
            trial_ids = list(cv_input["trial_id"])
            self.trials_original = self.trials.copy()
            for trial_id in trial_ids:
                self.trials = self.trials[self.trials["trial_id"] != trial_id]

            self.trials_validated = True
    
    def _trial_extension(self):
        if not os.path.exists(f"pre_process/extend_trials/{self.experiment_datetime}.csv"):
            self.trials_extended = False
            raise UserWarning(f"Extend trials file not found for {self.experiment_datetime} \n" +
                               "Are you sure there are no trials to be extended?")
        else:
            cv_input = pd.read_csv(f"pre_process/extend_trials/{self.experiment_datetime}.csv")
            trial_ids = list(cv_input["trial_id"])
            self.trials["end_frame_original"] = self.trials["end_frame"]
            offset = 5
            for trial_id in trial_ids:
                next_start_frame = self.trials.loc[self.trials["trial_id"] == trial_id + 1]["start_frame"]
                self.trials.loc[self.trials["trial_id"] == trial_id, "end_frame"] = next_start_frame.iloc[0] - offset

            self.trials_extended = True
    
    def show(self):
        if self.identities_corrected:
            output_filename = "output"
        else:
            output_filename = "output_raw"
        visualize(
            self.camera_filename,
            self.output_folder,
            output_filename,
            self.rectification_corners,
            self.robot_pixels,
            self.fish_pixels,
            self.trials,
            self.robot_states,
        )
    
    def load_results(self):
        sub_folder, filename = "results", "results"
        folder = f"{self.input_folder}/{sub_folder}"
        filepath = f"{folder}/{filename}.pkl"
        if self.renew_results or not os.path.exists(filepath):
            print("...Starting results computation")
            if not os.path.exists(folder):
                os.makedirs(folder)
            corners = np.array([
                [self.arena_xlim[0], self.arena_ylim[0]],
                [self.arena_xlim[1], self.arena_ylim[0]],
                [self.arena_xlim[1], self.arena_ylim[1]],
                [self.arena_xlim[0], self.arena_ylim[1]],
            ])
            self.results = compute_results(
                self.trials,
                self.fish_poses,
                self.robot_poses,
                self.period,
                corners,
                folder,
                filename,
            )
            print("...Finished results computation")
        else:
            with open(filepath, "rb") as f:
                self.results = pickle.load(f)
    
    # def render_tracking(self,
    #                     render_results=False):
    #     render(
    #         self.output_folder,
    #         "tracking",
    #         self.tracking_area_corners,
    #         self.robot_poses,
    #         self.fish_poses,
    #         self.trials,
    #         self.results,
    #     )

    def save(self):
        pickle.dump(self, open(self.output_folder + "/experiment.pkl", "wb"))


def get_experiment(**kwargs) -> Experiment:
    if kwargs["experiment_id"] is not None:
        return get_experiment_by_id(kwargs["experiment_id"])
    elif kwargs["parent_folder"] is not None and kwargs["experiment_datetime"] is not None:
        return get_experiment_by_time(kwargs["parent_folder"], kwargs["experiment_datetime"])
    else:
        raise ValueError(f"No valid arguments provided to find the experiment file: {kwargs}")


def get_experiment_by_time(parent_folder,
                   experiment_datetime) -> Experiment:
    _, folder = get_data_folders(parent_folder, experiment_datetime)
    if not os.path.exists(folder + "/experiment.pkl"):
        raise FileNotFoundError(f"No experiment file found at {folder}")
    else:
        return pickle.load(open(folder + "/experiment.pkl", "rb"))

def get_experiment_by_id(id) -> Experiment:
    filepath = "pre_process/experiment_ids.csv"
    if not os.path.exists(filepath):
        raise FileNotFoundError("No experiment_ids.csv file found")
    else:
        df = pd.read_csv(filepath)
        row = df.loc[df["id"] == id]
        parent_folder = row["parent_folder"].iloc[0]
        experiment_datetime = row["experiment_datetime"].iloc[0]

        return get_experiment_by_time(parent_folder, experiment_datetime)


if __name__ == "__main__":
    args = parse_args()
    variables = vars(args)

    experiment = Experiment(**variables)
