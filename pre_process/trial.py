class Trial:
    def __init__(self,
                 id,
                 experiment_init_datetime,
                 start_datetime,
                 end_datetime) -> None:
        self.id = id
        self.start_datetime = start_datetime
        self.end_datetime = end_datetime
        self.duration = end_datetime - start_datetime

        self.start_relative_time = self.start_datetime - experiment_init_datetime
        self.end_relative_time = self.end_datetime - experiment_init_datetime 
    
    def filter_data(self,
                    robot_poses,
                    robot_directions,
                    robot_states,
                    fish_poses,
                    parameters):
        # Filter by time
        self.robot_poses = robot_poses.loc[(robot_poses["time"] >= self.start_datetime) & (robot_poses["time"] <= self.end_datetime)]
        self.robot_directions = robot_directions.loc[(robot_directions["time"] >= self.start_datetime) & (robot_directions["time"] <= self.end_datetime)]
        self.robot_states = robot_states.loc[(robot_states["time"] >= self.start_datetime) & (robot_states["time"] <= self.end_datetime)]
        self.fish_poses = fish_poses.loc[(fish_poses["time"] >= self.start_datetime) & (fish_poses["time"] <= self.end_datetime)]
        self.parameters = parameters.loc[(parameters["time"] >= self.start_datetime) & (parameters["time"] <= self.end_datetime)]

        # Reset index
        self.robot_poses.reset_index(drop=True, inplace=True)
        self.robot_directions.reset_index(drop=True, inplace=True)
        self.robot_states.reset_index(drop=True, inplace=True)
        self.fish_poses.reset_index(drop=True, inplace=True)
        self.parameters.reset_index(drop=True, inplace=True)
        
        # Compute dependent variables
        robot_poses_extended = self.robot_poses.copy()
        robot_poses_extended['relative_time_s'] = self.robot_poses['time'].apply(lambda x: (x - self.start_datetime).total_seconds())
        self.robot_poses = robot_poses_extended
    
    def __str__(self) -> str:
        out = f"""Trial #{self.id}
start_datetime \t= {self.start_datetime}
end_datetime \t= {self.end_datetime}
duration \t= {self.duration}\n"""
        if hasattr(self, "start_relative_time"):
            out += f"start_relative_time \t= {self.start_relative_time}\n"
        if hasattr(self, "end_relative_time"):
            out += f"end_relative_time \t= {self.end_relative_time}\n"
        out += "----------------------------------------------------"
        return out
    

