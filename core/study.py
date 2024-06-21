import os
import pandas as pd
from core.experiment import Experiment, get_experiment_by_id
from priors.foldering import get_data_folders

class Study:
    def __init__(self):
        self.experiment_ids = [
            0, 1, 2, 3, 4,
        ]

        self.initialize()
        self.set_data()
    
    def initialize(self):
        self.experiments = {experiment_id: self.load_experiment(experiment_id) for experiment_id in self.experiment_ids}
        self.merge_motion_quantities()
        self.set_folders()
    
    def reset(self):
        self.initialize()
        self.set_data()
        self.save_data()
    
    @property
    def list_experiments(self) -> list[Experiment]:
        return list(self.experiments.values())

    def load_experiment(self, experiment_id) -> Experiment:
        return get_experiment_by_id(experiment_id)
    
    @property
    def motion_quantities(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        return self.robot_mq_df, self.fish_mq_df
    
    def _merge_motion_quantities(self, mqs):
        dfs = []
        for experiment_id, mq in zip(self.experiment_ids, mqs):
            df = mq.df
            df['experiment_id'] = experiment_id
            dfs.append(df)
        return pd.concat(dfs)

    def merge_motion_quantities(self):
        robot_mqs = [e.robot_motion_quantities for e in self.list_experiments]
        self.robot_mq_df = self._merge_motion_quantities(robot_mqs)
        fish_mqs = [e.fish_motion_quantities for e in self.list_experiments]
        self.fish_mq_df = self._merge_motion_quantities(fish_mqs)
        rel_mqs = [e.relative_motion_quantities for e in self.list_experiments]
        self.rel_mq_df = self._merge_motion_quantities(rel_mqs)

    def renew_experiments(self):
        for experiment in self.list_experiments:
            print(f"Renewing experiment {experiment.id}")
            experiment.derive_quantities()
            experiment.save()
        self.reset()

    def set_data(self):
        data = self.rel_mq_df
        data = data.dropna()
        self.data = data

    def save_data(self):
        self.data.to_csv('core/study_data.csv', index=False)
    
    def set_folders(self):
        output_folder, _ = get_data_folders(self.list_experiments[0].parent_folder, "ALL")
        self.output_folder = output_folder
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        self.plot_output_folder = f"{output_folder}/plots"
        if not os.path.exists(self.plot_output_folder):
            os.makedirs(self.plot_output_folder)

study = Study()