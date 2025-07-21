
from core.experiment import Experiment
from core.study import study


if __name__ == "__main__":
    df = study.data.copy()
    print(df.shape)
    df = df.dropna(subset=["rev", "apv", "min_wall_distance_fish", "distance"])
    print(df.shape)
    df = df.dropna()
    print(df.shape)