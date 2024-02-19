import pickle as pkl

folder = "data/RoboFish_Commander_Data/JUNE 13 (SET 2)/20230613-165310"

with open(folder + "/fish_poses.pkl", "rb") as f:
    fish_poses = pkl.load(f)

    print(fish_poses)