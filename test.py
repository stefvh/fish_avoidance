import pickle as pkl
folder = "data/RoboFish_Commander_Data/JUNE 13 (SET 2)/20230613-165310"

with open(folder + "/parameters.pkl", "rb") as f:
    parameters = pkl.load(f)

print(parameters)