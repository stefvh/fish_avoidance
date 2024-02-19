import csv
import pandas as pd

file_id = "Export_1_2023-02-07T15_32_04_2868874"

folder_name = "data/BioTracker_Tracks"

file_name = folder_name + "/" + file_id + ".csv"

df = pd.read_csv(file_name, sep=";", comment="#", skiprows=3)

print(df)
