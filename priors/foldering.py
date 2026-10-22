ROOT_INPUT_FOLDER = "data/0_DEV"

def get_data_folders(parent_folder, experiment_datetime):
    input_folder = f"{ROOT_INPUT_FOLDER}/{parent_folder}/{experiment_datetime}"
    output_folder = f"{input_folder}/output"

    return input_folder, output_folder