import numpy as np
import fish_models


def run_live_female_female_constant_speed_couzin_model(
    fish_pose: np.ndarray, fish_direction: np.ndarray, freq: int
):
    time_step_s = 1 / freq
    world_bounds = None  # TODO: set world bounds

    model = fish_models.models.released.parametrized.LiveFemaleFemaleConstantSpeedCouzinModel(
        time_step_s=time_step_s,
    )

    view = np.zeros((1, 4), dtype=np.float32)
    view[0, 0] = fish_pose[0]
    view[0, 1] = fish_pose[1]
    view[0, 2] = fish_direction[0]
    view[0, 3] = fish_direction[1]

    speed, turn = model.choose_action(view, 0)
    speed *= time_step_s
    turn *= time_step_s

    return speed, turn


def run_cluster_model(
    fish_pose: np.ndarray,
    freq: int,
):
    model = fish_models.ModelStorage.load_from_model_name(
        "ClusterModel_v1_1", device="cpu"
    )
    buffer_size = 10
    frame_buffer = np.zeros((1, buffer_size, model.raycast.view_size))

    view = model.raycast.cast(np.array([fish_pose]), 0)
    frame_buffer[0, :-1] = frame_buffer[0, 1:]
    frame_buffer[0, -1] = view
    speed, turn = model.choose_action(
        np.concatenate(frame_buffer[0, model.stack_frames]), 9000
    )
    speed /= freq
    turn /= freq

    return speed, turn


def run(
    model_name: str,
    fish_pose: np.ndarray,
    fish_direction: np.ndarray,
    freq: int,
):
    if model_name == "LiveFemaleFemaleConstantSpeedCouzinModel":
        speed, turn = run_live_female_female_constant_speed_couzin_model(
            fish_pose, fish_direction, freq
        )
    elif model_name == "ClusterModel":
        speed, turn = run_cluster_model(fish_pose, freq)
    else:
        raise ValueError(f"Model name {model_name} not recognized.")
    
    return speed, turn
