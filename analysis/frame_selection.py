
def get_peak_frames(results):
    # For each trial_id in results, find the highest frame number where state in "APPROACHING*"
    peak_frames = []

    for trial_id in results["trial_id"].unique():
        trial_results = results[results["trial_id"] == trial_id]
        dip_distance = trial_results["robot_distance"].min()
        dip_idx = trial_results["robot_distance"].idxmin()
        dip_frame = trial_results["frame_number"][dip_idx]

        t_frames = [dip_frame]

        threshold = 2.0
        # add frames going back from dip_frame until the robot_distance is above or below threshold
        for i in range(1, 100):
            if dip_idx - i < 0 or dip_idx + i >= len(trial_results):
                break

            dist = trial_results["robot_distance"][dip_idx - i]
            if dist > dip_distance + threshold or dist < dip_distance - threshold:
                t_frames.append(trial_results["frame_number"][dip_idx - i])
                break
        # add frames going forward from dip_frame until the robot_distance is above or below threshold
        for i in range(1, 100):
            if dip_idx - i < 0 or dip_idx + i >= len(trial_results):
                break

            dist = trial_results["robot_distance"][dip_idx + i]
            if dist > dip_distance + threshold or dist < dip_distance - threshold:
                t_frames.append(trial_results["frame_number"][dip_idx + i])
                break

        frame_window = 10
        t_min = min(t_frames)
        t_max = max(t_frames)

        peak_frames += list(range(t_min - frame_window, t_max + frame_window))

    return peak_frames