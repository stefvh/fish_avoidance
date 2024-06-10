from pre_process import (
    check_velocities,
    detect_tracking_errors,
    verify_tracking_fixes
)
import warnings
warnings.filterwarnings('ignore')

def run(experiment,
        start_frame,
        end_frame,):
    detect_tracking_errors.run(experiment, start_frame + 1, end_frame - 1)
    verify_tracking_fixes.run(experiment, start_frame + 1, end_frame - 1)
    check_velocities.run(experiment)