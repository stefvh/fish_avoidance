from matplotlib import pyplot as plt
from core.experiment import Experiment
from core.study import study
import seaborn as sns

def run():
    fig, axes = plt.subplots(2, 6, figsize=(10, 20))
    fish_speed_lim = 25

    robot_mq, fish_mq = study.motion_quantities
    robot_mq['speed'].plot.hist(ax=axes[0, 0], bins=20)
    df = fish_mq[fish_mq['speed'] < fish_speed_lim]
    df['speed'].plot.hist(ax=axes[1, 0], bins=20)

    for i, experiment in enumerate(study.list_experiments):
        robot_mq = experiment.robot_motion_quantities
        fish_mq = experiment.fish_motion_quantities
        robot_mq.df['speed'].plot.hist(ax=axes[0, i+1], bins=20)
        df = fish_mq.df[fish_mq.df['speed'] < fish_speed_lim]
        max_fish_speed = fish_mq.df['speed'].max()
        axes[1, i+1].annotate(f'Max: {max_fish_speed:.2f}', xy=(0.5, 0.5), xycoords='axes fraction', ha='center', va='center')
        df['speed'].plot.hist(ax=axes[1, i+1], bins=20, density=True)
        sns.kdeplot(df['speed'], ax=axes[1, i+1], color='orange')
    
    for i, experiment in enumerate(study.list_experiments):
        axes[0, i+1].set_title(f'Experiment {experiment.id}')
        axes[1, i+1].set_title(f'Experiment {experiment.id}')

    for ax in axes.flatten():
        ax.set_yticks([])
        ax.set_ylabel('')
    for ax in axes[0, :].flatten():
        ax.set_xlim(0, 32.5)
    for ax in axes[1, :].flatten():
        ax.set_xlim(0, fish_speed_lim)
    axes[0, 0].set_title('Robot')
    axes[1, 0].set_title('Fish')
    for ax in axes[:, 0]:
        ax.set_ylabel('Frequency')
    plt.savefig(f"{study.plot_output_folder}/velocities.pdf", bbox_inches='tight')

if __name__ == '__main__':
    run()
