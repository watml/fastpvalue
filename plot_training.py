from utils.args_DV import DV
import os
import numpy as np
from utils.funcs import plot_curves
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

idx_pick = np.arange(19, 3000, 20)


fig_saved = os.path.join(DV.dir_fig, "{}" + f"_{DV.dataset}.png")
loss = []
error = []
corr = []

for seed in DV.seeds_train:
    dir_cur = DV.dir_train.format(seed)
    data_saved = os.path.join(dir_cur, "results.npz")
    data = np.load(data_saved)
    loss.append(data["loss"])
    error.append(data["error"])
    corr.append(data["correlation"])

num_point = len(loss[0])

curves = dict(
    loss=(np.stack(loss)[:, idx_pick], "empirical loss"),
    error=(np.stack(error), "relative difference"),
    corr=(np.stack(corr), "Spearman correlation")
)

x = np.arange(1, num_point + 1, 1)[idx_pick] * DV.interval_report / 1000

for key, value in curves.items():
    if key == "loss":
        plot_curves(x, [value[0]], fig_saved.format(key), ylabel=value[1], xlabel="#batches (k)", yscale="log")
    else:
        clrs = sns.color_palette("tab20", 7) + sns.color_palette("tab20b", 4)
        curve, ylabel = value

        fig, ax = plt.subplots(figsize=(32, 24))
        # ax2 = ax.twinx()

        curve_pick = curve[0, idx_pick]

        curve_mean = curve[:, idx_pick].mean(axis=0)
        curve_std = curve[:, idx_pick].std(axis=0)

        ax.plot(x, curve_pick, color=clrs[10], linewidth=10, label="specific")

        ax.plot(x, curve_mean, color=clrs[0], linewidth=10, label="average")
        ax.fill_between(x, curve_mean - curve_std, curve_mean + curve_std, alpha=0.2, facecolor=clrs[0])
        if key == "error":
            ax.set_yscale("log")
            curve_best = np.minimum.accumulate(curve, axis=1)[:, idx_pick]
        else:
            curve_best = np.maximum.accumulate(curve, axis=1)[:, idx_pick]
        curve_best_mean = curve_best.mean(axis=0)
        curve_best_std = curve_best.std(axis=0)

        ax.plot(x, curve_best_mean, color=clrs[6], linewidth=10, label="best")
        ax.fill_between(x, curve_best_mean - curve_best_std, curve_best_mean + curve_best_std, alpha=0.2,
                        facecolor=clrs[6])

        ax.set_xlabel('#batches (k)', fontsize=100)
        ax.set_ylabel(ylabel, fontsize=100)

        ax.tick_params(axis='x', labelsize=80)
        ax.tick_params(axis='y', labelsize=80)

        ax.grid()
        ax.legend(fontsize=100, framealpha=0.5)

        plt.savefig(fig_saved.format(key), bbox_inches='tight')
        plt.close(fig)