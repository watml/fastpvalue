import os
from compare_estimators import *
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from matplotlib.ticker import MaxNLocator
from utils.funcs import export_legend

# see section Qualitative in https://matplotlib.org/stable/users/explain/colors/colormaps.html
clrs = sns.color_palette("tab20", 20)
clrs += sns.color_palette("tab20b", 20)
dict_color = dict(
    GELS_paired=18,
    GELS_shapley_paired=22,
    GELS_ranking_paired=0,
    sampling_lift_paired=12,
    WSL_paired=37,
    kernelSHAP_paired=6,
    unbiased_kernelSHAP=34,
    group_testing=4,
    ARM=2,
    complement=10,
    permutation=9,
    MSR_paired=8,
    AME_paired=16,
    simSHAP=31,
)
dict_color_tmp = dict_color.copy()
for key, value in dict_color_tmp.items():
    if "_paired" in key:
        dict_color.update({key[:-7] : value})
xticks = range(nue_track_avg, nue_avg + 1, nue_track_avg)

fig_format = os.path.join(root, "fig", "dataset={};metric={};semivalue={}_{};{}.png")
os.makedirs(os.path.join(root, "fig"), exist_ok=True)

# plot the legend
dict_names = dict(
    GELS="GELS (ours)",
    GELS_shapley="GELS-Shapley (ours)",
    GELS_ranking="GELS-R (ours)",
    sampling_lift="sampling lift",
    kernelSHAP="kernelSHAP",
    unbiased_kernelSHAP="unbiased kernelSHAP",
    group_testing="group testing",
    ARM="ARM",
    complement="complement",
    permutation="permutation",
    WSL="weighted sampling lift",
    AME="AME",
    MSR="MSR",
)
for estimator, label in dict_names.items():
    plt.plot([], [], label=label, color=clrs[dict_color[estimator]], linewidth=30)
legend = plt.legend(ncol=5, fontsize=100)
export_legend(legend, os.path.join(root, "fig", "legend.png"))
# plot the figures
for dataset in datasets:
    for metric in metrics:
        for key in semivalues.keys():
            path_cur = dir_format.format(metric, key[0], key[1], "exact_value")
            value_saved = os.path.join(root, dataset, path_cur, "values.npz")
            values_exact = np.load(value_saved)["values"]
            norm_exact = np.linalg.norm(values_exact)


            error_dict = defaultdict(list)
            correlation_dict = defaultdict(list)
            estimators = semivalues[key]
            for estimator in estimators:
                path_cur = os.path.join(root, dataset, dir_format.format(metric, key[0], key[1], estimator))

                estimates_collect = []
                for seed in seeds:
                    estimate_saved = os.path.join(path_cur, f"seed={seed}.npz")
                    estimates_collect.append(np.load(estimate_saved)["estimates_traj"])

                all_tmp = np.stack(estimates_collect)
                if "ranking" not in estimator:
                    err_tmp = np.linalg.norm(all_tmp - values_exact[None, None, :], axis=2) / norm_exact
                    error_dict[estimator] = err_tmp.mean(axis=0)

                num_seed, num_traj = len(seeds), nue_avg // nue_track_avg
                ranking_tmp = np.empty((num_seed, num_traj), dtype=np.float64)
                for i in range(num_seed):
                    for j in range(num_traj):
                        res = stats.spearmanr(all_tmp[i, j], values_exact)
                        ranking_tmp[i, j] = res.correlation
                correlation_dict[estimator] = ranking_tmp.mean(axis=0)

            for is_corr, d in enumerate([error_dict, correlation_dict]):
                fig, ax = plt.subplots(figsize=(32, 24))
                plt.grid()

                for estimator, traj in d.items():
                    if "paired" in estimator:
                        if is_corr:
                            ax.plot(xticks, traj, linestyle="--", c=clrs[dict_color[estimator]], linewidth=10)
                            ax.set_yscale("logit", use_overline=True)
                        else:
                            plt.semilogy(xticks, traj, linestyle="--", c=clrs[dict_color[estimator]], linewidth=10)
                    else:
                        if is_corr:
                            ax.plot(xticks, traj, label=estimator, c=clrs[dict_color[estimator]], linewidth=10)
                            ax.set_yscale("logit", use_overline=True)
                        else:
                            plt.semilogy(xticks, traj, label=estimator, c=clrs[dict_color[estimator]], linewidth=10)
                ax.tick_params(axis='x', labelsize=80)
                ax.tick_params(axis='y', labelsize=80)


                plt.xlabel("#utility evaluations per datum", fontsize=100)
                if is_corr:
                    plt.ylabel("Spearman correlation", fontsize=100)
                else:
                    plt.ylabel("relative difference", fontsize=100)
                if is_corr:
                    fig_saved = fig_format.format(dataset, metric, key[0], key[1], "correlation")
                else:
                    fig_saved = fig_format.format(dataset, metric, key[0], key[1], "error")
                plt.savefig(fig_saved, bbox_inches='tight')
                plt.close(fig)




