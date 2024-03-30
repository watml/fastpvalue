import os.path
from utils.args_DV import DV
import numpy as np
from utils.load_datasets import load_dataset
from utils.models import DVEstimator
import torch
import matplotlib.pyplot as plt
from scipy import stats
from utils.funcs import plot_curves
import seaborn as sns

##############
idx_range = np.arange(99, 10000, 100)
##############

trajs = []
for s in DV.seeds_test:
    data_saved = DV.test_saved.format(s)
    trajs.append(np.load(data_saved)["marginals"])
trajs = np.stack(trajs)
target = trajs.mean(axis=1).mean(axis=0)
target_norm = np.linalg.norm(target)

_, _, (X_test, y_test), _ = load_dataset(dataset=DV.dataset, n_test=DV.n_test)
X_test, y_test = X_test[:, 0, :, :, :], y_test[:, 0]
values_pred_corr = []
values_pred_err = []
for seed in DV.seeds_train:
    dir_cur = DV.dir_train.format(seed)
    tmp = DV.weights.copy()
    tmp /= tmp.sum()
    scalar = np.multiply((DV.n_valued + 1 - np.arange(1, DV.size_max + 1)) / DV.n_valued, tmp).sum()
    estimator = DVEstimator(scalar)
    estimator.double()
    model_saved = torch.load(os.path.join(dir_cur, "epoch30.model"))
    estimator.load_state_dict(model_saved["model_best_corr"])
    values_pred_corr.append(estimator.estimate(X_test, y_test))
    estimator.load_state_dict(model_saved["model_best_err"])
    values_pred_err.append(estimator.estimate(X_test, y_test))

values_pred_corr = np.stack(values_pred_corr)
err_pred_corr = np.linalg.norm(values_pred_corr - target[None, :], axis=1) / target_norm
corr_pred_corr = []
for v in values_pred_corr:
    res = stats.spearmanr(v, target)
    corr_pred_corr.append(res.correlation)
corr_pred_corr = np.tile(np.array(corr_pred_corr)[:, None], (1, len(idx_range)))
err_pred_corr = np.tile(err_pred_corr[:, None], (1, len(idx_range)))

values_pred_err = np.stack(values_pred_err)
err_pred_err = np.linalg.norm(values_pred_err - target[None, :], axis=1) / target_norm
corr_pred_err = []
for v in values_pred_err:
    res = stats.spearmanr(v, target)
    corr_pred_err.append(res.correlation)
corr_pred_err = np.tile(np.array(corr_pred_err)[:, None], (1, len(idx_range)))
err_pred_err = np.tile(err_pred_err[:, None], (1, len(idx_range)))


trajs = np.cumsum(trajs, axis=1) / np.arange(1, DV.num_marginal_per_seed + 1, 1)[None, :, None]
trajs = trajs[:, idx_range]
err_appr = np.linalg.norm(trajs - target[None, None, :], axis=2) / target_norm
corr_appr = np.empty((len(DV.seeds_test), len(idx_range)), dtype=np.float64)
for i in range(len(DV.seeds_test)):
    for j in range(len(idx_range)):
        res = stats.spearmanr(trajs[i, j], target)
        corr_appr[i, j] = res.correlation

curves = dict(
    err=(err_appr, err_pred_corr, err_pred_err),
    corr=(corr_appr, corr_pred_corr, corr_pred_err)
)
for key, curves in curves.items():
    labels = ["Monte Carlo", "TrELS-SC (ours)", "TrELS-RD (ours)"]
    x = idx_range * 2 / 1000
    if key == "err":
        ylabel = "relative difference"
        yscale = "log"
    else:
        yscale = "linear"
        ylabel = "Spearman correlation"

    clrs = sns.color_palette("tab10", 10)

    plot_curves(x, curves, os.path.join(DV.dir_fig, f"appr_{key}_{DV.dataset}.png"), labels=labels,
                xlabel="#utility evaluations per datum (k)", ylabel=ylabel, yscale=yscale,
                clrs=[clrs[1], clrs[-1], clrs[0]])


