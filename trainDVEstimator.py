import numpy as np
from utils.funcs import set_numpy_seed, set_torch_seed, os_lock
from utils.args_DV import DV
from utils.load_datasets import load_dataset
from utils.models import DVEstimator
import torch
from collections import defaultdict
from scipy import stats
import os
import copy

def _sampler(DV, seed_perm):
    tmp = np.arange(1, DV.size_max + 1, dtype=np.float64)
    weights = np.multiply(np.divide((DV.n_valued + 1) - tmp, tmp), DV.weights)
    weights /= weights.sum()

    subsets = np.zeros((DV.batch_size, DV.n_valued + 1), dtype=bool)
    ues_batch = np.empty(DV.batch_size, dtype=np.float64)
    count = 0
    seeds = DV.seeds_training.copy()
    with set_numpy_seed(seed_perm):
        np.random.shuffle(seeds)
    s_range = np.arange(1, DV.size_max + 1)
    pos_range = np.arange(DV.n_valued + 1)
    for seed in seeds:
        np.random.seed(seed)
        data_saved = DV.training_saved.format(seed)
        ues = np.load(data_saved)["ues"]
        ues_pos = 0
        for _ in range(DV.num_sample_per_seed):
            s = np.random.choice(s_range, p=weights)
            pos = np.random.choice(pos_range, size=s, replace=False)
            subsets[count, pos] = True
            ues_batch[count] = ues[ues_pos]
            count += 1
            ues_pos += 1
            if count == DV.batch_size:
                yield subsets, ues_batch
                count = 0
                subsets.fill(False)


def load_val_dataset(DV):
    values_aggr = np.zeros(DV.n_val, dtype=np.float64)
    for s in DV.seeds_val:
        data_saved = DV.val_saved.format(s)
        values_aggr += np.load(data_saved)["marginals"].mean(axis=0)
    return values_aggr / len(DV.seeds_val)


def main(seed_main, dir_cur, DV):
    print(dir_cur)
    os.makedirs(dir_cur, exist_ok=True)
    log_file = os.path.join(dir_cur, "log.txt")
    with open(log_file, "w"):
        pass

    values_val = load_val_dataset(DV)
    norm_val = np.linalg.norm(values_val)
    (X_valued, y_valued), (X_all, y_all), _, _ = load_dataset(dataset=DV.dataset, n_valued=DV.n_valued,
                                                              n_perf=DV.n_perf + DV.n_val)
    X_val, y_val = X_all[DV.n_perf:], y_all[DV.n_perf:]


    tmp = DV.weights.copy()
    tmp /= tmp.sum()
    scalar = np.multiply((DV.n_valued + 1 - np.arange(1, DV.size_max + 1)) / DV.n_valued, tmp).sum()
    with set_torch_seed(seed_main):
        model = DVEstimator(scalar, X_valued[:, 0, :, :, :], y_valued[:, 0])
    model.double()
    optimizer = torch.optim.Adam(model.parameters(), lr=DV.lr_DVE)

    records = defaultdict(list)
    error_best = np.inf
    correlation_best = -2.0
    batch_count = 0
    for epoch in range(DV.epochs):
        for batch in _sampler(DV, epoch + seed_main):

            loss = model(batch[0], batch[1])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_count += 1

            if batch_count % DV.interval_report == 0:
                records["loss"].append(loss.item())
                with torch.no_grad():
                    values_predicted = model.estimate(X_val, y_val)
                error_cur = np.linalg.norm(values_predicted - values_val) / norm_val
                records["error"].append(error_cur)
                res = stats.spearmanr(values_predicted, values_val)
                correlation_cur = res.correlation
                records["correlation"].append(correlation_cur)

                mesg = f"T{batch_count} | loss {loss.item()} | error {error_cur} | correlation {correlation_cur}"
                # print(mesg)
                with open(log_file, "a") as f:
                    f.write(mesg + "\n")

                if correlation_cur > correlation_best:
                    correlation_best = correlation_cur
                    model_best_corr = copy.deepcopy(model.state_dict())
                if error_cur < error_best:
                    error_best = error_cur
                    model_best_err = copy.deepcopy(model.state_dict())

        if (epoch + 1) % DV.epoch_save == 0:
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_best_corr": model_best_corr,
                    "model_best_err": model_best_err,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "records": records,
                },
                os.path.join(dir_cur, f"epoch{epoch + 1}.model")
            )

    np.savez_compressed(os.path.join(dir_cur, "results.npz"), **records)


if __name__ == "__main__":
    for seed in DV.seeds_train:
        dir_cur = DV.dir_train.format(seed)
        with os_lock(os.path.join(dir_cur, "results.npz")) as lock_state:
            if lock_state:
                main(seed, dir_cur, DV)



