import os
NUM_THREAD = 1
os.environ["OMP_NUM_THREADS"] = f"{NUM_THREAD}"
os.environ["OPENBLAS_NUM_THREADS"] = f"{NUM_THREAD}"
os.environ["MKL_NUM_THREADS"] = f"{NUM_THREAD}"
os.environ["VECLIB_MAXIMUM_THREADS"] = f"{NUM_THREAD}"
os.environ["NUMEXPR_NUM_THREADS"] = f"{NUM_THREAD}"
from generate_training_dataset import *
import numpy as np
from utils.funcs import os_lock
from utils.load_datasets import load_dataset
from utils.utilityFuncs import gameTraining
import traceback
from utils.args_DV import DV
from generate_test_dataset import apprDV


if __name__ == "__main__":
    try:
        (X_valued, y_valued), (X_all, y_all), _, num_class = load_dataset(dataset=DV.dataset, n_valued=DV.n_valued,
                                                                          n_perf=DV.n_perf + DV.n_val)
        X_perf, y_perf = X_all[:DV.n_perf], y_all[:DV.n_perf]
        X_val, y_val = X_all[DV.n_perf:][:, None, :, :, :], y_all[DV.n_perf:][:, None]
        game_args = dict(X_valued=X_valued, y_valued=y_valued, X_perf=X_perf, y_perf=y_perf, metric=DV.metric,
                         arch="LeNet", num_class=num_class, lr=DV.lr_game)
        runner = apprDV(game_func=gameTraining, game_args=game_args, num_player=DV.n_valued, weights=DV.weights,
                        num_marginal_per_seed=DV.num_marginal_per_seed, n_process=DV.n_process, X_extra=X_val,
                        y_extra=y_val, file_prog=os.path.join(DV.root, "generating_val.txt"))

        for seed in DV.seeds_val:
            data_saved = DV.val_saved.format(seed)
            with os_lock(data_saved, log=os.path.join(DV.root, "log.txt")) as lock_state:
                if lock_state:
                    runner.seed = seed
                    marginals = runner.run()
                    np.savez_compressed(data_saved, marginals=marginals)
    except:
        traceback.print_exc()
        with open(os.path.join(DV.root, "err.txt"), "a") as f:
            f.write("\n")
            traceback.print_exc(file=f)

