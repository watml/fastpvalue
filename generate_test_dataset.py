import os
NUM_THREAD = 1
os.environ["OMP_NUM_THREADS"] = f"{NUM_THREAD}"
os.environ["OPENBLAS_NUM_THREADS"] = f"{NUM_THREAD}"
os.environ["MKL_NUM_THREADS"] = f"{NUM_THREAD}"
os.environ["VECLIB_MAXIMUM_THREADS"] = f"{NUM_THREAD}"
os.environ["NUMEXPR_NUM_THREADS"] = f"{NUM_THREAD}"
import numpy as np
import multiprocessing as mp
from utils.funcs import vd_tqdm, os_lock
from tqdm import tqdm
from utils.load_datasets import load_dataset
from utils.utilityFuncs import gameTraining
import traceback
from utils.args_DV import DV


class apprDV:
    def __init__(self, *, game_func, game_args, num_player, weights, num_marginal_per_seed, n_process, X_extra, y_extra,
                 seed=None, file_prog=None):
        self.game_func = game_func
        self.game_args = game_args
        self.num_player = num_player
        self.weights = weights / weights.sum()
        self.size_max = len(weights)
        self.num_marginal_per_seed = num_marginal_per_seed
        self.n_process = n_process - 1 # leave one for doing aggregation
        self.X_extra, self.y_extra = X_extra, y_extra
        self.seed = seed
        self.file_prog = file_prog

        self.marginals=np.empty((num_marginal_per_seed, len(y_extra)), dtype=np.float64)
        self.pos_cur = 0

    def run(self):
        if self.n_process > 1:
            with mp.Pool(self.n_process) as pool:
                process = pool.imap(self.do_job, self.sampler())
                for marginal_chips in vd_tqdm(process, total=self.num_marginal_per_seed, miniters=self.n_process,
                                              file_prog=self.file_prog):
                    self.aggregate(marginal_chips)
        else:
            for subset in tqdm(self.sampler(), total=self.num_marginal_per_seed):
                self.aggregate(self.do_job(subset))
        self.pos_cur = 0
        return self.marginals

    def aggregate(self, marginals_chip):
        self.marginals[self.pos_cur] = marginals_chip
        self.pos_cur += 1

    def sampler(self):
        np.random.seed(self.seed)
        s_range = np.arange(self.size_max)
        pos_range = np.arange(self.num_player)
        subset = np.empty(self.num_player, dtype=bool)
        for _ in range(self.num_marginal_per_seed):
            s = np.random.choice(s_range, p=self.weights)
            pos = np.random.choice(pos_range, size=s, replace=False)
            subset.fill(False)
            subset[pos] = True
            yield subset.copy()

    def do_job(self, subset):
        game = self.game_func(**self.game_args)
        marginals_chip = np.empty(len(self.y_extra), dtype=np.float64)
        ue_base = game.evaluate(subset)
        for k, (X_extra, y_extra) in enumerate(zip(self.X_extra, self.y_extra)):
            z_extra = (X_extra.unsqueeze(0), y_extra.unsqueeze(0))
            marginals_chip[k] = game.dist_evaluate(subset, z_extra) - ue_base
        return marginals_chip


if __name__ == "__main__":
    try:
        (X_valued, y_valued), (X_perf, y_perf), (X_test, y_test), num_class = load_dataset(dataset=DV.dataset,
                                                                                         n_valued=DV.n_valued,
                                                                                         n_perf=DV.n_perf,
                                                                                         n_test=DV.n_test)
        game_args = dict(X_valued=X_valued, y_valued=y_valued, X_perf=X_perf, y_perf=y_perf, metric=DV.metric,
                         arch="LeNet", num_class=num_class, lr=DV.lr_game)
        runner = apprDV(game_func=gameTraining, game_args=game_args, num_player=DV.n_valued, weights=DV.weights,
                        num_marginal_per_seed=DV.num_marginal_per_seed, n_process=DV.n_process, X_extra=X_test,
                        y_extra=y_test, file_prog=os.path.join(DV.root, "generating_test.txt"))

        for seed in DV.seeds_test:
            data_saved = DV.test_saved.format(seed)
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

