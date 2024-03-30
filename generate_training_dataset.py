import os
NUM_THREAD = 1
os.environ["OMP_NUM_THREADS"] = f"{NUM_THREAD}"
os.environ["OPENBLAS_NUM_THREADS"] = f"{NUM_THREAD}"
os.environ["MKL_NUM_THREADS"] = f"{NUM_THREAD}"
os.environ["VECLIB_MAXIMUM_THREADS"] = f"{NUM_THREAD}"
os.environ["NUMEXPR_NUM_THREADS"] = f"{NUM_THREAD}"
import numpy as np
import multiprocessing as mp
from utils.funcs import *
from tqdm import tqdm
from utils.load_datasets import load_dataset
from utils.utilityFuncs import gameTraining
import traceback
from utils.args_DV import DV


class generator_training:
    def __init__(self, *, game_func, game_args, n_valued, weights, num_sample_per_seed, sample_per_proc, n_process,
                 seed=None, file_prog=None):
        self.game_func = game_func
        self.game_args = game_args
        self.n_valued = n_valued
        self.size_max = len(weights)
        self.num_sample_per_seed = num_sample_per_seed
        self.sample_per_proc = sample_per_proc
        self.n_process = n_process - 1 # leave one for doing aggregation
        self.seed = seed
        self.file_prog = file_prog

        self.ues = np.empty(self.num_sample_per_seed, dtype=np.float64)
        self.pos_cur = 0

        tmp = np.arange(1, self.size_max + 1, dtype=np.float64)
        self.weights = np.multiply(np.divide((self.n_valued + 1) - tmp, tmp), weights)
        self.weights /= self.weights.sum()

    def run(self):
        if self.n_process > 1:
            with mp.Pool(self.n_process) as pool:
                process = pool.imap(self.do_job, self.sampler())
                for results in vd_tqdm(process, total=-(-self.num_sample_per_seed // self.sample_per_proc),
                                       miniters=self.n_process, maxinterval=float('inf'), file_prog=self.file_prog):
                    self.aggregate(results)
        else:
            for subsets in tqdm(self.sampler(), total=-(-self.num_sample_per_seed // self.sample_per_proc)):
                self.aggregate(self.do_job(subsets))
        self.pos_cur = 0
        return self.ues

    def aggregate(self, results):
        total = len(results)
        self.ues[self.pos_cur:(self.pos_cur + total)] = results
        self.pos_cur += total

    def sampler(self):
        np.random.seed(self.seed)
        s_range = np.arange(1, self.size_max + 1)
        pos_range = np.arange(self.n_valued + 1)

        count = 0
        subsets = np.zeros((self.sample_per_proc, self.n_valued + 1), dtype=bool)
        for _ in range(self.num_sample_per_seed):
            s = np.random.choice(s_range, p=self.weights)
            pos = np.random.choice(pos_range, size=s, replace=False)
            subsets[count, pos] = True
            count += 1
            if count == self.sample_per_proc:
                yield subsets.copy()
                subsets.fill(False)
                count = 0
        if count:
            yield subsets[:count]

    def do_job(self, subsets):
        game = self.game_func(**self.game_args)
        ues = np.empty(len(subsets), dtype=np.float64)
        for k, subset in enumerate(subsets):
            ues[k] = game.evaluate(subset)
        return ues



if __name__ == "__main__":
    try:
        (X_valued, y_valued), (X_perf, y_perf), _, num_class = load_dataset(dataset=DV.dataset, n_valued=DV.n_valued,
                                                                          n_perf=DV.n_perf)
        game_args = dict(X_valued=X_valued, y_valued=y_valued, X_perf=X_perf, y_perf=y_perf, metric=DV.metric, arch="LeNet",
                         num_class=num_class, lr=DV.lr_game)
        generator = generator_training(game_func=gameTraining, game_args=game_args, n_valued=DV.n_valued,
                                       weights=DV.weights, num_sample_per_seed=DV.num_sample_per_seed,
                                       sample_per_proc=DV.sample_per_proc, n_process=DV.n_process,
                                       file_prog=os.path.join(DV.root, "generating_training.txt"))

        for seed in DV.seeds_training:
            data_saved = DV.traing_saved.format(seed)
            with os_lock(data_saved) as lock_state:
                if lock_state:
                    generator.seed = seed
                    ues = generator.run()
                    np.savez_compressed(data_saved, ues=ues)
    except:
        traceback.print_exc()
        with open(os.path.join(DV.root, "err.txt"), "a") as f:
            f.write("\n")
            traceback.print_exc(file=f)

