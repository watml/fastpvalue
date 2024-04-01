import numpy as np
import itertools
from scipy import special
import sys
import multiprocessing as mp
from tqdm import tqdm
from utils.funcs import vd_tqdm

class runEstimator:
    def __init__(self, *, estimator, n_process, semivalue, semivalue_param, game_func, game_args, num_player, nue_avg,
                 nue_per_proc, nue_track_avg, estimator_seed=2024, file_prog=None):
        self.estimator = estimator
        self.n_process = n_process - 1 # one process is used for aggregating results
        self.file_prog = file_prog
        self.semivalue = semivalue
        self.semivalue_param = semivalue_param
        self.game_func = game_func
        self.game_args = game_args
        self.estimator_seed = estimator_seed
        self.num_player = num_player

        # the number of utility evaluations used to do estimation on average (divided by the number of players)
        self.nue_avg = nue_avg

        # the number of utility evaluations each process will run in one batch.
        self.nue_per_proc = nue_per_proc

        # record the estimates of all players after using nue_track_avg, 2*nue_track_avg, ..., utility evaluations on average.
        self.nue_track_avg = nue_track_avg

    def run(self):
        estimator_args = dict(
            semivalue=self.semivalue,
            semivalue_param=self.semivalue_param,
            game_func=self.game_func,
            game_args=self.game_args,
            num_player=self.num_player,
            nue_avg=self.nue_avg,
            nue_per_proc=self.nue_per_proc,
            nue_track_avg=self.nue_track_avg,
            estimator_seed=self.estimator_seed
        )
        estimator = getattr(sys.modules[__name__], self.estimator)(**estimator_args)
        print(f"The number of utility evalutions each process runs in one batch is {estimator.nue_per_proc_run}")
        if self.n_process > 1:
            with mp.Pool(self.n_process) as pool:
                process = pool.imap(estimator.run, estimator.sampling())
                for chunk in vd_tqdm(process, total=-(-estimator.num_sample//estimator.batch_size),
                                  miniters=self.n_process, maxinterval=float('inf'), file_prog=self.file_prog):
                    estimator.aggregate(chunk)
        else:
            for samples in tqdm(estimator.sampling(), total=-(-estimator.num_sample//estimator.batch_size)):
                estimator.aggregate(estimator.run(samples))
        return estimator.finalize()


class estimatorTemplate:
    def __init__(self, *, semivalue, semivalue_param, game_func, game_args, num_player, nue_avg, nue_per_proc, nue_track_avg,
                 estimator_seed):
        self.semivalue = semivalue
        self.semivalue_param = semivalue_param
        self.game_func = game_func
        self.game_args = game_args
        self.num_player = num_player
        self.nue_avg = nue_avg
        self.nue_per_proc = nue_per_proc
        self.nue_track_avg = nue_track_avg
        self.estimator_seed = estimator_seed

        num_traj = self.nue_avg // self.nue_track_avg
        self.values_traj = np.empty((num_traj, self.num_player), dtype=np.float64)
        self.pos_traj = 0
        self.buff = self.interval_track = self.batch_size = None
        self.pos_buffer = 0
        self.samples = None

        self.lock_switch = True
        self.switch_state = False

    @property
    def switch(self):
        return self.switch_state

    @switch.setter
    def switch(self, state):
        if not self.lock_switch:
            self.switch_state = state

    @property
    def buffer_size(self):
        return self.interval_track + self.batch_size - 1

    def run(self):
        pass

    def _init_indiv(self):
        pass

    def sampling(self):
        self._init_indiv()
        np.random.seed(self.estimator_seed)

        count = 0
        for _ in range(self.num_sample):
            if not self.switch:
                self.samples[count] = self._generator()
                self.switch = True
            else:
                self.samples[count] = 1 - self.samples[count - 1]
                self.switch = False
            count += 1
            if count == self.batch_size:
                yield self.samples.copy()
                count = 0
        if count:
            yield self.samples[:count]

    def _generator(self):
        pass

    def aggregate(self, results_collect):
        self.buffer[self.pos_buffer:self.pos_buffer + len(results_collect)] = results_collect
        self.pos_buffer += len(results_collect)
        num_collect = self.pos_buffer // self.interval_track
        if num_collect:
            for i in range(num_collect):
                self._process(self.buffer[i*self.interval_track:(i+1)*self.interval_track])
                self.values_traj[self.pos_traj] = self._estimate()
                self.pos_traj += 1
            num_left = self.pos_buffer - (i + 1) * self.interval_track
            self.buffer[:num_left] = self.buffer[(i + 1) * self.interval_track:self.pos_buffer]
            self.pos_buffer = num_left

    def finalize(self):
        if self.pos_buffer:
            self._process(self.buffer[:self.pos_buffer])
            values_final = self._estimate()
        else:
            values_final = self.values_traj[-1]
        return values_final, self.values_traj

    def _process(self, inputs):
        pass

    def _estimate(self):
        pass

    def distribution_cardinality(self):
        if self.semivalue == "shapley":
            weights = np.full(self.num_player, 1. / self.num_player, dtype=np.float64)
        elif self.semivalue == "weighted_banzhaf":
            weights = np.ones(self.num_player, dtype=np.float64)
            for k in range(self.num_player):
                for i in range(k):
                    weights[k] *= (self.num_player - 1 - i) / (i + 1) * self.semivalue_param * (1 - self.semivalue_param)
                weights[k] *= (1 - self.semivalue_param) ** (self.num_player - 1 - 2 * k)
        elif self.semivalue == "beta_shapley":
            alpha, beta = self.semivalue_param
            weights = np.ones(self.num_player, dtype=np.float64)
            tmp_range = np.arange(1, self.num_player, dtype=np.float64)
            weights *= np.divide(tmp_range, tmp_range + (alpha + beta - 1)).prod()
            for s in range(self.num_player):
                r_cur = weights[s]
                tmp_range = np.arange(1, s + 1, dtype=np.float64)
                r_cur *= np.divide(tmp_range + (beta - 1), tmp_range).prod()
                tmp_range = np.arange(1, self.num_player - s, dtype=np.float64)
                r_cur *= np.divide((alpha - 1) + tmp_range, tmp_range).prod()
                weights[s] = r_cur
        else:
            raise NotImplementedError(f"Check {self.semivalue}")
        return weights


class exact_value(estimatorTemplate):
    def __init__(self, **kwargs):
        super(exact_value, self).__init__(**kwargs)
        self.values = np.zeros(self.num_player, dtype=np.float64)
        self.num_sample = 2 ** (self.num_player - 1)
        self.batch_size = -(-self.nue_per_proc // (2 * self.num_player))
        self.nue_per_proc_run = self.batch_size * 2 * self.num_player

    def sampling(self):
        count = 0
        samples = np.empty((self.batch_size, self.num_player-1), dtype=bool)
        for subset in itertools.product([True, False], repeat=self.num_player-1):
            samples[count] = subset
            count += 1
            if count == self.batch_size:
                yield samples.copy()
                count = 0
        if count:
            yield samples[:count]

    def run(self, samples):
        weights = np.empty(self.num_player, dtype=np.float64)
        for i in range(self.num_player):
            if self.semivalue == "shapley":
                weights[i] = special.beta(self.num_player - i, i + 1)
            elif self.semivalue == "weighted_banzhaf":
                weights[i] = (self.semivalue_param ** i) * ((1 - self.semivalue_param) ** (self.num_player - 1 - i))
            elif self.semivalue == "beta_shapley":
                weights[i] = 1
                alpha, beta = self.semivalue_param
                for k in range(1, i+1):
                    weights[i] *= (beta+k-1) / (alpha+beta+k-1)
                for k in range(i+1, self.num_player):
                    weights[i] *= (alpha+k-i-1) / (alpha+beta+k-1)
            else:
                raise NotImplementedError(f"Check {self.semivalue}")

        game = self.game_func(**self.game_args)
        fragment = np.zeros(self.num_player)
        right_index = np.zeros(self.num_player, dtype=bool)
        left_index = np.ones_like(right_index)
        for sample in samples:
            weight = weights[sample.sum()]
            right_index[:self.num_player - 1] = sample
            left_index[:self.num_player - 1] = sample
            fragment[-1] += weight * (game.evaluate(left_index) - game.evaluate(right_index))
            for player in range(self.num_player - 1):
                right_index[-1], right_index[player] = right_index[player], right_index[-1]
                left_index[-1], left_index[player] = left_index[player], left_index[-1]
                fragment[player] += weight * (game.evaluate(left_index) - game.evaluate(right_index))
                right_index[-1], right_index[player] = right_index[player], right_index[-1]
                left_index[-1], left_index[player] = left_index[player], left_index[-1]
        return fragment

    def aggregate(self, fragment):
        self.values += fragment

    def finalize(self):
        return self.values, self.values[None, :]


# the implementation is only for semivalues
class sampling_lift(estimatorTemplate):
    def __init__(self, **kwargs):
        super(sampling_lift, self).__init__(**kwargs)
        self.interval_track = self.nue_track_avg // 2
        self.num_sample = self.nue_avg // 2
        self.batch_size = -(-self.nue_per_proc // (2 * self.num_player))
        self.nue_per_proc_run = self.batch_size * 2 * self.num_player

        self.results_aggregate = dict(estimates=np.zeros(self.num_player, dtype=np.float64), count=0)
        self.buffer = np.empty((self.buffer_size, self.num_player), dtype=np.float64)
        self.samples = np.empty((self.batch_size, self.num_player - 1), dtype=bool)

    def _init_indiv(self):
        assert self.nue_track_avg % 2 == 0
        assert self.nue_avg % 2 == 0

    def _generator(self):
        if self.semivalue == "weighted_banzhaf":
            t = self.semivalue_param
        elif self.semivalue == "shapley":
            t = np.random.rand()
        elif self.semivalue == "beta_shapley":
            t = np.random.beta(self.semivalue_param[1], self.semivalue_param[0])
        else:
            raise NotImplementedError
        return np.random.binomial(1, t, size=self.num_player - 1).astype(bool)

    def run(self, samples):
        game = self.game_func(**self.game_args)
        results_collect = np.zeros((len(samples), self.num_player), dtype=np.float64)
        subset = np.zeros(self.num_player, dtype=bool)
        for i, sample in enumerate(samples):
            results = results_collect[i]
            subset[:self.num_player-1] = sample
            results[-1] -= game.evaluate(subset)
            subset[-1] = 1
            results[-1] += game.evaluate(subset)
            for player in range(self.num_player - 1):
                subset[-1], subset[player] = subset[player], subset[-1]
                results[player] += game.evaluate(subset)
                subset[player] = 0
                results[player] -= game.evaluate(subset)
                subset[player] = 1
                subset[-1], subset[player] = subset[player], subset[-1]
            subset[-1] = 0
        return results_collect

    def _process(self, inputs):
        num_pre = self.results_aggregate["count"]
        num_cur = len(inputs) + num_pre
        self.results_aggregate["estimates"] *= num_pre / num_cur
        self.results_aggregate["estimates"] += inputs.sum(axis=0) / num_cur
        self.results_aggregate["count"] = num_cur

    def _estimate(self):
        return self.results_aggregate["estimates"]


class sampling_lift_paired(sampling_lift):
    def __init__(self, **kwargs):
        super(sampling_lift_paired, self).__init__(**kwargs)
        self.lock_switch = False

    def _init_indiv(self):
        assert self.nue_track_avg % 2 == 0
        assert self.nue_avg % 2 == 0
        if self.semivalue == "weighted_banzhaf":
            assert self.semivalue_param == 0.5
        if self.semivalue == "beta_shapley":
            assert self.semivalue_param[0] == self.semivalue_param[1]


class WSL(sampling_lift):
    def __init__(self, **kwargs):
        super(WSL, self).__init__(**kwargs)
        self.weights = self.distribution_cardinality() * self.num_player

    def _init_indiv(self):
        assert self.nue_track_avg % 2 == 0
        assert self.nue_avg % 2 == 0
        assert self.semivalue != "shapley"  # for the Shapley, sampling_lift = WSL

    def _generator(self):
        t = np.random.rand()
        return np.random.binomial(1, t, size=self.num_player - 1).astype(bool)

    def run(self, samples):
        results_collect = super(WSL, self).run(samples)
        scalars = self.weights[samples.sum(axis=1)]
        return scalars[:, None] * results_collect


class WSL_paired(WSL):
    def __init__(self, **kwargs):
        super(WSL_paired, self).__init__(**kwargs)
        self.lock_switch = False


class permutation(sampling_lift):
    # the evaluation of U(0) is not counted for the total budget of utility evaluations.
    def __init__(self, **kwargs):
        super(sampling_lift, self).__init__(**kwargs)
        self.num_sample = self.nue_avg
        self.interval_track = self.nue_track_avg
        self.batch_size = -(-self.nue_per_proc // self.num_player)
        self.nue_per_proc_run = self.batch_size * self.num_player

        self.results_aggregate = dict(estimates=np.zeros(self.num_player, dtype=np.float64), count=0)
        self.buffer = np.empty((self.buffer_size, self.num_player), dtype=np.float64)
        self.samples = np.empty((self.batch_size, self.num_player), dtype=np.int64)

    def _init_indiv(self):
        assert self.semivalue == "shapley"

    def _generator(self):
        return np.random.permutation(self.num_player)

    def run(self, samples):
        game = self.game_func(**self.game_args)
        results_collect = np.zeros((len(samples), self.num_player), dtype=np.float64)
        subset = np.zeros(self.num_player, dtype=bool)
        empty_value = game.evaluate(subset)
        for i, sample in enumerate(samples):
            results = results_collect[i]
            pre_value = empty_value
            for j in range(self.num_player):
                player = sample[j]
                results[player] -= pre_value
                subset[player] = True
                cur_value = game.evaluate(subset)
                results[player] += cur_value
                pre_value = cur_value
            subset.fill(False)
        return results_collect


class MSR(estimatorTemplate):
    def __init__(self, **kwargs):
        super(MSR, self).__init__(**kwargs)
        self.num_sample = self.nue_avg * self.num_player
        self.interval_track = self.nue_track_avg * self.num_player
        self.batch_size = self.nue_per_proc
        self.nue_per_proc_run = self.batch_size

        self.results_aggregate = np.zeros((4, self.num_player), dtype=np.float64)
        self.buffer = np.empty((self.buffer_size, self.num_player + 1), dtype=np.float64)
        self.samples = np.empty((self.batch_size, self.num_player), dtype=bool)

    def _init_indiv(self):
        assert self.semivalue == "weighted_banzhaf"
        assert 0 < self.semivalue_param and self.semivalue_param < 1

    def _generator(self):
        return np.random.binomial(1, self.semivalue_param, size=self.num_player).astype(bool)

    def run(self, samples):
        game = self.game_func(**self.game_args)
        results_collect = np.empty((len(samples), self.num_player + 1), dtype=np.float64)
        results_collect[:, :self.num_player] = samples
        for i, sample in enumerate(samples):
            results_collect[i, -1] = game.evaluate(sample)
        return results_collect

    def _process(self, inputs):
        subsets = inputs[:, :self.num_player]
        ues = inputs[:, [-1]]
        self.results_aggregate[0] += (ues * subsets).sum(axis=0)
        self.results_aggregate[1] += subsets.sum(axis=0)
        subsets = 1 - subsets
        self.results_aggregate[2] += (ues * subsets).sum(axis=0)
        self.results_aggregate[3] += subsets.sum(axis=0)

    def _estimate(self):
        counts = self.results_aggregate[1].copy()
        counts[counts == 0] = -1
        left = np.divide(self.results_aggregate[0], counts)
        counts = self.results_aggregate[3].copy()
        counts[counts == 0] = -1
        right = np.divide(self.results_aggregate[2], counts)
        return left - right


class MSR_paired(MSR):
    def __init__(self, **kwargs):
        super(MSR_paired, self).__init__(**kwargs)
        self.lock_switch = False

    def _init_indiv(self):
        assert self.semivalue == "weighted_banzhaf"
        assert self.semivalue_param == 0.5


class kernelSHAP(MSR):
    @staticmethod
    def calculate_constants(game_func, game_args, num_player):
        game = game_func(**game_args)
        subset = np.zeros(num_player, dtype=bool)
        v_empty = game.evaluate(subset)
        subset.fill(True)
        v_full = game.evaluate(subset)
        return v_empty, v_full

    def __init__(self, **kwargs):
        super(MSR, self).__init__(**kwargs)
        self.num_sample = self.nue_avg * self.num_player
        self.interval_track = self.nue_track_avg * self.num_player
        self.batch_size = self.nue_per_proc
        self.nue_per_proc_run = self.batch_size

        self.results_aggregate = dict(mat_A=np.zeros((self.num_player, self.num_player), dtype=np.float64),
                                      vec_b=np.zeros(self.num_player, dtype=np.float64),
                                      count=0)
        self.buffer = np.empty((self.buffer_size, self.num_player + 1), dtype=np.float64)
        self.samples = np.empty((self.batch_size, self.num_player), dtype=bool)

        with mp.Pool(1) as pool:
            self.constants = pool.apply(self.calculate_constants, (self.game_func, self.game_args, self.num_player))

    def _init_indiv(self):
        assert self.semivalue == "shapley"

        tmp = np.arange(1, self.num_player, dtype=np.float64)
        weights = 1 / np.multiply(tmp, tmp[::-1])
        self.weights = weights / weights.sum()
        self.s_range = np.arange(1, self.num_player)
        self.pos_range = np.arange(self.num_player)

    def _generator(self):
        s = np.random.choice(self.s_range, p=self.weights)
        pos = np.random.choice(self.pos_range, size=s, replace=False)
        subset = np.zeros(self.num_player, dtype=bool)
        subset[pos] = True
        return subset

    def _process(self, inputs):
        subsets = inputs[:, :self.num_player]
        ues = inputs[:, [-1]]
        A_tmp = subsets.T @ subsets
        b_tmp = subsets * (ues - self.constants[0])

        num_pre = self.results_aggregate["count"]
        num_cur = len(b_tmp) + num_pre
        self.results_aggregate["mat_A"] *= num_pre / num_cur
        self.results_aggregate["mat_A"] += A_tmp / num_cur
        self.results_aggregate["vec_b"] *= num_pre / num_cur
        self.results_aggregate["vec_b"] += b_tmp.sum(axis=0) / num_cur
        self.results_aggregate["count"] = num_cur

    def _estimate(self):
        A_inv = np.linalg.pinv(self.results_aggregate["mat_A"])
        vec_b = self.results_aggregate["vec_b"]
        vec_1 = np.ones(len(vec_b))
        v_empty, v_full = self.constants
        tmp = vec_b - (np.dot(vec_1, np.dot(A_inv, vec_b)) - v_full + v_empty) / np.dot(vec_1, np.dot(A_inv, vec_1))
        return np.dot(A_inv, tmp)


class kernelSHAP_paired(kernelSHAP):
    def __init__(self, **kwargs):
        super(kernelSHAP_paired, self).__init__(**kwargs)
        self.lock_switch = False


class unbiased_kernelSHAP(kernelSHAP):
    def __init__(self, **kwargs):
        super(MSR, self).__init__(**kwargs)
        self.num_sample = self.nue_avg * self.num_player
        self.interval_track = self.nue_track_avg * self.num_player
        self.batch_size = self.nue_per_proc
        self.nue_per_proc_run = self.batch_size

        self.results_aggregate = dict(estimates=np.zeros(self.num_player, dtype=np.float64), count=0)
        self.buffer = np.empty((self.buffer_size, self.num_player + 1), dtype=np.float64)
        self.samples = np.empty((self.batch_size, self.num_player), dtype=bool)

        with mp.Pool(1) as pool:
            self.constants = pool.apply(self.calculate_constants, (self.game_func, self.game_args, self.num_player))
        self.scalar = 2 * np.reciprocal(np.arange(1, self.num_player, dtype=np.float64)).sum()

    def _process(self, inputs):
        subsets = inputs[:, :self.num_player]
        ues = inputs[:, [-1]]
        tmp = (subsets - subsets.sum(axis=1, keepdims=True) / self.num_player) * (ues - self.constants[0])

        num_pre = self.results_aggregate["count"]
        num_cur = len(tmp) + num_pre
        self.results_aggregate["estimates"] *= num_pre / num_cur
        self.results_aggregate["estimates"] += tmp.sum(axis=0) / num_cur
        self.results_aggregate["count"] = num_cur

    def _estimate(self):
        return (self.constants[1] - self.constants[0]) / self.num_player + \
               self.results_aggregate["estimates"] * self.scalar


class unbiased_kernelSHAP_paired(unbiased_kernelSHAP, kernelSHAP_paired):
    def __init__(self, **kwargs):
        super(unbiased_kernelSHAP_paired, self).__init__(**kwargs)
        self.lock_switch = False


class ARM(MSR):
    def __init__(self, **kwargs):
        super(MSR, self).__init__(**kwargs)
        self.num_sample = (self.nue_avg * self.num_player) // 2
        self.interval_track = (self.nue_track_avg * self.num_player) // 2
        self.batch_size = -(-self.nue_per_proc // 2)
        self.nue_per_proc_run = self.batch_size * 2

        self.results_aggregate = np.zeros((4, self.num_player), dtype=np.float64)
        self.buffer = np.empty((self.buffer_size, 2, self.num_player + 1), dtype=np.float64)
        self.samples = np.empty((self.batch_size, 2, self.num_player), dtype=bool)

    def _init_indiv(self):
        assert (self.nue_avg * self.num_player) % 2 == 0
        assert (self.nue_track_avg * self.num_player) % 2 == 0

        weight = self.distribution_cardinality()
        weight_left = np.divide(weight, np.arange(1, self.num_player + 1))
        self.weight_left = weight_left / weight_left.sum()
        weight_right = np.divide(weight, np.arange(self.num_player, 0, -1))
        self.weight_right = weight_right / weight_right.sum()

        self.s_range_left = np.arange(1, self.num_player + 1)
        self.s_range_right = np.arange(self.num_player)
        self.pos_range = np.arange(self.num_player)

    def _generator(self):
        subset = np.zeros((2, self.num_player), dtype=bool)
        s = np.random.choice(self.s_range_left, p=self.weight_left)
        pos_left = np.random.choice(self.pos_range, size=s, replace=False)
        s = np.random.choice(self.s_range_right, p=self.weight_right)
        pos_right = np.random.choice(self.pos_range, size=s, replace=False)
        subset[0, pos_left] = True
        subset[1, pos_right] = True
        return subset

    def run(self, samples):
        game = self.game_func(**self.game_args)
        results_collect = np.empty((len(samples), 2, self.num_player + 1), dtype=np.float64)
        results_collect[:, :, :self.num_player] = samples
        for i, sample in enumerate(samples):
            results_collect[i, 0, -1] = game.evaluate(sample[0])
            results_collect[i, 1, -1] = game.evaluate(sample[1])
        return results_collect

    def _process(self, inputs):
        subsets = inputs[:, 0, :self.num_player]
        ues = inputs[:, 0, [-1]]
        self.results_aggregate[0] += (ues * subsets).sum(axis=0)
        self.results_aggregate[1] += subsets.sum(axis=0)
        subsets = 1 - inputs[:, 1, :self.num_player]
        ues = inputs[:, 1, [-1]]
        self.results_aggregate[2] += (ues * subsets).sum(axis=0)
        self.results_aggregate[3] += subsets.sum(axis=0)


# the implementation follows Algorithm 2 in "Efficient Sampling Approaches to Shapley Value Approximation"
class complement(estimatorTemplate):
    def __init__(self, **kwargs):
        super(complement, self).__init__(**kwargs)
        self.num_sample = (self.nue_avg * self.num_player) // 2
        self.interval_track = (self.nue_track_avg * self.num_player) // 2
        self.batch_size = -(-self.nue_per_proc // 2)
        self.nue_per_proc_run = self.batch_size * 2

        self.results_aggregate = np.zeros((2, self.num_player, self.num_player), dtype=np.float64)
        self.buffer = np.empty((self.buffer_size, self.num_player + 1), dtype=np.float64)
        self.samples = np.empty((self.batch_size, self.num_player), dtype=bool)

    def _init_indiv(self):
        assert self.semivalue == "shapley"
        assert (self.nue_avg * self.num_player) % 2 == 0
        assert (self.nue_track_avg * self.num_player) % 2 == 0

        self.s_range = np.arange(1, self.num_player + 1)

    def _generator(self):
        subset = np.zeros(self.num_player, dtype=bool)
        s = np.random.choice(self.s_range)
        pi = np.random.permutation(self.num_player)
        subset[pi[:s]] = True
        # Note what in the above is equal to
        # pos = np.random.choice(np.arange(self.num_player), size=s, replace=False)
        # subset[pos] = True
        # But we stay loyal to the original paper
        return subset

    def run(self, samples):
        game = self.game_func(**self.game_args)
        results_collect = np.zeros((len(samples), self.num_player + 1), dtype=np.float64)
        results_collect[:, :self.num_player] = samples
        for i, sample in enumerate(samples):
            results_collect[i, -1] += game.evaluate(sample)
            results_collect[i, -1] -= game.evaluate(~sample)
        return results_collect

    def _process(self, inputs):
        for take in inputs:
            subset = take[:self.num_player].astype(bool)
            subset_c = ~subset
            v = take[-1]
            subset_size = subset.sum()
            self.results_aggregate[0, subset, subset_size - 1] += v
            self.results_aggregate[0, subset_c, self.num_player - subset_size - 1] -= v
            self.results_aggregate[1, subset, subset_size - 1] += 1
            self.results_aggregate[1, subset_c, self.num_player - subset_size - 1] += 1

    def _estimate(self):
        # what in the below seems to fail occasionally, it returns nan for some entry while it should be a real number.
        # tmp = np.divide(self.results_aggregate[0], self.results_aggregate[1], where=self.results_aggregate[1] != 0)
        # return tmp.mean(axis=1)
        counts = self.results_aggregate[1].copy()
        counts[counts == 0] = -1
        return np.mean(np.divide(self.results_aggregate[0], counts), axis=1)


# the implementation follows Proposition 3.2 in "Measuring the Effect of Training Data on Deep Learning Predictions via
# Randomized Experiments"
class AME(estimatorTemplate):
    def __init__(self, **kwargs):
        super(AME, self).__init__(**kwargs)
        self.num_sample = self.nue_avg * self.num_player
        self.interval_track = self.nue_track_avg * self.num_player
        self.batch_size = self.nue_per_proc
        self.nue_per_proc_run = self.batch_size

        self.results_aggregate = dict(mat_A=np.zeros((self.num_player, self.num_player), dtype=np.float64),
                                      vec_b=np.zeros(self.num_player, dtype=np.float64))
        self.buffer = np.empty((self.buffer_size, self.num_player + 2), dtype=np.float64)
        self.samples = np.empty((self.batch_size, self.num_player + 1), dtype=np.float64)

    def _init_indiv(self):
        if self.semivalue == "weighted_banzhaf":
            assert 0 < self.semivalue_param and self.semivalue_param < 1
            self.variance = 1 / self.semivalue_param / (1 - self.semivalue_param)
        elif self.semivalue == "beta_shapley":
            assert 1 < self.semivalue_param[0] and 1 < self.semivalue_param[1]
            alpha, beta = self.semivalue_param
            ab = alpha + beta
            self.variance = (ab - 1) * (ab - 2) / (alpha - 1) / (beta - 1)
        else:
            raise NotImplementedError

    def _generator(self):
        sample = np.empty(self.num_player + 1, dtype=np.float64)
        if self.semivalue == "weighted_banzhaf":
            prob = self.semivalue_param
        elif self.semivalue == "beta_shapley":
            prob = np.random.beta(self.semivalue_param[1], self.semivalue_param[0])
        else:
            raise NotImplementedError
        sample[:-1] = np.random.binomial(1, prob, size=self.num_player)
        sample[-1] = prob
        return sample

    def run(self, samples):
        game = self.game_func(**self.game_args)
        results_collect = np.zeros((len(samples), self.num_player + 2), dtype=np.float64)
        results_collect[:, :-1] = samples
        for i, sample in enumerate(samples):
            subset = sample[:self.num_player].astype(bool)
            results_collect[i, -1] = game.evaluate(subset)
        return results_collect

    def _process(self, inputs):
        subsets = inputs[:, :self.num_player]
        ps = inputs[:, [-2]]
        ues = inputs[:, [-1]]
        tmp = subsets * (1 / ps) - (1 - subsets) * (1 / (1 - ps))
        self.results_aggregate["mat_A"] += tmp.T @ tmp
        self.results_aggregate["vec_b"] += (ues * tmp).sum(axis=0)

    def _estimate(self):
        return self.variance * (np.linalg.pinv(self.results_aggregate["mat_A"]) @ self.results_aggregate["vec_b"])


class AME_paired(AME):
    def __init__(self, **kwargs):
        super(AME_paired, self).__init__(**kwargs)
        self.lock_switch = False

    def _init_indiv(self):
        super(AME_paired, self)._init_indiv()
        if self.semivalue == "weighted_banzhaf":
            assert self.semivalue_param == 0.5
        if self.semivalue == "beta_shapley":
            assert self.semivalue_param[0] == self.semivalue_param[1]


# the implementation follows Appendix C.3 in "Data Banzhaf: A Robust Data Valuation Framework for Machine Learning"
class group_testing(sampling_lift):
    def __init__(self, **kwargs):
        super(sampling_lift, self).__init__(**kwargs)
        self.num_sample = self.nue_avg * self.num_player
        self.interval_track = self.nue_track_avg * self.num_player
        self.batch_size = self.nue_per_proc
        self.nue_per_proc_run = self.batch_size

        self.results_aggregate = dict(estimates=np.zeros(self.num_player, dtype=np.float64), count=0)
        self.buffer = np.empty((self.buffer_size, self.num_player), dtype=np.float64)
        self.samples = np.empty((self.batch_size, self.num_player + 1), dtype=bool)

    def _init_indiv(self):
        assert self.semivalue == "shapley"

        tmp = 1 / np.arange(1, self.num_player + 1, dtype=np.float64)
        weights = tmp + tmp[::-1]
        self.const = weights.sum()
        self.weights = weights / self.const
        self.s_range = np.arange(1, self.num_player + 1)
        self.pos_range = np.arange(self.num_player + 1)

    def _generator(self):
        subset = np.zeros(self.num_player + 1, dtype=bool)
        s = np.random.choice(self.s_range, p=self.weights)
        pos = np.random.choice(self.pos_range, size=s, replace=False)
        subset[pos] = True
        return subset

    def run(self, samples):
        game = self.game_func(**self.game_args)
        results_collect = np.zeros((len(samples), self.num_player), dtype=np.float64)
        for i, sample in enumerate(samples):
            tmp = sample * game.evaluate(sample)
            results_collect[i] = tmp[:self.num_player] - tmp[-1]
        return results_collect * self.const


class group_testing_paired(group_testing):
    def __init__(self, **kwargs):
        super(group_testing_paired, self).__init__(**kwargs)
        self.lock_switch = False


class GELS_ranking(kernelSHAP):
    def __init__(self, **kwargs):
        super(MSR, self).__init__(**kwargs)
        self.num_sample = self.nue_avg * self.num_player
        self.interval_track = self.nue_track_avg * self.num_player
        self.batch_size = self.nue_per_proc
        self.nue_per_proc_run = self.batch_size

        self.results_aggregate = np.zeros((2, self.num_player), dtype=np.float64)
        self.buffer = np.empty((self.buffer_size, self.num_player + 1), dtype=np.float64)
        self.samples = np.empty((self.batch_size, self.num_player), dtype=bool)

    def _init_indiv(self):
        self.num_player -= 1
        weights = self.distribution_cardinality()
        self.num_player += 1
        tmp = np.arange(1, self.num_player, dtype=np.float64)
        tmp = np.multiply(tmp / self.num_player, (self.num_player - tmp) / (self.num_player - 1))
        tmp = np.reciprocal(tmp)
        weights = np.multiply(weights, tmp)
        self.weights = weights / weights.sum()
        self.s_range = np.arange(1, self.num_player)
        self.pos_range = np.arange(self.num_player)

    def _generator(self):
        return super(GELS_ranking, self)._generator()

    def _process(self, inputs):
        subsets = inputs[:, :self.num_player]
        ues = inputs[:, [-1]]
        self.results_aggregate[0] += (ues * subsets).sum(axis=0)
        self.results_aggregate[1] += subsets.sum(axis=0)

    def _estimate(self):
        counts = self.results_aggregate[1].copy()
        counts[counts == 0] = -1
        return np.divide(self.results_aggregate[0], counts)


class GELS_ranking_paired(GELS_ranking):
    def __init__(self, **kwargs):
        super(GELS_ranking_paired, self).__init__(**kwargs)
        self.lock_switch = False

    def _init_indiv(self):
        super(GELS_ranking_paired, self)._init_indiv()
        if self.semivalue == "weighted_banzhaf":
            assert self.semivalue_param == 0.5
        if self.semivalue == "beta_shapley":
            assert self.semivalue_param[0] == self.semivalue_param[1]


class GELS(GELS_ranking):
    def __init__(self, **kwargs):
        super(MSR, self).__init__(**kwargs)
        self.num_sample = self.nue_avg * self.num_player
        self.interval_track = self.nue_track_avg * self.num_player
        self.batch_size = self.nue_per_proc
        self.nue_per_proc_run = self.batch_size

        weights = self.distribution_cardinality()
        self.scalar = (np.divide(weights, np.arange(self.num_player, 0, -1)) * self.num_player).sum()
        self.num_player += 1

        self.results_aggregate = np.zeros((2, self.num_player), dtype=np.float64)
        self.buffer = np.empty((self.buffer_size, self.num_player + 1), dtype=np.float64)
        self.samples = np.zeros((self.batch_size, self.num_player), dtype=bool)

    def _estimate(self):
        estimates = super(GELS, self)._estimate() * self.scalar
        return estimates[:-1] - estimates[-1]


class GELS_paired(GELS, GELS_ranking_paired):
    # For the Shapley value, this estimator is equal to group_testing_paired
    def __init__(self, **kwargs):
        super(GELS_paired, self).__init__(**kwargs)
        self.lock_switch = False

    def _init_indiv(self):
        GELS_ranking_paired._init_indiv(self)


class GELS_shapley(GELS_ranking):
    def __init__(self, **kwargs):
        super(GELS_shapley, self).__init__(**kwargs)
        with mp.Pool(1) as pool:
            self.constants = pool.apply(self.calculate_constants, (self.game_func, self.game_args, self.num_player))

    def _init_indiv(self):
        assert self.semivalue == "shapley"

        tmp = 1 / np.arange(1, self.num_player, dtype=np.float64)
        weights = np.multiply(tmp, tmp[::-1])
        self.weights = weights / weights.sum()
        self.scalar = tmp.sum()
        self.s_range = np.arange(1, self.num_player)
        self.pos_range = np.arange(self.num_player)

    def _estimate(self):
        estimates = super(GELS_shapley, self)._estimate() * self.scalar
        offset = (self.constants[1] - self.constants[0] - estimates.sum()) / self.num_player
        return estimates + offset


class GELS_shapley_paired(GELS_shapley):
    # This estimator is equal to unbiased_kernelSHAP_paired
    def __init__(self, **kwargs):
        super(GELS_shapley_paired, self).__init__(**kwargs)
        self.lock_switch = False


class simSHAP(kernelSHAP):
    def __init__(self, **kwargs):
        super(MSR, self).__init__(**kwargs)
        self.num_sample = self.nue_avg * self.num_player
        self.interval_track = self.nue_track_avg * self.num_player
        self.batch_size = self.nue_per_proc
        self.nue_per_proc_run = self.batch_size

        self.results_aggregate = dict(estimates=np.zeros(self.num_player, dtype=np.float64), count=0)
        self.buffer = np.empty((self.buffer_size, self.num_player + 1), dtype=np.float64)
        self.samples = np.empty((self.batch_size, self.num_player), dtype=bool)

        with mp.Pool(1) as pool:
            self.constants = pool.apply(self.calculate_constants, (self.game_func, self.game_args, self.num_player))

    def _init_indiv(self):
        assert self.semivalue == "shapley"

        tmp = np.arange(1, self.num_player, dtype=np.float64)
        weights = 1 / np.multiply(tmp, tmp[::-1])
        self.gamma = weights.sum()
        self.weights = weights / self.gamma
        self.s_range = np.arange(1, self.num_player)
        self.pos_range = np.arange(self.num_player)

    def _process(self, inputs):
        subsets = inputs[:, :self.num_player]
        ues = inputs[:, [-1]]
        sizes = subsets.sum(axis=1, keepdims=True)

        tmp = ((self.num_player - sizes) * subsets - sizes * (1 - subsets)) * ues
        num_pre = self.results_aggregate["count"]
        num_cur = num_pre + ues.shape[0]
        self.results_aggregate["estimates"] *= num_pre / num_cur
        self.results_aggregate["estimates"] += tmp.sum(axis=0) / num_cur
        self.results_aggregate["count"] = num_cur

    def _estimate(self):
        return self.results_aggregate["estimates"] * self.gamma \
               + (self.constants[1] - self.constants[0]) / self.num_player


class simSHAP_paired(simSHAP):
    # it is equal to GELS_shapley_paired
    def __init__(self, **kwargs):
        super(simSHAP_paired, self).__init__(**kwargs)
        self.lock_switch = False
