import os
import numpy as np
from utils.funcs import vd_DotMap

size_max = 1000
dataset = "FMNIST"
root = os.path.join("exp", "train_value_estimators")

DV = vd_DotMap(dict(
    dir_fig=os.path.join(root, "fig"),
    ########################## common
    n_process=100,
    dataset=dataset,
    metric="accuracy",
    lr_game=0.01,
    n_valued=10000,
    n_perf=500,
    n_test=200,
    root=root,
    size_max=size_max,
    weights=np.reciprocal(np.sqrt(np.arange(1, size_max + 1))),
    ########################### for generating the training dataset
    sample_per_proc=20,
    seeds_training=np.arange(50),
    num_sample_per_seed=200000,
    training_saved=os.path.join(root, dataset, "training_dataset", "seed={}.npz"),
    ########################## for generating the test dataset
    seeds_test=np.arange(30),
    num_marginal_per_seed=10000,
    test_saved=os.path.join(root, dataset, "test_dataset", "seed={}.npz"),
    ########################## for generating the validation dataset
    n_val=200,
    seeds_val=np.arange(30),
    val_saved=os.path.join(root, dataset, "val_dataset", "seed={}.npz"),
    ########################## for training DVEstimator
    lr_DVE=1e-3,
    batch_size=10000,
    epochs=30,
    interval_report=10,
    epoch_save=5,
    seeds_train=np.arange(30),
    dir_train=os.path.join(root, dataset, "trained", "seed={}"),
))

os.makedirs(DV.dir_fig, exist_ok=True)

