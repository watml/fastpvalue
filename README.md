# Faster Approximation of Probabilistic and Distributional Values via Least Squares

This repository is to provide an implementation to replicate all the results reported in the paper *Faster Approximation of Probabilistic and Distributional Values via Least Squares* accepted to ICLR 2024: 

    @inproceedings{LiYu24,  
      title       = {Faster Approximation of Probabilistic and Distributional Values via Least Squares},
      author      = {W. Li and Y. Yu},
      booktitle   = {The Twelfth International Conference on Learning Representations {(ICLR)}},
      year        = {2024},
      url         = {https://openreview.net/forum?id=lvSMIsztka},
    }

## Comparing Estimators
The command `python compare_estimators.py -e` will calculate the exact estimates of the Shapley value, the Banzhaf value, Beta(2,2) and Beta(4,1) while the utility functions are set to report the accuracy and the cross-entropy loss on $D_{perf}$ for the datasets iris, wind, MNIST and FMNIST. Therefore, there are 32 combinations in total. 
Note that it takes a long time to complete.
Nevertheless, the precomputed exact estimates are provided in exp folder.

For runing estimators, use `python compare_estimators.py` instead. There are 14 estimators implemented for the Shapley value, 13 for the Banzhaf value, 11 for Beta(2,2) and 6 for Beta(4,1).
The user can check compare_estimators.py for all user-specified arguments.

After all results are generated, `python plot_estimators.py` will produce all the figures. 

## Training Estimators
All user-specified arguments are in utils/args_DV.py. 
The default dataset is set to be MNIST.
The following commands will generate the training dataset used for training estimators,
valiadation and test dataset for reporting the relative difference and Spearman correlation.
Note that n_process=100 in utils/args_DV.py means there will be 100 processes (each process is restricted to have only one thread) forked for generating the datasets,
and thus make sure that there are at least 100 cpus.

`
python generate_training_dataset.py
`

`
python generate_val_dataset.py
`

`
python generate_test_dataset.py
`

After all the datasets are generated, run `python trainDVEstimator.py` to train estimators.

Finally, run `python plot_training.py` and `python plot_appr.py` to plot the figures used in the paper.
