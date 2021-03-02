# Compositional Lifelong Learning

This is the source code used for [Lifelong Learning of Compositional Structures (Mendez and Eaton, 2021)](https://openreview.net/forum?id=ADWd4TJO13G). 

This package contains the implementations of nine algorithms that conform to our framework for compositional learning: ER, EWC, and VAN with factorized linear models, soft layer ordering, and soft gating. Deep learning variants are implemented with and without dynamic addition of new modules. Implementations for all baselines in the paper are also included: jointly trained, no-components, and frozen components.

## Installation

All dependencies are listed in the `env.yml` file (Linux). To install, create a Conda environment with:
 
```$ conda env create -f env.yml```

Activate the environment with:

```$ conda activate 2020-compositional```

You should be able to run any of the experiment scripts at that point.

## Code structure

The code structure is the following:

* `experiment_script.py` -- Script for running multiple experiments with soft layer ordering in parallel
* `experiment_script_gated.py` -- Script for running multiple experiments with soft gating in parallel
* `experiment_script_linear.py` -- Script for running multiple experiments with linear factored model in parallel
* `experiment_script_limiteddata.py` -- Script for running multiple experiments in parallel varying the number of data points
* `lifelong_experiment.py` -- Experiment script for running training. Takes as argument the data set, number of tasks, number of random seeds, and all hyper-parameters
* `lifelong_experiment_linear.py` -- Same as `lifelong_experiment.py` but ignores the `batch_size` argument and uses `dataset.max_batch_size` instead
* `lifelong_experiment_pixelmnist.py` -- Experiment script for training the pixel visualization MNIST experiments
* `make_lifelong_table.py` -- Make the tables for this README
* `datasets/`
    * `DATASET_NAME` -- each directory contains the raw data, the processing code we used (where applicable), and the processed data. 
    * `datasets.py` -- a Python class wrapper for each data set, which creates a `torch.utils.data.TensorDataset` for each task in the data set.
* `learners/`
    * `base_learning_classes.py` -- Base classes for compositional, dynamic + compositional, joint, and no-components agents.
    * `*_compositional.py` -- Compositional agents
    * `*_dynamic.py` -- Dynamic + compositional agents
    * `*_joint.py` -- Jointly trained baselines
    * `*_nocomponents.py` -- No-components baselines
    * `er_*.py` -- ER-based agents
    * `ewc_*.py` -- EWC-based agents
    * `van_*.py` -- VAN-based agents
    * `fm_*.py` -- Frozen components agents
* `models/`
    * `base_net_classes.py` -- Base classes for deep compositional models
    * `linear.py` -- No-components linear model
    * `linear_factored.py` - Factored linear model
    * `cnn*.py` -- Convolutional nets
    * `mlp*.py` -- Fully-connected nets
    * no suffix -- No-components baseline models
    * `*_soft_lifelong.py` -- Soft layer ordering
    * `*_soft_lifelong_dynamic.py` -- Soft layer ordering supporting dynamic component additions
    * `*_soft_gated_lifelong.py` -- Soft gating net
    * `*_soft_gated_lifelong_dynamic` -- Soft gating net supporting dynamic component additions
    * `mlp_soft_lifelong_pixelmnist.py` -- MLP with soft ordering for pixel visualization experiments on MNIST
* `utils/`
    * `kfac_ewc.py` -- Kronecker-factored Hessian approximator for EWC, implemented as a PyTorch optimizer/preconditioner
    * `make_*.py` -- Code for creating the different figures throughout the paper
    * `replay_buffers.py` -- Implementation of the replay buffers for ER-based algorithms as a `torch.utils.data.TensorDataset`

## Reproducing results
Tables 1 and 2 contain the main results in the paper. 

To reproduce the soft layer ordering results, execute:

```
$ python experiment_script.py
$ python make_lifelong_table.py -T 10 10 20 20 50 -d MNIST Fashion CUB CIFAR Omniglot -e 100 -alg er_compositional er_joint er_nocomponents er_dynamic ewc_compositional ewc_joint ewc_nocomponents ewc_dynamic van_compositional van_nocomponents van_joint van_dynamic fm_compositional fm_dynamic -n 10 -r results/
```

And to reproduce the soft gating results, execute:

```
$ python experiment_script_gated.py
$ python make_lifelong_table.py -T 10 10 20 50 -d MNIST Fashion CIFAR Omniglot -e 100 -alg er_compositional er_joint er_dynamic ewc_compositional ewc_joint ewc_dynamic van_compositional van_joint van_dynamic fm_compositional fm_dynamic -n 10 -r results/gated/
```

Table 1: Accuracy of all algorithms using soft layer ordering.
| Base   | Algorithm     | MNIST         | Fashion       | CUB           | CIFAR         | Omniglot      |
|:-------|:--------------|:--------------|:--------------|:--------------|:--------------|:--------------|
| ER     | Dyn. + Comp.  | **97.6±0.2**% | **96.6±0.4**% | 79.0±0.5%     | **77.6±0.3**% | **71.7±0.5**% |
| ER     | Compositional | 96.5±0.2%     | 95.9±0.6%     | **80.6±0.3**% | 58.7±0.5%     | 71.2±1.0%     |
| ER     | Joint         | 94.2±0.3%     | 95.1±0.7%     | 77.7±0.5%     | 65.8±0.4%     | 70.7±0.3%     |
| ER     | No Comp.      | 91.2±0.3%     | 93.6±0.6%     | 44.0±0.9%     | 51.6±0.6%     | 43.2±4.2%     |
| EWC    | Dyn. + Comp.  | **97.2±0.2**% | **96.5±0.4**% | **73.9±1.0**% | **77.6±0.3**% | **71.5±0.5**% |
| EWC    | Compositional | 96.7±0.2%     | 95.9±0.6%     | 73.6±0.9%     | 48.0±1.7%     | 53.4±5.2%     |
| EWC    | Joint         | 66.4±1.4%     | 69.6±1.6%     | 65.4±0.9%     | 42.9±0.4%     | 58.6±1.1%     |
| EWC    | No Comp.      | 66.0±1.1%     | 68.8±1.1%     | 50.6±1.2%     | 36.0±0.7%     | 68.8±0.4%     |
| VAN    | Dyn. + Comp.  | **97.3±0.2**% | **96.4±0.4**% | 73.0±0.7%     | **73.0±0.4**% | **69.4±0.4**% |
| VAN    | Compositional | 96.5±0.2%     | 95.9±0.6%     | **74.5±0.7**% | 54.8±1.2%     | 68.9±0.9%     |
| VAN    | Joint         | 67.4±1.4%     | 69.2±1.9%     | 65.1±0.7%     | 43.9±0.6%     | 63.1±0.9%     |
| VAN    | No Comp.      | 64.4±1.1%     | 67.0±1.3%     | 49.1±1.6%     | 36.6±0.6%     | 68.9±1.0%     |
| FM     | Dyn. + Comp.  | 99.1±0.0%     | 97.3±0.3%     | 78.3±0.4%     | 78.4±0.3%     | 71.0±0.4%     |
| FM     | Compositional | 84.1±0.8%     | 86.3±1.3%     | 80.1±0.3%     | 48.8±1.6%     | 63.0±3.3%     |


Table 2: Accuracy of all algorithms using soft gating. No Comp. results are the same as in Table 1.
| Base   | Algorithm     | MNIST         | Fashion       | CIFAR         | Omniglot      |
|:-------|:--------------|:--------------|:--------------|:--------------|:--------------|
| ER     | Dyn. + Comp.  | **98.2±0.1**% | **97.1±0.4**% | 74.9±0.3%     | 73.7±0.3%     |
| ER     | Compositional | 98.0±0.2%     | 97.0±0.4%     | **75.9±0.4**% | **73.9±0.3**% |
| ER     | Joint         | 93.8±0.3%     | 94.6±0.7%     | 72.0±0.4%     | 72.6±0.2%     |
| ER     | No Comp.      | 91.2±0.3%     | 93.6±0.6%     | 51.6±0.6%     | 43.2±4.2%     |
| EWC    | Dyn. + Comp.  | **98.2±0.1**% | **97.0±0.4**% | 76.6±0.5%     | 73.6±0.4%     |
| EWC    | Compositional | 98.0±0.2%     | **97.0±0.4**% | **76.9±0.3**% | **74.6±0.2**% |
| EWC    | Joint         | 68.6±0.9%     | 69.5±1.8%     | 49.9±1.1%     | 63.5±1.2%     |
| EWC    | No Comp.      | 66.0±1.1%     | 68.8±1.1%     | 36.0±0.7%     | 68.8±0.4%     |
| VAN    | Dyn. + Comp.  | **98.2±0.1**% | **97.1±0.4**% | 66.6±0.7%     | 69.1±0.9%     |
| VAN    | Compositional | 98.0±0.2%     | 96.9±0.5%     | **68.2±0.5**% | **72.1±0.3**% |
| VAN    | Joint         | 67.3±1.7%     | 66.4±1.9%     | 51.0±0.8%     | 65.8±1.3%     |
| VAN    | No Comp.      | 64.4±1.1%     | 67.0±1.3%     | 36.6±0.6%     | 68.9±1.0%     |
| FM     | Dyn. + Comp.  | 98.4±0.1%     | 97.0±0.4%     | 77.2±0.3%     | 74.0±0.4%     |
| FM     | Compositional | 94.8±0.4%     | 96.3±0.4%     | 77.2±0.3%     | 74.1±0.3%     |



## Citing this work

If you use this work, please cite our paper

```
@inproceedings{
    mendez2021lifelong,
    title={Lifelong Learning of Compositional Structures},
    author={Jorge A Mendez and Eric Eaton},
    booktitle={International Conference on Learning Representations},
    year={2021},
    url={https://openreview.net/forum?id=ADWd4TJO13G}
}
```