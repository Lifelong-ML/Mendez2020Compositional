# Compositional Lifelong Learning

This is the source code used for [Lifelong Learning of Compositional Structures (Mendez and Eaton, 2020)](https://arxiv.org/abs/2007.07732). 

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
| ER     | Dyn. + Comp.  | **97.7±0.2**% | **96.3±0.4**% | 77.6±0.8%     | **75.9±0.5**% | **69.0±0.7**% |
| ER     | Compositional | 96.5±0.2%     | 95.3±0.7%     | **79.2±0.7**% | 56.0±0.8%     | 67.8±1.0%     |
| ER     | Joint         | 94.2±0.3%     | 94.7±0.7%     | 76.8±0.5%     | 63.8±0.6%     | 67.9±0.5%     |
| ER     | No Comp.      | 91.2±0.3%     | 93.1±0.6%     | 43.1±1.0%     | 49.5±0.8%     | 40.0±3.9%     |
| EWC    | Dyn. + Comp.  | **97.3±0.2**% | **96.1±0.4**% | **72.7±0.9**% | **71.7±0.9**% | **67.7±0.6**% |
| EWC    | Compositional | 96.7±0.2%     | 95.3±0.6%     | 72.4±1.2%     | 43.4±1.1%     | 52.2±7.3%     |
| EWC    | Joint         | 66.3±1.4%     | 69.1±1.4%     | 65.3±0.7%     | 41.3±0.8%     | 61.2±0.7%     |
| EWC    | No Comp.      | 64.3±0.8%     | 58.5±2.9%     | 47.7±1.4%     | 34.1±0.9%     | 66.2±1.2%     |
| VAN    | Dyn. + Comp.  | **97.4±0.3**% | **96.0±0.4**% | 72.5±0.8%     | **69.6±1.2**% | **67.1±0.6**% |
| VAN    | Compositional | 96.4±0.2%     | 95.3±0.6%     | **73.7±1.1**% | 52.5±1.3%     | 65.3±1.2%     |
| VAN    | Joint         | 67.4±1.4%     | 66.1±2.4%     | 64.4±0.8%     | 41.4±0.8%     | 60.2±1.1%     |
| VAN    | No Comp.      | 64.4±1.1%     | 59.4±2.7%     | 48.3±1.9%     | 34.1±0.8%     | 64.7±1.0%     |
| FM     | Dyn. + Comp.  | 99.1±0.0%     | 97.0±0.3%     | 78.2±0.4%     | 74.3±0.9%     | 67.7±0.7%     |
| FM     | Compositional | 84.1±0.8%     | 85.9±1.3%     | 79.2±0.6%     | 46.0±1.6%     | 58.3±3.0%     |

Table 2: Accuracy of all algorithms using soft gating. No Comp. results are the same as in Table 1.
| Base   | Algorithm     | MNIST         | Fashion       | CIFAR         | Omniglot      |
|:-------|:--------------|:--------------|:--------------|:--------------|:--------------|
| ER     | Dyn. + Comp.  | **98.2±0.1**% | **96.7±0.4**% | **72.2±0.7**% | **71.5±0.7**% |
| ER     | Compositional | 98.0±0.2%     | 96.3±0.4%     | 71.5±0.8%     | 70.8±0.6%     |
| ER     | Joint         | 93.7±0.4%     | 93.3±1.5%     | 68.9±1.0%     | 70.3±0.3%     |
| EWC    | Dyn. + Comp.  | **98.2±0.1**% | 96.5±0.4%     | 65.9±0.8%     | 67.3±1.3%     |
| EWC    | Compositional | 98.0±0.3%     | **96.6±0.4**% | **73.4±0.7**% | **70.4±0.7**% |
| EWC    | Joint         | 66.1±1.0%     | 65.3±1.7%     | 49.8±1.2%     | 62.4±0.5%     |
| VAN    | Dyn. + Comp.  | **98.2±0.1**% | **96.8±0.4**% | 64.4±0.8%     | 65.5±1.3%     |
| VAN    | Compositional | 98.0±0.2%     | 96.5±0.4%     | **65.9±0.8**% | **69.4±0.7**% |
| VAN    | Joint         | 67.3±1.7%     | 62.6±3.4%     | 49.2±0.8%     | 62.3±1.6%     |
| FM     | Dyn. + Comp.  | 98.5±0.1%     | 96.7±0.4%     | 72.9±0.7%     | 71.7±0.6%     |
| FM     | Compositional | 94.8±0.4%     | 95.8±0.4%     | 75.2±0.7%     | 72.0±0.5%     |

## Citing this work

If you use this work, please cite our paper

```
@article{mendez2020lifelong,
  title={Lifelong Learning of Compositional Structures},
  author={Mendez, Jorge A. and Eaton, Eric},
  journal={arXiv preprint arXiv:2007.07732},
  year={2020}
}
```