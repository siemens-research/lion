### Paper

This repository is meant to publish the code used to generate the results in the paper

```Phillip Swazinna, Steffen Udluft, Thomas Runkler. "User-Interactive Offline Reinforcement Learning." International Conference on Learning Representations (ICLR), 2023.```

--> You can check out the [paper website](https://pswazinna.github.io/LION/) or read the full paper [here!](https://openreview.net/forum?id=a4COps0uokg)

### LION: Learning in Interactive Offline eNvironments

The main contribution in the paper is the development of the LION algorithm to train trade-off conditional policies that remain adaptive after training.

The project uses python 3.8.10 and the dependencies presented in requirements.txt

we go through the necessary steps to reproduce experiments from back to front:

- if you would like to visualize the results of a performed policy training, simply run `plot_policy.py`. We provide experimental results for LION and lambda-TD3+BC on the global-1.0 dataset so you can just run it without performing experiments first.

- to start a LION policy training, run `policy_training.py`. We provide pretrained transition models for the global-1.0 dataset, so you do not have to train models for a long time before performing experiments. However, the dataset itself is too large and it needs to be downloaded from https://github.com/siemens/industrialbenchmark/tree/offline_datasets/datasets and be placed in the `datasets/` folder.

- to switch to a lambda-TD3+BC training, simply change the used config dictionary to `model_free_config` in line 435

- similarly, if you want to perform experiments on different datasets or the 2D environments, switch to the `simpleenv_config`. You need to also alter the model config in line 434 to `default_model_config` and uncomment lines 447 & 448

- if you would like to train models yourself, you can run `recurrent_model_simple.py` (for the IB experiments) or `transition_model.py` (for the simple 2D env). The dataset for the 2D Env can be generated (see below), while the Industrial Benchmark Datasets need to be downloaded at https://github.com/siemens/industrialbenchmark/tree/offline_datasets/datasets and be placed in the `datasets/` folder.

- you can run a dataset analysis for the 2D Env datasets (see Fig. 2 in the paper) using `analyse_data.py`

- you can generate the dataset for the 2D Env yourself with `generate_data.py`

Additional:
With `visualize_beta.py` and `visualize_data_dist.py` you can visualize the beta distributions from which lambda was sampled, the performance in the corresponding experiments, as well as the distance to the original policy model. With `visualize_data_dist.py` you can also visualize the dependency on eta.


#### Copyright

The copyright remains with the codeowners under the MIT license (see `license.txt`)