# stableLLM

## Description

This repository contains the code for training and evaluating the stableLLM model. It utilizes the neox library for the GPT-Neo model.

## Installation

To install this repository, first clone the repository and its submodules:

```bash
git clone --recursive https://github.com/Stability-AI/gpt-neox.git
```

Setup the environment:

For SLURM (optional):
```bash
srun --exclusive --account=stablegpt --nodes=1 --partition=g40 --gres=gpu:1 --pty bash -i
source /etc/profile.d/modules.sh
module load cuda/11.7
```

Virtual Environment:

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-py38_23.3.1-0-Linux-x86_64.sh
bash Miniconda3-py38_23.3.1-0-Linux-x86_64.sh
```
    
```bash
source ~/.bashrc
conda env create -f env.yaml
conda activate stable-neox-env
```

Next install the third party libraries:

```bash
bash setup.sh
```

## Test Installation

To test the installation, run the following command:

```bash
bash test.sh
```

This test runs the model on a small dataset and checks that the output is correct.

## Run Training

To run a complete training run, use the provided `deploy.sh` script that uses slurm to configure everything:

```bash
bash deploy.sh <run_name> <num_nodes> <config_file_name>
```

**Note:** For <config_file_name>, make sure you do not include the .yml extension.

If you want to test a single node training with sbatch, you can use the following command:

```bash
bash deploy.sh single_node_test 1 ./configs/test_config
```

To perform a multi-node training run, you can use the following command:

```bash
bash deploy.sh multi_node_test 2 ./configs/test_multinode_config
```

For multinode, ensure that your config file has the following settings:

```bash
# gpt-neox/configs/config.yml
"launcher": "openmpi",
"deepspeed_mpi": true,
```

or

```bash
# gpt-neox/configs/config.yml
"launcher": "slurm",
"deepspeed_slurm": true,
```

Depending on the launcher you want to use.
