#!/bin/bash
#SBATCH --account="stability"
#SBATCH --comment="stablegpt"
#SBATCH --job-name="jpllm"
#SBATCH --partition=g40
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=11G
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --output=slurm_job_logs/neox_%j.out
#SBATCH --error=slurm_job_logs/neox_%j.err

module load openmpi cuda/11.7

###########################################################
# Pre-load
###########################################################
HOME=/fsx/home-$(whoami)
CUDNN_HOME=/fsx/quentin/cudnn-linux-x86_64-8.6.0.163_cuda11-archive
export LD_LIBRARY_PATH=$CUDNN_HOME/lib:$LD_LIBRARY_PATH
export CPATH=$CUDNN_HOME/include:$CPATH

CONDA_HOME=/admin/home-$(whoami)/micromamba/envs/stable-neox-env
export PATH=$CONDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CONDA_HOME/lib:$LD_LIBRARY_PATH
export CPATH=$CONDA_HOME/include:$CPATH
###########################################################

ds_report

###########################################################
# CUDA/Torch Setup
###########################################################
mkdir -p /fsx/home-$(whoami)/stablegpt/nccl-logs/$SLURM_JOB_ID  
export NCCL_DEBUG_FILE=$HOME/stablegpt/nccl-logs/$SLURM_JOB_ID/debug.%h.%p
export NCCL_DEBUG=info
export NCCL_DEBUG_SUBSYS=all
export NCCL_TREE_THRESHOLD=0
export NCCL_PROTO=simple
# Network issues without the following two NCCL vars set; See https://github.com/NVIDIA/nccl/issues/676
export NCCL_IBEXT_DISABLE=1
export NCCL_SOCKET_IFNAME=^docker0,lo

export CUDA_LAUNCH_BLOCKING=0
export TORCH_EXTENSIONS_DIR=$HOME/.cache/torch_extensions
# export TORCH_CPP_LOG_LEVEL=INFO
# export TORCH_DISTRIBUTED_DEBUG=INFO
export PYTHONFAULTHANDLER=1
###########################################################


###########################################################
# FI Setup (if required)
###########################################################
export FI_EFA_FORK_SAFE=1
export FI_LOG_LEVEL=warn
export FI_EFA_ENABLE_SHM_TRANSFER=0
export FI_PROVIDER=efa
export FI_EFA_TX_MIN_CREDITS=64
export FI_EFA_USE_DEVICE_RDMA=-1 # use for p4dn
###########################################################


###########################################################
# MPI Setup (if required)
###########################################################
export OMPI_MCA_mtl_base_verbose=1
export OMPI_MCA_btl="^openib"
###########################################################


###########################################################
# Network Setup
###########################################################
export HOSTNAMES=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=12802
export COUNT_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l)

echo "Master Addr: $MASTER_ADDR"
echo "Node Count: $COUNT_NODE"
echo "Host Names: $HOSTNAMES"
lsof -i
cat /etc/hosts

# Write the hostfile for this job
export MASTER_ADDR=$(echo $MASTER_ADDR | cut -d '-' -f 2- | tr '-' '.')
bash $HOME/write_hostfile.sh
export DLTS_HOSTFILE=$HOME/hostfiles/hosts_$SLURM_JOBID
###########################################################


sig_handler()
{
    echo "BATCH interrupted"
    wait # wait for all children, this is important!
}
trap 'sig_handler' SIGINT SIGTERM SIGCONT


###########################################################
# Environment Setup
# TODO: Replace with your own environment setup
###########################################################
source $HOME/.bashrc
eval "$(micromamba shell hook --shell=bash)"
micromamba activate stable-neox-env
###########################################################


###########################################################
# NeoX Setup
###########################################################
export WANDB_MODE="online"
export PYTORCH_CUDA_ALLOC_CONF='max_split_size_mb:256'

wandb login --host=https://stability.wandb.io --relogin local-XXX 

TRAIN_PATH=$HOME/repos/gpt-neox
cd $TRAIN_PATH

git config --global --add safe.directory $TRAIN_PATH
python ./deepy.py train.py --conf_dir configs /fsx/home-meng/repos/gpt-neox/configs/stable-lm/stable-lm-jp-1b-experimental.yml