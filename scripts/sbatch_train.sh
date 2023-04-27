#!/bin/bash
#SBATCH --account="stablegpt"
#SBATCH --job-name="neox"
#SBATCH --partition=g40
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=11G
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=logs/neox_%j.out
#SBATCH --error=logs/neox_%j.err

# NOTE: This script is tailored for launching multi-node gpt-neox jobs on AWS cluster.
# WARNING: This asusmes your work is stored in /fsx/home-<username>/

module load openmpi cuda/11.7


###########################################################
# Pre-load
###########################################################
# NOTE: We use Quentin`s cudnn install for now; you may want to use
# the system installs instead.
CUDNN_HOME=/fsx/quentin/cudnn-linux-x86_64-8.6.0.163_cuda11-archive
export LD_LIBRARY_PATH=$CUDNN_HOME/lib:$LD_LIBRARY_PATH
export CPATH=$CUDNN_HOME/include:$CPATH
# Force PyTorch to use system CUDA & NCCL install
# export PATH="/usr/local/cuda-11.7/bin:$PATH"
# export LD_LIBRARY_PATH="/usr/local/cuda-11.7/lib64:$LD_LIBRARY_PATH"
# export LD_PRELOAD="/opt/nccl/build/lib/libnccl.so"

# TODO: Set CONDA_HOME to point to your conda env
# E.g. /fsx/home-guac/miniconda3/envs/stable-neox-env
CONDA_HOME={TODO: FILL-IN}
export PATH=$CONDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CONDA_HOME/lib:$LD_LIBRARY_PATH
export CPATH=$CONDA_HOME/include:$CPATH

# OPTIONAL: EFA handling
# export LD_LIBRARY_PATH=/opt/aws-ofi-nccl/lib:/opt/amazon/efa/lib64:/usr/local/cuda-11.7/efa/lib:/usr/local/cuda-11.7/lib:/usr/local/cuda-11.7/lib64:/usr/local/cuda-11.7:/opt/nccl/build/lib:/opt/aws-ofi-nccl-install/lib:/opt/aws-ofi-nccl/lib:$LD_LIBRARY_PATH
# export PATH=/opt/amazon/efa/bin:/opt/amazon/openmpi/bin:/usr/local/cuda-11.7/bin:$PATH
###########################################################

ds_report

###########################################################
# CUDA/Torch Setup
###########################################################
mkdir -p /fsx/home-$(whoami)/stablegpt/nccl-logs/$SLURM_JOB_ID  
export NCCL_DEBUG_FILE=/fsx/home-$(whoami)/stablegpt/nccl-logs/$SLURM_JOB_ID/debug.%h.%p
export NCCL_DEBUG=info
export NCCL_DEBUG_SUBSYS=all
export NCCL_TREE_THRESHOLD=0
export NCCL_PROTO=simple
# Network issues without the following two NCCL vars set; See https://github.com/NVIDIA/nccl/issues/676
export NCCL_IBEXT_DISABLE=1
export NCCL_SOCKET_IFNAME=^docker0,lo
export CUDA_LAUNCH_BLOCKING=0
export TORCH_EXTENSIONS_DIR=/fsx/home-$(whoami)/.cache/torch_extensions
export PYTHONFAULTHANDLER=1
# OPTIONAL
# export TORCH_CPP_LOG_LEVEL=INFO
# export TORCH_DISTRIBUTED_DEBUG=INFO
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
# OPTIONAL
# export OMPI_MCA_pml="^cm"
# export OMPI_MCA_btl_tcp_if_exclude="lo,docker1"
# export OMPI_MCA_btl_base_verbose=30
# export OMPI_MCA_plm_rsh_no_tree_spawn=1
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
bash /fsx/home-$(whoami)/write_hostfile.sh
mkdir -p /fsx/home-$(whoami)/hostfiles
export DLTS_HOSTFILE=/fsx/home-$(whoami)/hostfiles/hosts_$SLURM_JOBID
###########################################################


###########################################################
# Warning to notify a job has been interrupted
###########################################################
sig_handler()
{
    echo "BATCH interrupted"
    wait # wait for all children, this is important!
}
trap 'sig_handler' SIGINT SIGTERM SIGCONT
###########################################################


###########################################################
# Environment Setup
# TODO: Replace with your own environment setup if you haven't
# set the CONDA_HOME variable above
###########################################################
# source .bashrc
# eval "$(conda shell hook --shell=bash)"
# conda activate stable-neox-env
###########################################################


###########################################################
# NeoX Setup
###########################################################
export WANDB_MODE="online"
export PYTORCH_CUDA_ALLOC_CONF='max_split_size_mb:256'

TRAIN_PATH=/fsx/home-$(whoami)/{TODO: FILL-IN PROJECT PATH}
CONFIG_PATH={TODO: FILL-IN}
WANDB_TOKEN=local-{TODO: FILL-IN TOKEN FROM `https://stability.wandb.io`}

cd $TRAIN_PATH

wandb login --relogin --host https://stability.wandb.io $WANDB_TOKEN
git config --global --add safe.directory $TRAIN_PATH

python ./deepy.py train.py $CONFIG_PATH
