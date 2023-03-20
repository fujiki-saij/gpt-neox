#!/bin/bash
#SBATCH --account="stablegpt"
#SBATCH --job-name="neox"
# #SBATCH --time=08:00:00
#SBATCH --partition=g40
#SBATCH --nodes=32
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=11G
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --exclude=ip-26-0-130-[12-13,19,22,32,37-38,41,44,52,60,87,116,127,132,134,147-148,150,163-164,183,193],ip-26-0-131-[4-5,38,51,77,85,89,107-108],ip-26-0-140-150
#SBATCH --output=logs/neox_%j.out
#SBATCH --error=logs/neox_%j.err

module load openmpi cuda/11.7

###########################################################
# Pre-load
###########################################################
CUDNN_HOME=/fsx/quentin/cudnn-linux-x86_64-8.6.0.163_cuda11-archive
export LD_LIBRARY_PATH=$CUDNN_HOME/lib:$LD_LIBRARY_PATH
export CPATH=$CUDNN_HOME/include:$CPATH

CONDA_HOME=$HOME/micromamba/envs/stable-neox-env
export PATH=$CONDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CONDA_HOME/lib:$LD_LIBRARY_PATH
export CPATH=$CONDA_HOME/include:$CPATH

# export LD_LIBRARY_PATH=/opt/aws-ofi-nccl/lib:/opt/amazon/efa/lib64:/usr/local/cuda-11.7/efa/lib:/usr/local/cuda-11.7/lib:/usr/local/cuda-11.7/lib64:/usr/local/cuda-11.7:/opt/nccl/build/lib:/opt/aws-ofi-nccl-install/lib:/opt/aws-ofi-nccl/lib:$LD_LIBRARY_PATH
# export PATH=/opt/amazon/efa/bin:/opt/amazon/openmpi/bin:/usr/local/cuda-11.7/bin:$PATH
# Force PyTorch to use system NCCL install
# export PATH="/usr/local/cuda-11.7/bin:$PATH"
# export LD_LIBRARY_PATH="/usr/local/cuda-11.7/lib64:$LD_LIBRARY_PATH"
# export LD_PRELOAD="/opt/nccl/build/lib/libnccl.so"
###########################################################

ds_report

###########################################################
# CUDA/Torch Setup
###########################################################
mkdir -p $HOME/stablegpt/nccl-logs/$SLURM_JOB_ID  
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

TRAIN_PATH=$HOME/stablegpt/gpt-neox
cd $TRAIN_PATH

python ./deepy.py train.py --conf_dir configs 
