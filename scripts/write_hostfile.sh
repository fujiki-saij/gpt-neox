#!/bin/bash
hostfile=$HOME/hostfiles/hosts_$SLURM_JOBID
for i in $(scontrol show hostnames "$SLURM_JOB_NODELIST")
do
    echo $i slots=8 >>$hostfile
done
