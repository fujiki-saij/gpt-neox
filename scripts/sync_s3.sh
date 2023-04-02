# Args:
#  checkpoint-path: The checkpoint path to sync
#  s3-path: The s3 path to sync to
#  start-step: Starting step to sync
#  end-step: Ending step to sync (INCLUSIVE)
#  checkpoint-factor: Step size to sync (default 1000)
#
# Usage:
# sh sync_s3.sh <checkpoint-path> <s3-path> <start-step> <end-step> <checkpoint-factor>
#
# Example:
# sh sync_s3.sh test-lm neox-lm 1000 10000 1000

CHECKPOINT_PATH=$1
S3_PATH=$2
START=$2
END=$3
STEP=$4

for i in `seq $START $STEP $END`
do
	aws s3 sync ${CHECKPOINT_PATH}/global_step${i}/ s3://${S3_PATH}/global_step${i}/
    sudo rm -rf ${CHECKPOINT_PATH}/global_step${i}/
done
