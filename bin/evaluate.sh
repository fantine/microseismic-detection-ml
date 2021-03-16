#!/bin/bash

# Run a ML model training job
#
# e.g. bin/train.sh model_config dataset label
#
# @param {model_config} Name of ML model configuration to use.
#            This should correspond to a configuration file named as follows:
#            config/${model_config}.sh.
# @param {dataset} Dataset identifier.
#            Check the variables `datapath`, `train_file`, and `eval_file`,
#            to ensure that this maps to the correct input data.
# @param {job_id} Job name of the ML model to evaluate.
# @param {label} Optional label to add to the job name.

# Get arguments
model_config=$1
dataset=$2
job_id=$3
label=$4

# Set path to input data
. "config/datapath.sh"
eval_file="${DATAPATH}/tfrecords/${dataset}/test-*.tfrecord.gz"
ckpt="${DATAPATH}/models/${job_id}/ckpt"

# Check the ML model config file
config_file=config/$model_config.sh
if [ ! -f "$config_file" ]; then
  echo "ML model config file not found: $config_file";
  exit 1;
fi

# Read the ML model config file
. "$config_file"

# Define the job name
now=$(date +%Y%m%d_%H%M%S)
job_name=evaluate_${now}_${model_config}_${dataset}_${label}
log_file="log/${job_name}.log"

# Set package and module name
package_path=ml_framework/
module_name=ml_framework.evaluate

# Run the job
echo 'Running ML evaluation.'
echo "Logging to file: $log_file"
python -m $module_name \
--job_dir=$ckpt \
$MODULE_ARGS \
--eval_file=$eval_file 2>&1 | tee $log_file