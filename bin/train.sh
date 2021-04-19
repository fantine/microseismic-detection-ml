#!/bin/bash

# Run ML model training
#
# e.g. bin/train.sh model_config dataset label
#
# @param {model_config} Name of ML model configuration to use.
#            This should correspond to a configuration file named as follows:
#            config/${model_config}.sh.
# @param {dataset} Dataset identifier.
#            Check the variables `train_file` and `eval_file` to make sure that
#            this maps to the correct data.
# @param {label} Optional label to add to the job name.

# Get arguments
model_config=$1
dataset=$2
label=$3

# Check the datapath config file
datapath_file=config/datapath.sh
if [ ! -f "$datapath_file" ]; then
  echo "Datapath config file not found: $datapath_file";
  exit 1;
fi

# Set datapaths
. "config/datapath.sh"
train_file="${DATAPATH}/tfrecords/${dataset}/train-*.tfrecord.gz"
eval_file="${DATAPATH}/tfrecords/${dataset}/eval-*.tfrecord.gz"

# Check the ML model config file
if [ "$label" != "hptuning" ]; then
  config_file=config/$model_config.sh
else
  config_file=config/autogenerated/$model_config.sh
  # Stripping the time stamp from the model name
  model_config=${model_config:16}
fi
if [ ! -f "$config_file" ]; then
  echo "ML model config file not found: $config_file";
  exit 1;
fi

# Read the ML model config file
. "$config_file"

# Define the job name
now=$(date +%Y%m%d_%H%M%S)
job_name=train_${now}_${model_config}_${dataset}_${label}
job_dir="${DATAPATH}/models/${job_name}"
log_file="log/${job_name}.log"

# Set package and module name
package_path=ml_framework/
module_name=ml_framework.train

# Run the job
if [ "$label" != "hptuning" ]; then
  echo 'Running ML job.'
  echo "Logging to file: $log_file"
  python -m $module_name \
  --job_dir=$job_dir \
  $MODULE_ARGS \
  --train_file=$train_file \
  --eval_file=$eval_file 2>&1 | tee $log_file
else # if this is a hyperparameter tuning job, run it in the foreground
  echo "Logging to file: $log_file"
  python -m $module_name \
  --job_dir=$job_dir \
  $MODULE_ARGS \
  --train_file=$train_file \
  --eval_file=$eval_file \
  > $log_file 2>&1
fi