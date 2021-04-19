#!/bin/bash

# Run ML model hyperparameter tuning
#
# e.g. bin/tunehp.sh model_config dataset label
#
# @param {model_config} Name of ML model configuration to use.
#            This should correspond to a configuration file named as follows:
#            config/${model_config}.sh.
# @param {dataset} Dataset identifier.
#            Check the variables `train_file`, and `eval_file` in `bin/train.sh`
#            to make sure that this maps to the correct data.

# Get arguments
model_config=$1
dataset=$2

# Check the ML model config file
config_file=config/$model_config.sh
if [ ! -f "$config_file" ]; then
  echo "ML model config file not found: $config_file";
  exit 1;
fi

hptuning_config=config/${model_config}_hptuning.yaml
if [ ! -f "$hptuning_config" ]; then
  echo "Hyperparameter tuning config file not found: $hptuning_config";
  exit 1;
fi

# Set job name
now=$(date +%Y%m%d_%H%M%S)
job_name=hptuning_job_${now}_${model_config}_${dataset}
log_file="log/${job_name}.log"

# Set package and module name
package_path=hptuning/
module_name=hptuning.bayes_opt

echo 'Running hyperparameter tuning job.'
echo "Logging to file: $log_file"
python -m $module_name \
  --model_config=$model_config \
  --hptuning_config=$hptuning_config \
  --dataset=$dataset \
  --label=$now 2>&1 | tee $log_file
