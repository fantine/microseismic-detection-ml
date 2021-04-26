#!/bin/bash

# Run hyperparameter tuning.
#
# e.g. $ tunehp.sh mlmodel
#
# @param {model_config} Name of model to run (e.g. mlmodel)
#                This should map to a config file named config/${model_config}.sh.
# @param {job_config} Configuration of job (e.g. config_mlmodel)
#                This should map to a config file named config/${config}.yaml.
#                Defaults to config/config.yaml
# @param {label} Optional label to add to the job ID

# Get params
model_config=$1
dataset=$2
label=$3

# Check config file
config_file=config/$model_config.sh
if [ ! -f "$config_file" ]; then
  echo "Config file not found: $config_file";
  exit 1;
fi

hparams_config=config/hptuning/hp_$model_config.yaml
if [ ! -f "$hparams_config" ]; then
  echo "Config file not found: $hparams_config";
  exit 1;
fi

if [ -z "$dataset" ]; then
  dataset=$default_dataset
fi

# Set job name and directory
now=$(date +%Y%m%d_%H%M%S)
job_name=job_${now}_randomsearch_${model_config}_${dataset}_${label}


echo 'Running randomsearch locally in the background.'
log_file="log/$job_name.log"
echo "Logging to file: $log_file"
python hptuning/random_search.py \
  --model_template=$model_config \
  --hparams_config=$hparams_config \
  --dataset=$dataset 2>&1 | tee $log_file
