# Microseismic event detection using deep learning

- **Author:** Fantine Huot

## Getting started

### Update the submodules
After cloning the repository, make sure to run the following commands to
initialize and update the submodules.

```
git submodule init
git submodule update
```

### Requirements

- TensorFlow

## Folder structure

- **bin:** Scripts to run jobs.
- **config:** Configuration files. 
- **log**: Log files.
- **trainer**: Machine learning model trainer.

## Train a machine learning (ML) model

This repository provides a parameterized, modular framework for creating and
running ML jobs.

### Run a job
To train a machine learning model, use the following command:
```
bin/train.sh model_config dataset
```

- `model_config`: Name of ML model configuration to use. This should correspond 
to a configuration file named `config/model_config.sh`.
- `dataset`: Dataset identifier. Check the variables `datapath`, `train_file`,
and `eval_file` in `bin/train.sh` to ensure that this maps to the correct input
 data.
- `label`: Optional label to add to the job name.

### Set parameters for a job
Parameters for an ML job can be set by creating a corresponding configuration
file: `config/your_model_config.sh`. 

### Create a new ML model architecture
- Create a new `your_model.py` file inside the `trainer/model` folder. Look at
other models inside the folder for examples.
- Reference your new model in `trainer/model/__init__.py`.
- Set the `model` argument to your new model's name in your model configuration
file `config/your_model_config.sh`.

## Hyperparameter tuning

The hyperparameters are tuned using bayesian optimization. 

### Run a hyperparameter tuning job
To tune the hyperparameters for a machine learning model, use the following
command:
```
bin/tunehp.sh model_config dataset
```

- `model_config`: Name of ML model configuration to use. This should correspond 
to a configuration file named `config/model_config.sh`.
- `dataset`: Dataset identifier. Check the variables `datapath`, `train_file`,
and `eval_file` in `bin/train.sh` to ensure that this maps to the correct input
 data.

### Define the domain for hyperparameter tuning

You can define the domain to explore for hyperparameter tuning by creating a
corresponding configuration file: `config/your_model_config_hptuning.yaml`. 
Look at other hyperparameter tuning configuration files for examples.