# Microseismic event detection using machine learning

- **Author:** Fantine Huot

Microseismic analysis is the primary tool available for fracture
characterization in unconventional reservoirs. As distributed acoustic sensing
(DAS) fibers are installed in the target reservoir and are thus close to the microseismic events, they hold vast potential for their high-resolution
analysis.  


However, accurately detecting microseismic signals in continuous data is
challenging and time-consuming. DAS acquisitions generate substantial data
volumes, and microseismic events have a low signal-to-noise ratio in individual
DAS channels. 


In this project, we design, train, and deploy a machine learning model to automatically detect thousands of microseismic events in DAS data acquired
inside a shale reservoir. The stimulation of two offset wells generates the microseismic activity.

The deep learning model achieves an accuracy of over 98\% on our benchmark
dataset of manually-picked events and even detects low-amplitude events missed
during manual picking.  

## Getting started

### Update the submodules
After cloning the repository, run the following commands to initialize and
update the submodules.

```
git submodule init
git submodule update
```

### Requirements

You can run the project from an interactive bash session within the provided
[Docker](https://www.docker.com]) container:
```
docker run --gpus all -it fantine/ml_framework:latest bash
```
If you do not have root permissions to run Docker, [Singularity](https://singularity.lbl.gov) might be a good alternative for you. Refer to 
`containers/README.md` for more details.

## Folder structure

- **bin:** Scripts to run machine learning jobs.
- **config:** Configuration files. 
- **containers:** Details on how to use containers for this project. 
- **gcp_io:** Input/output utility functions to read or write on local disk and
Google Cloud Storage.
- **log:** Directory for log files.
- **ml_framework:** Machine learning framework.
- **plot_utils:** Utility functions for making figures.
- **tfrecords:** Utility functions for converting files to TFRecords.

## Create and run a machine learning (ML) task

This repository provides a parameterized, modular framework for creating and
running ML models.

### Set the datapath

Set the `DATAPATH` variable inside `config/datapath.sh` to the data or scratch
to which you want write data files.

### Train an ML model
To train an ML model, use the following command:
```
bin/train.sh model_config dataset
```

- `model_config`: Name of ML model configuration to use. This should correspond 
to a configuration file named `config/model_config.sh`.
- `dataset`: Dataset identifier. Check the variables `train_file` and `eval_file` in `bin/train.sh` to ensure that this maps to the correct data.
- `label`: Optional label to add to the job name.

### Set ML model parameters
The parameters for an ML task can be configured by creating a corresponding configuration file: `config/your_model_config.sh`. Look at other ML model
configuration files in `config/` for examples.

### Create a new ML model architecture
- Create a new `your_model.py` file inside the `ml_framework/model` folder. Look at other models inside the folder for examples.
- Reference your new model in `ml_framework/model/__init__.py`.
- Set the `model` argument to your new model's name in your model configuration
file `config/your_model_config.sh`.

## Hyperparameter tuning

The hyperparameters are tuned using bayesian optimization. 

### Run a hyperparameter tuning task
To tune the hyperparameters for an ML model, use the following
command:
```
bin/tunehp.sh model_config dataset
```

- `model_config`: Name of the ML model configuration to use. This should correspond to a configuration file named `config/model_config.sh`.
- `dataset`: Dataset identifier. Check the variables `train_file` and `eval_file` in `bin/train.sh` to ensure that this maps to the correct data.

### Define the domain for hyperparameter tuning

You can define the domain to explore for hyperparameter tuning by creating a
corresponding configuration file: `config/your_model_config_hptuning.yaml`. 
Look at other hyperparameter tuning configuration files in `config/` for examples.