# Microseismic event detection on fiber-optic data using machine learning

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

```bash
git submodule init
git submodule update
```

### Requirements

You can run the project from an interactive bash session within the provided
[Docker](https://www.docker.com]) container:
```bash
docker run --gpus all -it fantine/ml_framework:latest bash
```
If you do not have root permissions to run Docker, [Singularity](https://singularity.lbl.gov) might be a good alternative for you. Refer to 
`containers/README.md` for more details.


## Folder structure

- **bin:** Scripts to run machine learning jobs.
- **config:** Configuration files. 
- **containers:** Details on how to use containers for this project. 
- **docs:** Documentation.
- **log:** Directory for log files.
- **ml_framework:** Machine learning framework.
- **tfrecords:** Utility functions for converting files to TFRecords.

## Set the datapath for the project

Set the `DATAPATH` variable inside `config/datapath.sh` to the data or scratch directory
to which you want write data files.

## Create and run a machine learning model

This repository provides a parameterized, modular framework for creating and
running ML models.

- [Convert input data to TensorFlow records](docs/convert_tfrecords.md)
- [Machine learning training and inference](docs/ml_framework.md)
- [Hyperparameter tuning](docs/hptuning.md)
