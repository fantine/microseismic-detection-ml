# Machine learning training and inference

## Train an ML model
To train an ML model, use the following command:
```bash
bin/train.sh model_config dataset
```

- `model_config`: Name of ML model configuration to use. This should correspond 
to a configuration file named `config/model_config.sh`.
- `dataset`: Dataset identifier. Check the variables `train_file` and
`eval_file` in `bin/train.sh` to make sure that this maps to the correct data.

For example, to run the tutorial machine learning model on the baseline dataset:
```bash
bin/train.sh cnn2d_tutorial baseline
```

The training logs are written to a file of the form `log/job_id.log`. Take note
of the `job_id` for it will be required at inference: the trained network
is saved at `DATAPATH/models/job_id/`.

### Configure ML model parameters
The parameters for an ML task can be configured by creating a corresponding
configuration file: `config/your_model_config.sh`. Look at other ML model
configuration files in `config/` for examples.

### Create a new ML model architecture
- Create a new `your_model.py` file inside the `ml_framework/model` folder.
Look at other models inside the folder for examples.
- Reference your new model in `ml_framework/model/__init__.py`.
- Set the `model` argument to your new model's name in your model configuration
file `config/your_model_config.sh`.

## Evaluate a ML model

Once the model is trained, evaluate the model's performance:
```bash
bin/evaluate.sh model_config dataset job_id
```

- `model_config`: Name of ML model configuration to use. This should correspond 
to a configuration file named `config/model_config.sh`.
- `dataset`: Dataset identifier. Check the variable `eval_file` in 
`bin/evaluate.sh` to make sure that this maps to the correct data. 
- `job_id`: The job identifier of the machine learning training from which to
load the trained network. 

This evaluates the model's performance on the test dataset and saves the
corresponding logits to the following file: 
`DATAPATH/models/job_id/eval_logits.npy`
The logits are saved in the same order as provided in the input data pipeline,
which means that they correspond to each line of the TFRecord manifest file.

## Run inference on continuous data

Once the model is trained, run inference on continuous data:
```bash
bin/predict.sh model_config dataset job_id
```

- `model_config`: Name of ML model configuration to use. This should correspond 
to a configuration file named `config/model_config.sh`.
- `dataset`: Dataset identifier. Check the variable `test_file` in 
`bin/predict.sh` to make sure that this maps to the correct data. 
- `job_id`: The job identifier of the machine learning training from which to
load the trained network. 

`test_file` should be a Unix glob pattern that matches the continuous data
Numpy files on which to run inference. For each continuous data file 
`path_to_continuous_data/filename.npy`, a sliding window runs through the data
and the correspoding logits are saved to 
`path_to_continuous_data/filename_logits.npy`. The amount of overlap between
sliding windows can be set by adding an `overlap` argument to the model
configuration.


