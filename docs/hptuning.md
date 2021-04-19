# Hyperparameter tuning

The hyperparameters are tuned using Bayesian optimization. 

## Run a hyperparameter tuning task
To tune the hyperparameters for an ML model:
```bash
bin/tunehp.sh model_config dataset
```

- `model_config`: Name of the ML model configuration to use. This should
correspond to a configuration file named `config/model_config.sh`.
- `dataset`: Dataset identifier. Check the variables `train_file` and
`eval_file` in `bin/train.sh` to ensure that this maps to the correct data.

## Define the domain for hyperparameter tuning

You can define the domain to explore for hyperparameter tuning by creating a
corresponding configuration file: `config/your_model_config_hptuning.yaml`. 
Look at other hyperparameter tuning configuration files in `config/` for
examples.