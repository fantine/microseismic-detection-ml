# Convert input data to TensorFlow records (TFRecords)

Once all the machine learning examples are preprocessed, we convert them into 
TensorFlow records to optimize the input pipeline for machine learning
training and inference. 

Reading large numbers of small files significantly impacts I/O performance.
For large datasets, we preprocess the input data into larger (~100MB) TFRecord
files to get maximum I/O throughput. 

## Create TFRecords manifest files
To generate the TFRecords, we create a manifest file for the dataset.
The manifest file serves as a recipe for generating the TFRecords.
It is a text file that contains all the filenames that constitute the dataset.

We generate a manifest files for each of the three datasets: training,
evaluation, and testing. We take the full list of data files, separate them
into these three datasets, and shuffle them, before writing out each manifest
file.

## Create TensorFlow records
Convert the data files into TFRecord files with the following command:
```bash
python -m tfrecords.convert_tfrecords -c config/tfrecord_baseline_train.yaml
```

The `-c` flag specifies a configuration file.
You can create your own TFRecord configuration file inside the `config/` folder
and replace `config/tfrecord_baseline_train.yaml` with the name of your
TFRecord configuration file. Look at other TFRecord configuration files for
examples.

In the TFRecord configuration file, the `DATAPATH` specified in 
`config/datapath.sh` will automatically be prefixed to all the paths, so all
the paths should be specified as relative paths.

## TFRecord configuration files
Variables in the TFRecord configuration file:

- `manifest_file`: Manifest file that contains all the filenames that
constitute the dataset to convert into TFRecords. If no manifest file is found,
the script will use `input_file_pattern` to create the list of filenames, and
write them to the file specified by `manifest_file`.

- `input_file_pattern`: A Unix glob file pattern. When a manifest file is
provided, this variable is ignored. If no manifest file is found, the script
uses this file pattern to create the list of files for the manifest file.

- `output_file_prefix`: Filename prefix to write the TFRecords.

- `num_shards`: Number of TFRecord shards to generate. Adjust this number to
generate files of about 100Mb. 

- `min_val` and `max_val` (optional): When specified, the data are clipped and
rescaled using these values, scaling the dataset to the [0, 1] range.