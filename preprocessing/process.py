"""Data processing."""

import logging
import os

import numpy as np

from preprocessing import parameters
from processing_utils import processing_utils as processing

logging.basicConfig(level=logging.INFO)


def _process(data, clip_value, std_channels):
  data = np.clip(data, -clip_value, clip_value)
  data = data / np.expand_dims(std_channels, axis=1)
  return data


def get_start_channel(filename):
  basename = os.path.basename(filename)
  return int(basename[-16:-12])


def process(file_pattern, in_dir, out_dir, clip_value, std_channels):
  filenames = processing.get_filenames(file_pattern)

  for i, filename in enumerate(filenames):
    if i % 1000 == 0:
      logging.info('Processed %s files.', i)
    data = np.load(filename)
    start_ch = get_start_channel(filename)
    n_channels = data.shape[0]
    data = _process(data, clip_value,
                    std_channels[start_ch:start_ch + n_channels])
    out_file = filename.replace(in_dir, out_dir)
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    np.save(out_file, data)


def main():
  file_pattern = os.path.join(parameters.unprocessed_datapath, '*/*')
  std_channels = np.load(parameters.std_channels_file)
  process(
      file_pattern,
      in_dir=parameters.unprocessed_datapath,
      out_dir=parameters.processed_datapath,
      clip_value=parameters.clip_value,
      std_channels=std_channels,
  )


if __name__ == '__main__':
  main()
