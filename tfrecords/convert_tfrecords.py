import argparse
import enum
import logging
import os
import random
import re
import sys
from typing import Text

import numpy as np
import tensorflow as tf
import yaml


random.seed(42)


_DATAPATH_FILE = 'config/datapath.sh'


class CompressionType(enum.Enum):
  GZIP = 'GZIP'
  NONE = ''

def _float_feature(data):
  return tf.train.Feature(float_list=tf.train.FloatList(value=data.reshape(-1)))


_FILE_EXTENSION = {
  CompressionType.GZIP: '.gz',
  CompressionType.NONE: '',
}


def create_tf_example(inputs, labels):
  feature_dict = {
      'inputs': _float_feature(inputs),
      'labels': _float_feature(labels),
  }
  return tf.train.Example(features=tf.train.Features(feature=feature_dict))


class DataLoader():
  def __init__(self, min_val, max_val):
    self.min_val = min_val
    self.max_val = max_val

  def _clip_and_rescale(self, data):
    data = np.clip(data, self.min_val, self.max_val)
    return np.divide((data - self.min_val), (self.max_val - self.min_val))

  @staticmethod
  def _get_label(filename):
    if 'noise' in filename:
      return np.zeros((1,), dtype=np.float32)
    return np.ones((1,), dtype=np.float32)

  def read(self, filename):
    inputs = np.float32(np.load(filename))
    if self.min_val != 0.0 or self.max_val != 1.0:
      inputs = self._clip_and_rescale(inputs)
    labels = self._get_label(filename)
    return inputs, labels


def _get_file_suffix(compression_type):
  return '.tfrecord{}'.format(_FILE_EXTENSION[compression_type])


def read_manifest(manifest_file):
  with open(manifest_file, 'r') as f:
    file_list = [line.rstrip() for line in f]
  logging.info('Converting %s files into TFRecords.', len(file_list))
  return file_list


def _glob(file_pattern):
  return sorted(tf.io.gfile.glob(file_pattern))


def create_manifest(manifest_file, file_pattern, shuffle=True):
  file_list = _glob(file_pattern)
  if shuffle:
    random.shuffle(file_list)
  os.makedirs(os.path.dirname(manifest_file), exist_ok=True)
  with open(manifest_file, 'w') as f:
    for filename in file_list:
      f.write(filename + '\n')

def _get_datapath():
  regex_pattern = r'DATAPATH="(\S+)"'
  with open(_DATAPATH_FILE, 'r') as f:
    datapath_text = f.read()
  regex_match = re.search(regex_pattern, datapath_text)
  if regex_match:
    return regex_match.group(1)
  raise ValueError('Please set a correct datapath in {}'.format(_DATAPATH_FILE))


def convert_to_tfrecords(params):
  datapath = _get_datapath()
  print(datapath)
  manifest_file = os.path.join(datapath, params.manifest_file)
  if not os.path.exists(manifest_file):
    logging.info('Creating manifest file: %s', manifest_file)
    create_manifest(manifest_file, os.path.join(datapath, params.input_file_pattern))
  else:
    logging.info('Using the existing manifest file: %s', manifest_file)

  file_list = read_manifest(manifest_file)
  file_shards = np.array_split(file_list, params.num_shards)
  file_suffix = _get_file_suffix(params.compression_type)
  options = tf.io.TFRecordOptions(compression_type=params.compression_type)
  data_loader = DataLoader(params.min_val, params.max_val)
  output_file_prefix = os.path.join(datapath, params.output_file_prefix)

  os.makedirs(os.path.dirname(output_file_prefix), exist_ok=True)
  for i, file_shard in enumerate(file_shards):
    tfrecord_file = '{}-{:04d}-of-{:04d}{}'.format(
        output_file_prefix, i, params.num_shards, file_suffix)
    logging.info('Writing %s', tfrecord_file)
    with tf.io.TFRecordWriter(tfrecord_file, options=options) as writer:
      for filename in file_shard:
        inputs, outputs = data_loader.read(filename)
        tf_example = create_tf_example(inputs, outputs)
        writer.write(tf_example.SerializeToString())

class ArgumentParser():

  def __init__(self):
    config_parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        add_help=False)

    config_parser.add_argument(
        '-c', '--config-file',
        help='Parse script arguments from config file.',
        default=None,
        metavar='FILE')

    self._config_parser = config_parser

    self._parser = argparse.ArgumentParser(parents=[config_parser])

  @staticmethod
  def _parse_config(items):
    argv = []
    for k, v in items:
      argv.append('--{}'.format(k))
      argv.append(v)
    return argv

  def _add_arguments(self, defaults=None):
    parser = self._parser

    parser.add_argument(
        '--input_file_pattern',
        help='Input data files.',
        default='',
    )
    parser.add_argument(
        '--output_file_prefix',
        help='Output file prefix.',
        default='tfrecords/',
    )
    parser.add_argument(
        '--input_height',
        help='Input data height.',
        type=int,
    )
    parser.add_argument(
        '--input_width',
        help='Input data width.',
        type=int,
        default=1,
    )
    parser.add_argument(
        '--input_depth',
        help='Input data depth.',
        type=int,
        default=1,
    )
    parser.add_argument(
        '--input_channels',
        help='Input data channels.',
        type=int,
        default=1,
    )
    parser.add_argument(
        '--num_shards',
        help='Number of shards to generate.',
        type=int,
        default=0,
    )
    parser.add_argument(
        '--compression_type',
        help='File compression type.',
        type=CompressionType,
        choices=list(CompressionType),
        default=CompressionType.GZIP,
    )
    parser.add_argument(
        '--manifest_file',
        help='Manifest file.',
        default='tfrecords/manifests/manifest.txt',
    )
    parser.add_argument(
        '--log_level',
        help='Logging level.',
        default='INFO',
    )
    parser.add_argument(
        '--min_val',
        help='Minimum value.',
        type=float,
        default=0.0,
    )
    parser.add_argument(
        '--max_val',
        help='Maximum value.',
        type=float,
        default=1.0,
    )

  def parse_known_args(self, argv):
    args, remaining_argv = self._config_parser.parse_known_args(argv)
    if args.config_file:
      with open(args.config_file, 'r') as config:
        defaults = yaml.safe_load(config)
      defaults['config_file'] = args.config_file
    else:
      defaults = dict()
    self._add_arguments(defaults=defaults)
    self._parser.set_defaults(**defaults)

    return self._parser.parse_known_args(remaining_argv)


def _set_logging(log_level: Text):
  """Sets the logging level to `log_level`."""
  logger = tf.get_logger()
  logger.setLevel(log_level)
  return


def main():
  params, _ = ArgumentParser().parse_known_args(sys.argv[1:])
  _set_logging(params.log_level.upper())
  convert_to_tfrecords(params)


if __name__ == '__main__':
  main()
