import argparse
import enum
import logging
import os
import random
import re
import sys

import numpy as np
import tensorflow as tf
import yaml

from config import get_datapath


random.seed(42)


class CompressionType(enum.Enum):
  GZIP = 'GZIP'
  NONE = ''


_FILE_EXTENSION = {
    CompressionType.GZIP: '.gz',
    CompressionType.NONE: '',
}


logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)


# def _float_feature(data):
#   return tf.train.Feature(float_list=tf.train.FloatList(value=data.reshape(-1)))

def _bytes_feature(data):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[data]))


def create_tf_example(inputs, labels):
  feature_dict = {
      'inputs': _bytes_feature(inputs.tobytes()),
      'labels': _bytes_feature(labels.tobytes()),
  }
  return tf.train.Example(features=tf.train.Features(feature=feature_dict))


class DataLoader():
  def __init__(self, min_val, max_val):
    self.min_val = min_val
    self.max_val = max_val

  def _clip_and_rescale(self, data):
    data = np.clip(data, self.min_val, self.max_val)
    return np.divide((data - self.min_val), (self.max_val - self.min_val))

  @ staticmethod
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


def convert_to_tfrecords(params):
  datapath = get_datapath.get_datapath()
  manifest_file = os.path.join(datapath, params.manifest_file)
  if not os.path.exists(manifest_file):
    logging.info('Creating manifest file: %s', manifest_file)
    create_manifest(manifest_file, os.path.join(
        datapath, params.input_file_pattern))
  else:
    logging.info('Using the existing manifest file: %s', manifest_file)

  file_list = read_manifest(manifest_file)
  file_shards = np.array_split(file_list, params.num_shards)
  file_suffix = _get_file_suffix(params.compression_type)
  options = tf.io.TFRecordOptions(
      compression_type=params.compression_type.value)
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

  @ staticmethod
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


def main():
  params, _ = ArgumentParser().parse_known_args(sys.argv[1:])
  convert_to_tfrecords(params)


if __name__ == '__main__':
  main()
