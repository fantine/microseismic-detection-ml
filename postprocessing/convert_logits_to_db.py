import datetime
import logging
import os
from typing import List, Sequence, Text, Tuple
import sqlite3
import numpy as np
import tensorflow as tf

from config import get_datapath
from gcp_io import gcp_io


def _get_file_containing_timestamp(
        timestamp, filenames, file_times, file_length):
  """Gets the name of the file that contains `timestamp`."""
  # Search is faster on a sorted array
  index = np.searchsorted(file_times, timestamp, 'right') - 1
  # or if index is not valid
  if index < 0:
    return None
  timestamp = (timestamp - file_times[index]).total_seconds()
  if timestamp > file_length:
    return None
  return filenames[index]


def convert_timestamp_to_entry(
        timestamp,
        channel: int, raw_data_files, file_starttimes, file_length, dt
) -> Tuple[Text, int, int, float]:
  """Converts a time stamp to an entry to the database.

  Args:
      timestamp: Time stamp to convert to database entry.
      channel: Channel from which to pick.

  Returns:
      (filename, pick_chan, pick_samp, dt) entry.
  """
  filename = _get_file_containing_timestamp(
      timestamp, raw_data_files, file_starttimes, file_length)
  if filename is None:
    return None
  file_start = datetime.datetime.strptime(
      os.path.basename(filename), '%Y%m%d_%H%M%S.npy')
  pick_samp = int((timestamp - file_start).total_seconds() / dt)
  return filename, channel, pick_samp, dt


def write_to_database(entries: List[Tuple[Text, int, int, float]], filename: Text):
  """Writes entries to a database.

  Creates a database with columns origin_filename, pick_chan, pick_samp, dt,
  fills it with the values from `entries`, and saves it to `filename`.

  Args:
      entries: List of entries to the database.
      filename: File to which to write the database.
  """
  db_connect = sqlite3.connect(filename)
  db_cursor = db_connect.cursor()
  db_cursor.execute(
      'CREATE TABLE events (origin_filename text, pick_chan integer, '
      'pick_samp integer, dt real)'
  )
  db_cursor.executemany('INSERT INTO events VALUES (?, ?, ?, ?)', entries)
  db_connect.commit()
  db_connect.close()


def _convert_filenames_to_datetimes(
        filenames: List[Text]) -> Sequence[datetime.datetime]:
  """Convert filenames to datetimes.

  Args:
    filenames: A list of file names.

  Returns:
    An array of sorted datetimes corresponding to the filenames.
  """
  datetimes = []
  for filename in filenames:
    try:
      datetimes.append(datetime.datetime.strptime(
          os.path.basename(filename), '%Y%m%d_%H%M%S.npy'))
    except ValueError:
      logging.warning(
          'File %s does not match naming convention, skipping.',
          filename)
  datetimes.sort()
  return np.array(datetimes)


def _get_middle_channel(logits_file):
  first_channel = int(logits_file[40:44])
  last_channel = int(logits_file[48:52])
  return float(last_channel + first_channel) / 2.0


def get_eventtimes(logits_files, data_files):
  overlap = 0.9375
  input_shape = (512, 128)
  channel_threshold = 1
  n = 8
  n_votes = 2
  dt = 1 / 500
  filename_pattern = '%Y%m%d_%H%M%S'

  entries = []
  for i, logits_file in enumerate(logits_files):
    logits = gcp_io.read(logits_file)
    data = gcp_io.read(data_files[i])
    n1 = data.shape[0]
    w1, w2 = input_shape
    d1 = int((1. - overlap) * w1)
    # d2 = int((1. - overlap) * w2)
    predictions = tf.round(tf.nn.sigmoid(logits)).numpy()
    predictions = predictions.reshape((len(range(0, n1 - w1 + 1, d1)), -1))
    end = n * int(predictions.shape[1] / n)
    # consolidated = np.min(predictions[:, :end].reshape(
    #     predictions.shape[0], -1, n), axis=2)
    predictions = predictions[:, :end].reshape(predictions.shape[0], -1, n)
    consolidated = np.zeros((predictions.shape[0], predictions.shape[1]))
    for i in range(predictions.shape[0]):
      for j in range(predictions.shape[1]):
        consolidated[i, j] = np.min(predictions[i, j, :n_votes])
    prediction_th = np.where(
        np.sum(consolidated, axis=0) >= channel_threshold)[0]
    prediction_th = (prediction_th * n * w2 * (1. - overlap) + w2 // 4)
    basename = os.path.basename(logits_file)
    print('basename', basename)
    start_time = datetime.datetime.strptime(basename[:15], filename_pattern)
    print('file start', start_time)
    for i, sample in enumerate(prediction_th):
      if i == 0:
        print('first sample', sample)
      eventtime = start_time + datetime.timedelta(seconds=sample * dt)
      if i == 0:
        print('first event time', eventtime)
      channel = _get_middle_channel(basename)
      entries.append((eventtime, channel))
  return entries


def convert_to_database(logits_files, data_files, raw_data_files, file_length, dt, database_file):
  eventtimes = get_eventtimes(logits_files, data_files)
  file_starttimes = _convert_filenames_to_datetimes(raw_data_files)
  entries = []
  print('first eventtimes', eventtimes[0])
  for eventtime, channel in eventtimes:
    entry = convert_timestamp_to_entry(
        eventtime, channel, raw_data_files, file_starttimes, file_length, dt)
    if entry is not None:
      entries.append(entry)
  print('first entry', entries[0])
  write_to_database(entries, database_file)


def get_filenames(manifest_file: Text) -> List[Text]:
  """Gets all the raw data filenames."""
  with open(manifest_file, 'r') as f:
    filenames = [line.rstrip() for line in f]
  filenames.sort()
  return filenames


def main():
  datapath = get_datapath.get_datapath()
  logits_files = tf.io.gfile.glob(os.path.join(
      datapath, 'continuous_data/*_logits.npy'))
  data_files = [filename.replace('_logits.npy', '.npy')
                for filename in logits_files]
  raw_data_files = get_filenames('postprocessing/raw_data_filenames.txt')
  convert_to_database(
      logits_files, data_files, raw_data_files, file_length=30 * 60, dt=0.002,
      database_file='inference.db'
  )


if __name__ == '__main__':
  main()
