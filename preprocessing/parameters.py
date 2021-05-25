"""Parameters for data preprocessing."""

import os
from config import get_datapath

# pylint: disable=invalid-name

datapath = get_datapath.get_datapath()

unprocessed_datapath = os.path.join(datapath, 'unprocessed_data')
processed_datapath = os.path.join(datapath, 'processed_data')
std_channels_file = 'std_channels.npy'
clip_value = 11.21  # 99.5th percentile of absolute amplitudes
