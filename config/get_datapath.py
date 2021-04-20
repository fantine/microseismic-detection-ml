"""Retrieves the project datapath."""

import re


_DATAPATH_FILE = 'config/datapath.sh'


def get_datapath():
  """Gets the project datapath."""
  regex_pattern = r'DATAPATH="(\S+)"'
  with open(_DATAPATH_FILE, 'r') as f:
    datapath_text = f.read()
  regex_match = re.search(regex_pattern, datapath_text)
  if regex_match:
    return regex_match.group(1)
  raise ValueError(
      'Please set a correct datapath in {}'.format(_DATAPATH_FILE))
