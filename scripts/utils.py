import os
import re

from matplotlib import pyplot as plt

import numpy as np
import yaml

FREE = 0
UNKNOWN = 1
OCCUPIED = 2


def read_pgm(filename, byteorder='>'):
  """Read PGM file.
    Adapted from L310 coursework
  """
  with open(filename, 'rb') as fp:
    buf = fp.read()
  try:
    header, width, height, maxval = re.search(
        b'(^P5\s(?:\s*#.*[\r\n])*'
        b'(\d+)\s(?:\s*#.*[\r\n])*'
        b'(\d+)\s(?:\s*#.*[\r\n])*'
        b'(\d+)\s(?:\s*#.*[\r\n]\s)*)', buf).groups()
  except AttributeError:
    raise ValueError('Invalid PGM file: "{}"'.format(filename))
  maxval = int(maxval)
  height = int(height)
  width = int(width)
  img = np.frombuffer(buf,
                      dtype='u1' if maxval < 256 else byteorder + 'u2',
                      count=width * height,
                      offset=len(header)).reshape((height, width))
  return img.astype(np.float32) / 255.

def reference_map(map_name="/home/devin/catkin_ws/src/multi_robot/scripts/stage4", offset=18):
  with open(map_name + '.yaml') as fp:
    data = yaml.load(fp)
  img = read_pgm(os.path.join(os.path.dirname(map_name), data['image']))
  occupancy_grid = np.empty_like(img, dtype=np.int8)
  occupancy_grid[:] = UNKNOWN
  occupancy_grid[img < .1] = OCCUPIED
  occupancy_grid[img > .9] = FREE
  # Transpose (undo ROS processing).
  # Invert Y-axis.
  occupancy_grid = occupancy_grid[:, ::-1]
  occupancy_grid = occupancy_grid.T

  temp = np.ones(occupancy_grid.shape, np.int32)
  temp[offset:occupancy_grid.shape[0], offset:occupancy_grid.shape[1]] = occupancy_grid[:occupancy_grid.shape[0]-offset, :occupancy_grid.shape[1]-offset]

  return temp

def convert_to_publish(final_map):
  temp_map = final_map.copy()
  temp_map[temp_map == 1] = -1
  temp_map[temp_map == 2] = 100
  return temp_map

def iou(reference, mapped):
  '''
  Implements intersection over union loss function
  '''
  mask = np.where(reference != UNKNOWN)
  mask_rows = mask[0]
  mask_cols = mask[1]

  ref = reference.copy()
  temp_map = mapped.copy()
  temp_map = temp_map[:, ::-1]
  temp_map = temp_map[::-1, :]

  temp_map = temp_map[mask_rows, mask_cols]
  ref = ref[mask_rows, mask_cols]

  intersection = sum(ref == temp_map)
  union = len(ref)
  return intersection/union


def dice(reference, mapped, strict=True):
  '''
  strict (boolean): if strict is True then only consider the FREE mapped regions. This is a very hard
                    accuracy metric to use but works for transforms. If False then consider the
                    OCCUPIED and FREE regions, this is more lenient but not as accurate.
  '''
  ref = reference.copy()
  map = mapped.copy()

  # create mask
  if strict:
    ref[ref == UNKNOWN] = OCCUPIED
    ref[ref == FREE] = 1
    ref[ref == OCCUPIED] = 0

    map[map == UNKNOWN] = OCCUPIED
    map[map == FREE] = 1
    map[map == OCCUPIED] = 0
  else:
    ref[ref == FREE] = OCCUPIED
    ref[ref == UNKNOWN] = 0
    ref[ref == OCCUPIED] = 1

    map[map == FREE] = OCCUPIED
    map[map == UNKNOWN] = 0
    map[map == OCCUPIED] = 1

  # compare
  dice = np.sum(map[ref==1])*2.0 / (np.sum(map) + np.sum(ref))
  return dice