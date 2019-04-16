import rospy
from nav_msgs.msg import OccupancyGrid as OccupancyGrid_msg
from tf import TransformListener
from tf.transformations import euler_from_quaternion

from matplotlib import pyplot as plt

import cv2
import numpy as np
from utils import FREE, OCCUPIED, UNKNOWN, iou, dice

ROBOT_RADIUS = 0.105 / 2.
X = 0
Y = 1
YAW = 2

def merge_maps_naive(ref_map, merge_map):

    ref_map[(merge_map == FREE) & (ref_map != OCCUPIED)] = FREE
    ref_map[(merge_map == OCCUPIED)] = OCCUPIED

    return ref_map, merge_map, True

def check_distance(ref_bot, test_bot, threshold=1.0):
    dist = np.linalg.norm(ref_bot.slam.pose[:2] - test_bot.slam.pose[:2])
    if dist < threshold:
        return True
    else:
        return False

def marge_maps_transform(ref_map, merge_map, max_features=500, good_match_percent=0.15, threshold=0.75):
    ref_map_temp = np.uint8(ref_map.copy())
    merge_map_temp = np.uint8(merge_map.copy())

    ref_map_temp[ref_map_temp == FREE] = 0
    ref_map_temp[ref_map_temp == UNKNOWN] = 0
    ref_map_temp[ref_map_temp == OCCUPIED] = 255
    merge_map_temp[merge_map_temp == FREE] = 0
    merge_map_temp[merge_map_temp == UNKNOWN] = 0
    merge_map_temp[merge_map_temp == OCCUPIED] = 255

    ref_map_temp = cv2.GaussianBlur(ref_map_temp, (5, 5), 0)
    merge_map_temp = cv2.GaussianBlur(merge_map_temp, (5, 5), 0)

    ref_map_temp[ref_map_temp > 0] = 255
    merge_map_temp[merge_map_temp > 0] = 255

    orb = cv2.ORB_create(max_features)
    keypoints1, descriptors1 = orb.detectAndCompute(merge_map_temp, None)
    keypoints2, descriptors2 = orb.detectAndCompute(ref_map_temp, None)

    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors1, descriptors2, None)
    matches.sort(key=lambda x: x.distance, reverse=False)

    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    h, _ = cv2.findHomography(points1, points2, cv2.RANSAC)
    height, width = ref_map.shape

    merge_map_warp = np.float32(merge_map)
    merge_map_warp = np.uint8( 255 * ((merge_map_warp - merge_map_warp.min()) / (merge_map_warp.max() - merge_map_warp.min())))

    registered_map = cv2.warpPerspective(merge_map_warp, h, (width, height), borderValue=100)
    registered_map = np.int32(registered_map)
    # map back
    registered_map[registered_map < 85] = FREE
    registered_map[(registered_map >= 85) & (registered_map < 170)] = UNKNOWN
    registered_map[registered_map >= 170] = OCCUPIED

    acc = dice(merge_map, registered_map)
    rospy.loginfo("Map transform accuracy: {:.4f}".format(acc))

    if acc < threshold:
        rospy.logdebug("Accuracy for map merge too low to be considered for merge - {:.4f}".format(acc))
        return ref_map, registered_map, False

    ref_registered_merged, _, _ = merge_maps_naive(ref_map, registered_map)

    return ref_registered_merged, registered_map, True


class OccupancyGrid(object):
  '''
  Adapted from L310 coursework
  '''
  def __init__(self, values, origin, resolution, msg=None):
    self._original_values = values.copy()
    self.original_origin = origin
    self._values = values.copy()

    self.msg = msg

    # Inflate obstacles (using a convolution).
    inflated_grid = np.zeros_like(values)
    inflated_grid[values == OCCUPIED] = 1.
    w = 2 * int(ROBOT_RADIUS / resolution) + 1
    # inflated_grid = scipy.signal.convolve2d(inflated_grid, np.ones((w, w)), mode='same')
    self._values[inflated_grid > 0.] = OCCUPIED
    self._origin = np.array(origin[:2], dtype=np.float32)
    self._origin -= resolution / 2.
    assert origin[YAW] == 0.
    self._resolution = resolution

  @property
  def values(self):
    return self._values

  @property
  def resolution(self):
    return self._resolution

  @property
  def origin(self):
    return self._origin

  def draw(self):
    plt.imshow(self._original_values.T, interpolation='none', origin='lower',
               extent=[self._origin[X],
                       self._origin[X] + self._values.shape[0] * self._resolution,
                       self._origin[Y],
                       self._origin[Y] + self._values.shape[1] * self._resolution])
    plt.set_cmap('gray_r')

  def get_index(self, position):
    idx = ((position - self._origin) / self._resolution).astype(np.int32)
    if len(idx.shape) == 2:
      idx[:, 0] = np.clip(idx[:, 0], 0, self._values.shape[0] - 1)
      idx[:, 1] = np.clip(idx[:, 1], 0, self._values.shape[1] - 1)
      return (idx[:, 0], idx[:, 1])
    idx[0] = np.clip(idx[0], 0, self._values.shape[0] - 1)
    idx[1] = np.clip(idx[1], 0, self._values.shape[1] - 1)
    return tuple(idx)

  def get_position(self, i, j):
    return np.array([i, j], dtype=np.float32) * self._resolution + self._origin

  def is_occupied(self, position):
    return self._values[self.get_index(position)] == OCCUPIED

  def is_free(self, position):
    return self._values[self.get_index(position)] == FREE

class SLAM(object):
  '''
  Adapted from L310 coursework
  '''
  def __init__(self, robot_id):
    self.robot_id = robot_id
    self._occupancy_grid = None
    rospy.Subscriber('/robot{}/map'.format(self.robot_id), OccupancyGrid_msg, self.callback)
    self._tf = TransformListener()
    # self._occupancy_grid = None # NOTE why was this defined after calling callback?
    self._pose = np.array([np.nan, np.nan, np.nan], dtype=np.float32)

  def callback(self, msg):
    values = np.array(msg.data, dtype=np.int8).reshape((msg.info.width, msg.info.height))
    processed = np.empty_like(values)
    processed[:] = FREE
    processed[values < 0] = UNKNOWN
    processed[values > 50] = OCCUPIED
    processed = processed.T
    origin = [msg.info.origin.position.x, msg.info.origin.position.y, 0.]
    resolution = msg.info.resolution

    # if self._occupancy_grid is not None: # then merge new map with old map

    self._occupancy_grid = OccupancyGrid(processed, origin, resolution, msg)

  def update(self):
    # Get pose w.r.t. map.
    a = 'robot{}/occupancy_grid'.format(self.robot_id)
    b = 'robot{}/base_link'.format(self.robot_id)
    if self._tf.frameExists(a) and self._tf.frameExists(b):
      try:
        t = rospy.Time(0)
        position, orientation = self._tf.lookupTransform('/' + a, '/' + b, t)
        self._pose[X] = position[X]
        self._pose[Y] = position[Y]
        _, _, self._pose[YAW] = euler_from_quaternion(orientation)
      except Exception as e:
        print(e)
    else:
      print('Unable to find:', self._tf.frameExists(a), self._tf.frameExists(b))
    pass

  @property
  def ready(self):
    return self._occupancy_grid is not None and not np.isnan(self._pose[0])

  @property
  def pose(self):
    return self._pose

  @property
  def occupancy_grid(self):
    return self._occupancy_grid
