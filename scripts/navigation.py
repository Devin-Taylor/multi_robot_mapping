import rospy
from sensor_msgs.msg import LaserScan

import numpy as np


def braitenberg(front, front_left, front_right, left, right):
    const_u = 0.
    dist = replace_infnan(np.array([left, front_left, front, front_right, right])) # front, front_left, front_right, left, right
    dist = dist/4.

    response_l = np.array([-.15*dist[0]+.3, .8*np.exp(-5*dist[1]), -.1*dist[2], -.8*np.exp(-5*dist[3]), .15*dist[4]-.15]) # -dist[0]+1, -dist[4],
    response_r = np.array([.15*dist[0]-.15, -.8*np.exp(-5*dist[1]), -.1*dist[2], .8*np.exp(-5*dist[3]), -.15*dist[4]+.3]) #

    u_l = np.sum(response_l) + const_u
    u_r = np.sum(response_r) + const_u
    u = (u_r + u_l) / 2
    w = u_r - u_l

    return u, w

def simple_controller(front, front_left, front_right, left, right):
    const_w = 0.2
    const_u = 0.2
    u_reverse = -0.1
    w_reset = 1
    u = const_u
    w = 0

    x_near = 1.5
    x_close_front = 0.8
    x_close_side = 0.2

    if(front_left < front_right and front_left < x_near):
        w = -const_w
    if(front_left > front_right and front_right < x_near):
        w = const_w

    # If all else fails, stop and turn
    if(front < x_close_front or front_left < x_close_side or front_right < x_close_side):
        return u_reverse, w_reset
    return u, w


def replace_infnan(vec, lower=0.001, upper=3.5):
    vec = np.array(vec)

    if vec.size > 1:
        vec[vec == np.nan] = lower # lower bound is 0.12 so just set to some small value
        vec[vec == np.inf] = upper # upper bound is 3.5 so just set to some large value
        vec[vec == -np.inf] = upper
    else:
        if vec == np.inf:
            return upper
        elif vec == -np.inf:
            return upper
        elif vec == np.nan:
            return lower
    return vec

class SimpleLaser(object):
  '''
  Adapted from L310 coursework
  '''
  def __init__(self, robot_id):
    self.robot_id = robot_id
    rospy.Subscriber('robot{}/scan'.format(self.robot_id), LaserScan, self.callback)
    self._angles = [0., np.pi / 4., -np.pi / 4., np.pi / 2., -np.pi / 2.]
    self._width = np.pi / 180. * 3.1  # 3.1 degrees cone of view (3 rays).
    self._measurements = [float('inf')] * len(self._angles)
    self._indices = None

  def callback(self, msg):
    # Helper for angles.
    def _within(x, a, b):
      pi2 = np.pi * 2.
      x %= pi2
      a %= pi2
      b %= pi2
      if a < b:
        return a <= x and x <= b
      return a <= x or x <= b;

    # Compute indices the first time.
    if self._indices is None:
      self._indices = [[] for _ in range(len(self._angles))]
      for i, d in enumerate(msg.ranges):
        angle = msg.angle_min + i * msg.angle_increment
        for j, center_angle in enumerate(self._angles):
          if _within(angle, center_angle - self._width / 2., center_angle + self._width / 2.):
            self._indices[j].append(i)

    ranges = np.array(msg.ranges)
    for i, idx in enumerate(self._indices):
      # We do not take the minimum range of the cone but the 10-th percentile for robustness.
      self._measurements[i] = np.percentile(ranges[idx], 10)

  @property
  def ready(self):
    return not np.isnan(self._measurements[0])

  @property
  def measurements(self):
    return self._measurements
