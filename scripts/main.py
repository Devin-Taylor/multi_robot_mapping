#!/usr/bin/env python

from __future__ import absolute_import, division, print_function

import argparse
import json
import os
import sys
import time

import rospy
import tf
# Goal.
# Robot motion commands:
# http://docs.ros.org/api/geometry_msgs/html/msg/Twist.html
from geometry_msgs.msg import Twist
from nav_msgs.msg import OccupancyGrid as OccupancyGrid_msg

# For pose information.
import numpy as np
import scipy.signal
from mapping import (SLAM, OccupancyGrid, check_distance, marge_maps_transform,
                     merge_maps_naive)
from matplotlib import pyplot as plt
from navigation import SimpleLaser, braitenberg, simple_controller
from utils import *

if not os.path.exists("imgs"):
  os.mkdir("imgs")

ROOT = "/home/devin/catkin_ws/src/multi_robot"
RESULTS = os.path.join(ROOT, "results")

if not os.path.exists(RESULTS):
  os.mkdir(RESULTS)

class GlobalMap(object):
  def __init__(self):
    self._map = None

  @property
  def map(self):
    if self._map is None:
      return self._map
    else:
      return self._map.copy()

  @map.setter
  def map(self, map):
    self._map = map

class Robot(object):

  def __init__(self, id):
    self.id = id
    self.br = tf.TransformBroadcaster()
    self.vel_pub = rospy.Publisher('robot{}/cmd_vel'.format(self.id), Twist, queue_size=5)
    self.slam = SLAM(self.id)
    self.laser = SimpleLaser(self.id)

    self._map = None
    # self.available_map = None

    self.map_cache = {}
    self.transformed_map_cache = {}

  @property
  def map(self):
    return self._map.copy()

  def update_slam(self):
    self.slam.update()
    self._map = self.slam.occupancy_grid.values

  def broadcast_world_tsf(self, time_now, tsf=None):
    if tsf is None:
      translation = (0.0, 0.0, 0.0)
      rotation = (0.0, 0.0, 0.0, 1.0)
    else:
      raise NotImplementedError

    self.br.sendTransform(translation,
                          rotation,
                          time_now,
                          'robot{}/map'.format(self.id),
                          "world")

  def publish_velocity(self):
    u, w = simple_controller(*self.laser.measurements)
    vel_msg = Twist()
    vel_msg.linear.x = u
    vel_msg.angular.z = w
    self.vel_pub.publish(vel_msg)


def run(args):
  rospy.init_node('multirobot_slam')

  results = {
    "time": [],
    "frame": [],
    "accuracy": [],
    "r1_accuracy": [],
    "r2_accuracy": [],
    "r3_accuracy": []
  }

  num_robots = 3
  ref_bot_id = 1
  decentralised = False
  origin_known = True
  run_number = 2

  result_filename = "results_numbots-{}_decentralised-{}_originknown-{}_run-{}.json".format(num_robots, decentralised, origin_known, run_number)

  if origin_known:
    map_merging_function = merge_maps_naive
  else:
    map_merging_function = marge_maps_transform

  test_bot_ids = [i for i in range(num_robots) if i != ref_bot_id]

  # Update control every 100 ms.
  rate_limiter = rospy.Rate(100)

  robots = [Robot(x) for x in range(1, num_robots+1)]

  slam_publisher = rospy.Publisher("robot_combined/map", OccupancyGrid_msg, queue_size=1)
  combined_broadcaster = tf.TransformBroadcaster()
  global_map = GlobalMap()
  slam_msg = OccupancyGrid_msg()
  ref_map = reference_map(map_name="/home/devin/catkin_ws/src/multi_robot/map/project_medium_v2")

  # Make sure the robot is stopped.
  i = 0
  while i < 10 and not rospy.is_shutdown():
    rate_limiter.sleep()
    i += 1

  counter = 0
  start = time.time()
  while not rospy.is_shutdown():
    counter += 1

    current_time = rospy.Time.now().to_sec()
    time_now = rospy.Time.now()

    combined_broadcaster.sendTransform((0.0, 0.0, 0.0),
                                      (0.0, 0.0, 0.0, 1.0),
                                      time_now,
                                      'robot_combined/map',
                                      "world")

    for robot in robots:
      robot.update_slam()
      robot.broadcast_world_tsf(time_now)
      robot.publish_velocity()

    if global_map.map is None:
      global_map._map = robots[ref_bot_id].map.T # set to default as the reference bot

    proposed_map = robots[ref_bot_id].map.T

    if test_bot_ids:
      for test_bot_id in test_bot_ids:
        can_merge_flag = True
        if decentralised: # if decentralised, validate distance between robots
          if not check_distance(robots[ref_bot_id], robots[test_bot_id]):
            can_merge_flag = False
        if not can_merge_flag: # if not close enough for new update just merge in latest available map
          if robots[ref_bot_id].map_cache.get(test_bot_id) is not None:
            # try merge latest available map
            proposed_map, _, merged = map_merging_function(proposed_map, robots[ref_bot_id].map_cache.get(test_bot_id))
            if not merged: # last resort, merge previously valid transformed map in a naive way
              proposed_map, _, _ = merge_maps_naive(proposed_map, robots[ref_bot_id].map_cache.get(test_bot_id))
          continue

        proposed_map, transformed_map, merged = map_merging_function(proposed_map, robots[test_bot_id].map.T)
        if merged: # if it is mergable then save as last available map and transformed version of map
          robots[ref_bot_id].map_cache[test_bot_id] = robots[test_bot_id].map.T
          robots[ref_bot_id].transformed_map_cache[test_bot_id] = transformed_map
        else:
          # unfortunate repitition of code above but was due to saving computation
          if robots[ref_bot_id].map_cache.get(test_bot_id) is not None:
            proposed_map, _, merged = map_merging_function(proposed_map, robots[ref_bot_id].map_cache.get(test_bot_id))
            if not merged:
              proposed_map, _, _ = merge_maps_naive(proposed_map, robots[ref_bot_id].map_cache.get(test_bot_id))


    global_map._map = proposed_map
    final_map_converted = convert_to_publish(global_map.map)

    slam_msg = robots[0].slam.occupancy_grid.msg # HACK
    slam_msg.header.frame_id = "robot_combined/map"
    slam_msg.data = list(map(int, list(final_map_converted.flatten())))
    slam_publisher.publish(slam_msg)

    score = dice(ref_map, global_map.map, strict=False)
    rospy.loginfo("Mapping score: {:.2f}%".format(score*100))

    results['time'].append(time.time() - start)
    results['frame'].append(counter)
    results['accuracy'].append(score)

    for r in robots:
      results['r{}_accuracy'.format(r.id)].append(dice(ref_map, r.map.T, strict=False))


    with open(os.path.join(RESULTS, result_filename), "w+") as fd:
      json.dump(results, fd)

    rate_limiter.sleep()

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Runs multi-robot SLAM')
  args, unknown = parser.parse_known_args()
  try:
    run(args)
  except rospy.ROSInterruptException:
    pass
