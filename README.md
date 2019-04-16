# Multi Robot Mapping

This repository provides an implementation of multi-robot mapping, implemented for the L310 Multi-Robot Systems.

The implementation considers the different scenarios:

* Centralised and decentralised mapping server. When decentralised, communication of maps between robots can only happen when they are close to one another.

* Known and unknown initial positions in global map. When the initial position is unknown ORB feature detection with a perspective transform is used to estimate the transformation between individual robot coordinate systems.


### Environment

* Ubuntu 18.04

* ROS Melodic

* Gazebo v9.0

* RViz v1.13.1

* Python 3.6.5

### Usage

Launch envrionments:

> roslaunch multi_robot main.launch 

Run navigation:

> roslaunch multi_robot launch_slam.launch
