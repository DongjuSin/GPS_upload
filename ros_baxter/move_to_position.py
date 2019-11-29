import sys
import rospy
import baxter_interface
import ik_solver
from baxter_interface import CHECK_VERSION
from geometry_msgs.msg import (
  Point,
  Quaternion,
)

def try_float(x):
  try:
    return float(x)
  except ValueError:
    return None

def point2angles(line, orient):
  loc = Point(line[0], line[1], line[2])
  orient = Quaternion(orient[0], orient[1], orient[2], orient[3])

  limb_joints = ik_solver.ik_solve('left', loc, orient)

  return limb_joints, loc

def move_to(_line, _orient):
  left = baxter_interface.Limb('left')

  line = _line
  orient = _orient

  lcmd, loc = point2angles(line, orient)
  left.move_to_joint_positions(lcmd)