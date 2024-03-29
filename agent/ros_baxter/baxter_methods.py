# Importing Baxter stuff:
#!/usr/bin/env python
import argparse
import random
import sys

##
#sys.path.append("/home/baxter/ros_ws_sim/src/baxter_pykdl")
sys.path.append("/home/baxter/ros_ws/src/baxter_pykdl")

#sys.path.append("/home/baxter/berkeley/gps-ksu/python/gps/agent/ros_baxter/baxter_pykdl/src/baxter_pykdl")
#sys.path.append("/home/baxter/ros_ws/src/baxter_pykdl/src/baxter_pykdl")
##

import rospy
import roslib; roslib.load_manifest('gps_agent_pkg')

import baxter_interface
import baxter_external_devices

from baxter_interface import CHECK_VERSION
from baxter_pykdl import baxter_kinematics
#from baxter_pykdl.src.baxter_pykdl import baxter_kinematics

##
import cv2
import cv_bridge
from sensor_msgs.msg import Image
import numpy as np
import matplotlib.pyplot as plt

from geometry_msgs.msg import (
    Point,
    Quaternion,
)
import ik_solver

import pprint
import time
##


# Proposed joint name order of joint commands coming from the policy... not sure if this matters.
#baxter_joint_name_list = ['right_e0','right_s0','right_s1','right_w0','right_e1','right_w1','right_w2']
baxter_joint_name_list = ['right_s0', 'right_s1', 'right_e0', 'right_e1', 'right_w0', 'right_w1', 'right_w2']

class BaxterMethods:

    def __init__(self):
        self.camera_image = None
        #self._setup_baxter_world()
        ##
        #self.left_limb = baxter_interface.Limb('left')
        ##
        #self.limb = baxter_interface.Limb('right')
        #self.kin = baxter_kinematics('right')
        
    def _setup_baxter_world(self):
        print("Initializing node... ")
        #rospy.init_node("rsdk_joint_position_keyboard")
        rospy.init_node("baxter_world_node")
        print("Getting robot state... ")
        rs = baxter_interface.RobotEnable(CHECK_VERSION)
        init_state = rs.state().enabled

        def clean_shutdown():
            print("\nExiting example...")
            if not init_state:
                print("Disabling robot...")
                rs.disable()
        rospy.on_shutdown(clean_shutdown)

        print("Enabling robot... ")
        rs.enable()
        
        self.limb = baxter_interface.Limb('right')
        self.kin = baxter_kinematics('right')
        # map_keyboard()
        print("Done.")

    def set_baxter_joint_angles(self, joint_angles_list):
        if len(joint_angles_list) != 7:
            print "The number of joint angles passed to baxter are: " + str(len(joint_angles_list))
        #self.limb.set_joint_positions(baxter_list_to_dict(joint_angles_list),True)
	    #self.limb.move_to_joint_positions(joint_dict)
        print "joint_angles_list: ", baxter_list_to_dict(joint_angles_list)
        self.limb.move_to_joint_positions(baxter_list_to_dict(joint_angles_list), True)

    def set_baxter_joint_velocities(self, joint_angles_list):
        if len(joint_angles_list) != 7:
            print "The number of joint angles passed to baxter are: " + str(len(joint_angles_list))
        self.limb.set_joint_velocities(baxter_list_to_dict(joint_angles_list))

    def set_baxter_joint_positions(self, joint_angles_list):
        joint_dict = baxter_list_to_dict(joint_angles_list)
        self.limb.move_to_joint_positions(joint_dict)

    ##
    def move_baxter_to_joint_positions(self, joint_angles_list):
        if len(joint_angles_list) != 7:
            print "The number of joint angles passed to baxter are: " + str(len(joint_angles_list))
        self.limb.move_to_joint_positions(baxter_list_to_dict(joint_angles_list))    
    
    def set_baxter_joint_torques(self, torque):
        if len(torque) != 7:
            print "The number of joint torques passed to baxter are: " + str(len(torque))
        self.limb.set_joint_torques(baxter_list_to_dict(torque))
    ##

    def get_baxter_joint_angles_positions(self):
        observed_joint_angles_dict = self.limb.joint_angles()
        if len(observed_joint_angles_dict) != 7:
            print "The number of joint angles taken from baxter are: " + str(len(observed_joint_angles_dict))
        return baxter_dict_to_list(observed_joint_angles_dict)

    def get_baxter_joint_angles_velocities(self):
        observed_joint_velocities_dict = self.limb.joint_velocities()
        if len(observed_joint_velocities_dict) != 7:
            print "The number of joint angles taken from baxter are: " + str(len(observed_joint_velocities_dict))
        return baxter_dict_to_list(observed_joint_velocities_dict)

    def get_baxter_end_effector_pose(self):
        pose = self.limb.endpoint_pose()
        # return list(pose['position'])  + [0] * 3
        return list(pose['position'])

    def get_baxter_end_effector_velocity(self):
        pose = self.limb.endpoint_velocity()
        return list(pose['linear']) + list(pose['angular'])
        
    def get_baxter_end_effector_jacobian(self):
        return self.kin.jacobian()

    def _point2angles(self, line, orient):
        loc = Point(line[0], line[1], line[2])
        orient = Quaternion(orient[0], orient[1], orient[2], orient[3])

        limb_joints = ik_solver.ik_solve('left', loc, orient)

        return limb_joints, loc

    def _move_to(self, _line, _orient):
        lcmd, loc = self._point2angles(_line, _orient)
        print "lcmd: ", lcmd
        baxter_interface.Limb('left').move_to_joint_positions(lcmd)

    def get_baxter_camera_open(self):
        #baxter_interface.CameraController('head_camera').close()

        self.camera = baxter_interface.CameraController('head_camera')
        self.camera_image = None

        #self.camera.resolution = (320, 200)
        #self.camera.resolution = (960, 600)
        self.camera.resolution = (1280, 800)
        self.camera.fps = 120
        self.camera.open()

        #line = (0.73, 0.24, 0.23)  # for block_inserting task
        #orient = (0.05774884760470479, -0.3183705294930656, 0.937935098433174, 0.12483199781220328)    # for block_inserting task
        
        #line = (0.7672786658636204, 0.18026048679864173, 0.3130814576626296)    # for block_inserting_task_2
        #orient = (-0.007446286050295908, -0.4204327887615895, 0.8989695281164152, 0.12261570240535954)

        #line = (0.62, 0.42, 0.21)  # for ball_punching task
        #orient = (-0.2998637157505639, 0.850747219922869, -0.4240778662278425, -0.08042936743139853)    # for ball_punching task

        #line = (0.47, 0.75, 0.27)   # for waypoints reaching task
        #orient = (-0.38, 0.92, -0.02, -0.03)

        #self._move_to(line, orient) # this will be done at initialize_left_arm()

    def initialize_left_arm(self, initial_left_arm):
        '''
        line = (0.7672786658636204, 0.18026048679864173, 0.3130814576626296)    # for block_inserting_task_2
        orient = (-0.007446286050295908, -0.4204327887615895, 0.8989695281164152, 0.12261570240535954)
        '''

        '''
        line = tuple(initial_left_arm[0:3])
        orient = tuple(initial_left_arm[3:7])

        print "line: ", line
        print "orient", orient

        self._move_to(line, orient)
        '''
        left_joint_list = ['left_s0', 'left_s1', 'left_e0', 'left_e1', 'left_w0', 'left_w1', 'left_w2']
        
        angles = {}
        for i in range(len(initial_left_arm)):
            angles[left_joint_list[i]] = initial_left_arm[i]

        baxter_interface.Limb('left').move_to_joint_positions(angles)

    def get_baxter_camera_close(self):
        self.camera.close()

    # def get_baxter_camera_image(self):
    #     self.camera_image = None
    #     self.camera_subscriber = rospy.Subscriber('cameras/left_hand_camera/image', Image, self._get_img)
    #     while type(self.camera_image) == type(None):
    #         print "get_image..."
    #         continue
    #     #self.camera_image = np.array(self.camera_image, dtype=np.float32)

    #     pprint.pprint(self.camera_image)
    #     im = plt.imshow(self.camera_image)
    #     self.camera_subscriber.unregister()
    #     print "camera_image.shape: ", self.camera_image.shape

    #     return self.camera_image

    def get_baxter_camera_image(self):
        self.camera_image = None
        self.camera_subscriber = rospy.Subscriber('cameras/head_camera/image', Image, self._get_img)
        while type(self.camera_image) == type(None):
            continue
        #self.camera_image = np.array(self.camera_image, dtype=np.float32)

        #pprint.pprint(self.camera_image)
        self.camera_subscriber.unregister()

        #self.camera_image = self.camera_image[300:600, 230:530]    # for block_inserting task
        #self.camera_image = self.camera_image[300:600, 430:730]
        #self.camera_iamge = self.camera_image[300:600, 400:700]
        #self.camera_image = self.camera_image[300:600, 440:740]    # for ball_punching task
        self.camera_image = self.camera_image[500:800, 490:790]    # for grasping task

        alpha = 3.0     # parameter for modifying contrast
        beta = 25.0      # parameter for modifying brightness
        #self.camera_image = np.clip(alpha*self.camera_image + beta, 0, 255)
        self.camera_image = cv2.convertScaleAbs(self.camera_image, alpha=alpha, beta=beta)
        
        return self.camera_image

    def _get_img(self, msg):
        self.camera_image = cv_bridge.CvBridge().imgmsg_to_cv2(msg, desired_encoding='rgb8')

    def initialize_baxter_gripper(self):
        self.gripper = baxter_interface.Gripper('right')

    def baxter_calibrate_gripper(self):
        self.gripper.calibrate()

    def baxter_open_gripper(self):
        self.gripper.open()

    def baxter_close_gripper(self):
        self.gripper.close()

    def set_baxter_joint_speed(self, speed):
        self.limb.set_joint_position_speed(speed)

def baxter_dict_to_list(dictionary):
    joint_list = []
    for i in range(len(baxter_joint_name_list)):
        joint_list.append(dictionary[baxter_joint_name_list[i]])
    return joint_list

def baxter_list_to_dict(joint_list):
    joint_dict = {}
    for i in range(len(joint_list)):
        joint_dict[baxter_joint_name_list[i]] = joint_list[i]
    return joint_dict
