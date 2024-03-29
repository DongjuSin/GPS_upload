""" This file defines an agent for the MuJoCo simulator environment. """
import copy

import numpy as np
from matplotlib import pyplot as plt

# import mjcpy

from gps.agent.agent import Agent
from gps.agent.agent_utils import generate_noise, setup
from gps.agent.config import AGENT_BAXTER
from gps.proto.gps_pb2 import JOINT_ANGLES, JOINT_VELOCITIES, \
        END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES, \
        END_EFFECTOR_POINT_JACOBIANS, ACTION, RGB_IMAGE, RGB_IMAGE_SIZE, \
        CONTEXT_IMAGE, CONTEXT_IMAGE_SIZE, IMAGE_FEAT, NOISE

from gps.sample.sample import Sample

import baxter_methods
import time
import datetime

# to check directory to save data is already exists
import os

class AgentBaxter(Agent):
    """
    All communication between the algorithms and MuJoCo is done through
    this class.
    """
    def __init__(self, hyperparams):
        config = copy.deepcopy(AGENT_BAXTER)
        config.update(hyperparams)
        Agent.__init__(self, config)
        self._setup_conditions()
        self.set_initial_state()
        # self._setup_world(hyperparams['filename'])

        self.baxter = baxter_methods.BaxterMethods()
        if RGB_IMAGE in self.obs_data_types:
            self.baxter.use_camera = True
        self.baxter._setup_baxter_world()
        #self.baxter._setup_baxter_camera()
        #self.set_initial_state()

    def set_initial_state(self):
        self.x0 = []
        for i in range(self._hyperparams['conditions']):
            if END_EFFECTOR_POINTS in self.x_data_types:
                '''
                eepts = np.array(self.baxter.get_baxter_end_effector_pose()).flatten()
                self.x0.append(
                    np.concatenate([self._hyperparams['x0'][i], eepts, np.zeros_like(eepts)])
                )
                '''
                self.x0.append(self._hyperparams['x0'][i])
            else:
                self.x0.append(self._hyperparams['x0'][i])
            if IMAGE_FEAT in self.x_data_types:
                self.x0[i] = np.concatenate([self.x0[i], np.zeros((self._hyperparams['sensor_dims'][IMAGE_FEAT],))])
        
    '''
    def set_initial_state(self):
        self.x0 = []
        conditions = self._hyperparams['conditions']
        print "condition:", conditions
        for i in range(conditions):
            self._hyperparams['x0'] = setup(self._hyperparams['x0'], i+1)
        self.x0 = self._hyperparams['x0']
        
        print "x0:"
        print self.x0
        exit()
    '''

    def _setup_conditions(self):
        """
        Helper method for setting some hyperparameters that may vary by
        condition.
        """
        conds = self._hyperparams['conditions']
        for field in ('x0', 'x0var', 'pos_body_idx', 'pos_body_offset',
                      'noisy_body_idx', 'noisy_body_var', 'filename'):
            self._hyperparams[field] = setup(self._hyperparams[field], conds)

    ## NOT CALLED <-- REPLACED TO self.baxter._setup_baxter_world()
    def _setup_world(self, filename):
        """
        Helper method for handling setup of the MuJoCo world.
        Args:
            filename: Path to XML file containing the world information.
        """
        self._world = []
        self._model = []

        # Initialize Mujoco worlds. If there's only one xml file, create a single world object,
        # otherwise create a different world for each condition.
        if not isinstance(filename, list):
            self._world = mjcpy.MJCWorld(filename)
            # This holds the xml model 
            self._model = self._world.get_model()
            self._world = [self._world
                           for _ in range(self._hyperparams['conditions'])]
            self._model = [copy.deepcopy(self._model)
                           for _ in range(self._hyperparams['conditions'])]
        else:
            for i in range(self._hyperparams['conditions']):
                self._world.append(mjcpy.MJCWorld(self._hyperparams['filename'][i]))
                self._model.append(self._world[i].get_model())

        for i in range(self._hyperparams['conditions']):
            for j in range(len(self._hyperparams['pos_body_idx'][i])):
                idx = self._hyperparams['pos_body_idx'][i][j]
                self._model[i]['body_pos'][idx, :] += \
                        self._hyperparams['pos_body_offset'][i]
            self._world[i].set_model(self._model[i])
            x0 = self._hyperparams['x0'][i]
            idx = len(x0) // 2
            data = {'qpos': x0[:idx], 'qvel': x0[idx:]}
            self._world[i].set_data(data)
            self._world[i].kinematics()

        self._joint_idx = list(range(self._model[0]['nq']))
        self._vel_idx = [i + self._model[0]['nq'] for i in self._joint_idx]

        # Initialize x0.
        self.x0 = []
        for i in range(self._hyperparams['conditions']):
            if END_EFFECTOR_POINTS in self.x_data_types:
                eepts = self._world[i].get_data()['site_xpos'].flatten()
                self.x0.append(
                    np.concatenate([self._hyperparams['x0'][i], eepts, np.zeros_like(eepts)])
                )
            else:
                self.x0.append(self._hyperparams['x0'][i])

        cam_pos = self._hyperparams['camera_pos']
        for i in range(self._hyperparams['conditions']):
            self._world[i].init_viewer(AGENT_BAXTER['image_width'],
                                       AGENT_BAXTER['image_height'],
                                       cam_pos[0], cam_pos[1], cam_pos[2],
                                       cam_pos[3], cam_pos[4], cam_pos[5])

    def sample(self, itr, policy, condition, verbose=True, save=True, noisy=True):
        """
        Runs a trial and constructs a new sample containing information
        about the trial.
        Args:
            itr : to name data file with iteration number, can erase when it is not neccessary
            policy: Policy to be used in the trial.
            condition: Which condition setup to run.
            verbose: Whether or not to plot the trial.
            save: Whether or not to store the trial into the samples.
        """
        img = []
        fp = []
        obs = []

        # Create new sample, populate first time step.
        #self._init_tf(policy.dU)
        feature_fn = None
        if 'get_features' in dir(policy):
            feature_fn = policy.get_features

        mj_X = self._hyperparams['x0'][condition]
        U = np.zeros([self.T, self.dU])
        
        if noisy:
            noise = generate_noise(self.T, self.dU, self._hyperparams)
        else:
            noise = np.zeros((self.T, self.dU))

        # Not called
        if np.any(self._hyperparams['x0var'][condition] > 0):
            x0n = self._hyperparams['x0var'] * \
                    np.random.randn(self._hyperparams['x0var'].shape)
            mj_X += x0n
        noisy_body_idx = self._hyperparams['noisy_body_idx'][condition]

        # Not called
        if noisy_body_idx.size > 0:
            for i in range(len(noisy_body_idx)):
                idx = noisy_body_idx[i]
                var = self._hyperparams['noisy_body_var'][condition][i]

                self._model[condition]['body_pos'][idx, :] += \
                        var * np.random.randn(1, 3)


        # self._world[condition].set_model(self._model[condition])

        ## INIT BAXTER
        #self.baxter.move_baxter_to_joint_positions([0.32, -0.71, 0.68, 1.09, 0.07, 0.76, 0.13])   # for ball_punching task
        #self.baxter.move_baxter_to_joint_positions([0.27, -1.14, 0.98, 1.60, 0.15, 0.51, 0.27])
        
        self.baxter.move_baxter_to_joint_positions(self._hyperparams['x0'][condition][0:7])

        new_sample = self._init_sample(condition, feature_fn=feature_fn)   # new_sample: class 'Sample'

        for t in range(self.T):
        # for t in range(12):
            X_t = new_sample.get_X(t=t)
            obs_t = new_sample.get_obs(t=t)
            print obs_t.shape
            obs.append(obs_t)

            # set the ACTION for the bot gotten from the policy calculations, and apply.
            #mj_U = policy.act(X_t, obs_t, t, noise[t, :])
            mj_U = policy.act(X_t, obs_t, t, noise[t, :], condition)
            U[t, :] = mj_U

            # if verbose:
            #     self._world[condition].plot(mj_X)

            # every step but the last
            if (t + 1) < self.T:
                for _ in range(self._hyperparams['substeps']):

                    # This is the call to mjcpy to set the robot
                    # mj_X, _ = self._world[condition].step(mj_X, mj_U)

                    # Set the baxter joint velocities through the Baxter API
                    self.baxter.set_baxter_joint_velocities(mj_U)
                    #self.baxter.set_baxter_joint_positions(mj_U)
                    #self.baxter.set_baxter_joint_torques(mj_U)

                    # mj_X[self._joint_idx] = self.baxter.get_baxter_joint_angles_positions()
                    # mj_X[self._vel_idx] = self.baxter.get_baxter_joint_angles_velocities()

                    # mj_X = self.baxter.get_baxter_joint_angles()

                #TODO: Some hidden state stuff will go here.
                # self._data = self._world[condition].get_data()
                #time.sleep(1)
                print "current step(t): ", t
                self._set_sample(new_sample, mj_X, t, condition, feature_fn=feature_fn)
                if t == 0:
                    raw_input('first time step end')

            fp_t = new_sample.get(IMAGE_FEAT, t)
            # img_t = self._get_image_from_obs(obs_t)
            img_t = new_sample.get(RGB_IMAGE,t)
            # path = '/hdd/gps-master/experiments/test_obs/data_files/check_obs/' + 'img_%d' % t
            # np.save(path, img_t)
            fp.append(fp_t)
            img.append(img_t)
        fp = np.asarray(fp)
        img = np.asarray(img)
        obs = np.asarray(obs)

        ## dongju : to save feature points and image observed
        path = '/hdd/gps-master/experiments/' + 'block_insert_new' + '/data_files/check_fp'
        if not os.path.exists(path):
            os.mkdir(path)
            print path, ' is created'
        fname = path + '/fp_%d_%d.npz' % (itr, condition)
        np.savez_compressed(fname, fp = fp, img = img, obs = obs)

        new_sample.set(ACTION, U)
        new_sample.set(NOISE, noise)

        if save:
            self._samples[condition].append(new_sample)
        return new_sample

    def _init_sample(self, condition, feature_fn=None):
        """
        Construct a new sample and fill in the first time step.
        Args:
            condition: Which condition to initialize.
        """
        sample = Sample(self)
        ## modified
        #self.baxter.move_baxter_to_joint_positions([1.05, -0.01, 0.20, 0.50, 0.47, 0.80, -0.14])
        #self.baxter.move_baxter_to_joint_positions([0.27, -1.14, 0.98, 1.60, 0.15, 0.51, 0.27])    # for block_inserting task
        #self.baxter.move_baxter_to_joint_positions([0.32, -0.71, 0.68, 1.09, 0.07, 0.76, 0.13])   # for ball_punching task
        #self.baxter.move_baxter_to_joint_positions(self._hyperparams['x0'][condition][0:7])
        #self.baxter.initialize_left_arm([-0.22549517556152346, 0.36815538867187503, -1.5040681608032227, 0.5817622131408692, -0.5012282218688965, 1.8553497608276368, 0.08935438079223633]) # for block_inserting task
        
        self.baxter.initialize_left_arm(self._hyperparams['initial_left_arm'][condition]) # grasping task

        self.cnt = 0

        self.prev_positions = self.baxter.get_baxter_joint_angles_positions()
        sample.set(JOINT_ANGLES, np.array(self.prev_positions), t=0)
        sample.set(JOINT_VELOCITIES, np.array(self.baxter.get_baxter_joint_angles_velocities()), t=0)
        sample.set(END_EFFECTOR_POINTS, np.array(self.baxter.get_baxter_end_effector_pose()), t=0)
        sample.set(END_EFFECTOR_POINT_VELOCITIES, np.array(self.baxter.get_baxter_end_effector_velocity()), t=0)
        sample.set(END_EFFECTOR_POINT_JACOBIANS, np.array(self.baxter.get_baxter_end_effector_jacobian()), t=0)

        ## NEED TO ADD SENSOR 'RGB_IMAGE'
        ## NEED TO ADD 'get_baxter_camera_image()' in 'baxter_methods.py'
        if RGB_IMAGE in self.obs_data_types:
            #self.baxter.get_baxter_camera_open()
            self.img = self.baxter.get_baxter_camera_image()
            # np.savez('camera_image_' + str(condition) + '.npz', img=self.img)
            ## NEED TO CHECK IMAGE SHAPE
            ## NEED TO CHECK IMAGE TYPE - INT? / FLOAT?
            ## MUJOCO: [HEIGHT, WIDTH, CHANNELS] == [300, 480, 3]
            sample.set(RGB_IMAGE, np.transpose(self.img, (2, 1, 0)).flatten(), t = 0)
            sample.set(RGB_IMAGE_SIZE, [self._hyperparams['image_channels'],
                                        self._hyperparams['image_width'],
                                        self._hyperparams['image_height']], t=None)
            if IMAGE_FEAT in self.obs_data_types:
                raise ValueError('Image features should not be in observation, just state')
            if feature_fn is not None:
                obs = sample.get_obs()
                sample.set(IMAGE_FEAT, feature_fn(obs), t=0)
            else:
                sample.set(IMAGE_FEAT, np.zeros((self._hyperparams['sensor_dims'][IMAGE_FEAT],)), t=0)

        return sample

    def _init_tf(self, dU):
        self.dU = dU

    def _set_sample(self, sample, mj_X, t, condition, feature_fn=None):
        """
        Set the data for a sample for one time step.
        Args:
            sample: Sample object to set data for.
            mj_X: Data to set for sample.
            t: Time step to set for sample.
            condition: Which condition to set.
        """
        # print 'setting sample in timestep: ' + str(t) + 'and using joints of: ' + str(np.array(mj_X[self._joint_idx]))

        # Baxter setting joint angles and velocities

        ## modified
        curr_positions = self.baxter.get_baxter_joint_angles_positions()
        sample.set(JOINT_ANGLES, np.array(curr_positions), t=t+1)
        sample.set(JOINT_VELOCITIES, np.array(self.baxter.get_baxter_joint_angles_velocities()), t=t+1)
        sample.set(END_EFFECTOR_POINTS, np.array(self.baxter.get_baxter_end_effector_pose()), t=t+1)
        sample.set(END_EFFECTOR_POINT_VELOCITIES, np.array(self.baxter.get_baxter_end_effector_velocity()), t=t+1)
        sample.set(END_EFFECTOR_POINT_JACOBIANS, np.array(self.baxter.get_baxter_end_effector_jacobian()), t=t+1)
        time_out = True
        s0 = time.time()
        while (self.prev_positions == curr_positions):
            s1 = time.time()
            if s1-s0 >= 1.0:
                break
        '''
        #while(self.prev_positions != curr_positions and time_out):
        #while(self.prev_positions == curr_positions and time_out):
            
            curr_positions = self.baxter.get_baxter_joint_angles_positions()
            sample.set(JOINT_ANGLES, np.array(self.baxter.get_baxter_joint_angles_positions()), t=t)
            sample.set(JOINT_VELOCITIES, np.array(self.baxter.get_baxter_joint_angles_velocities()), t=t)
            sample.set(END_EFFECTOR_POINTS, np.array(self.baxter.get_baxter_end_effector_pose()), t=t)
            sample.set(END_EFFECTOR_POINT_VELOCITIES, np.array(self.baxter.get_baxter_end_effector_velocity()), t=t)
            sample.set(END_EFFECTOR_POINT_JACOBIANS, np.array(self.baxter.get_baxter_end_effector_jacobian()), t=t)
            s1 = time.time()
            if s1-s0 >= 0.5 :
                time_out = False
                #self.cnt = 0
        '''
        #self.cnt +=1
        #print ("Timeout count: "+str(self.cnt))
        self.prev_positions = curr_positions
        print('Joint Positions: ' + repr(self.prev_positions) + '\n')
        
        '''
        #f = open("policy without giving way points.txt", "a")
        #f = open("block_inserting_policy.txt", "a")
        if t == 18:
            f = open('joint_position.txt', 'a')
            f.write('Joint Positions: ' + str(self.prev_positions) + '\n')
            f.write('Timestep (t): ' + str(t) + '\n\n')
            f.close()
        '''

        ## NEED TO ADD SENSOR 'RGB_IMAGE'
        if RGB_IMAGE in self.obs_data_types:
            #self.baxter.get_baxter_camera_open()
            self.img = self.baxter.get_baxter_camera_image()
            image = self.img
            path = '/hdd/gps-master/experiments/'+'block_insert_new'+'/data_files/check_obs/' + 'img_%d' % t
            np.save(path, image)
            ## NEED TO CHECK IMAGE SHAPE
            ## MUJOCO: [HEIGHT, WIDTH, CHANNELS] == [300, 480, 3]
            ## transpose((1, 0, 2))? / transpose((2, 1, 0))?
            sample.set(RGB_IMAGE, np.transpose(self.img, (2, 1, 0)).flatten(), t = t+1)
            sample.set(RGB_IMAGE_SIZE, [self._hyperparams['image_channels'],
                                        self._hyperparams['image_width'],
                                        self._hyperparams['image_height']], t=None)
            if feature_fn is not None:
                obs = sample.get_obs()
                sample.set(IMAGE_FEAT, feature_fn(obs), t=t+1)
            else:
                sample.set(IMAGE_FEAT, np.zeros((self._hyperparams['sensor_dims'][IMAGE_FEAT],)), t=t+1)

    def _get_image_from_obs(self, obs):
        imstart = 0
        imend = 0
        image_channels = self._hyperparams['image_channels']
        image_width = self._hyperparams['image_width']
        image_height = self._hyperparams['image_height']
        for sensor in self._hyperparams['obs_include']:
            # Assumes only one of RGB_IMAGE or CONTEXT_IMAGE is present
            if sensor == RGB_IMAGE or sensor == CONTEXT_IMAGE:
                imend = imstart + self._hyperparams['sensor_dims'][sensor]
                break
            else:
                imstart += self._hyperparams['sensor_dims'][sensor]
        print obs.shape
        img = obs[imstart:imend]
        img = img.reshape((image_width, image_height, image_channels))
        img = img.astype(np.uint8)
        return img

    #def setup_camera(self):
    #    self.baxter._setup_baxter_camera()
    def open_camera(self):
        self.baxter.get_baxter_camera_open()
    
    def close_camera(self):
        self.baxter.get_baxter_camera_close()

    def init_gripper(self):
        self.baxter.initialize_baxter_gripper()
    
    def calibrate_gripper(self):
        self.baxter.baxter_calibrate_gripper()
    
    def open_gripper(self):
        self.baxter.baxter_open_gripper()
    
    def close_gripper(self):
        self.baxter.baxter_close_gripper()

    def move_to_position(self, joint_angles_list):
        self.baxter.move_baxter_to_joint_positions(joint_angles_list)
    
    def set_speed(self, speed=0.3):
        self.baxter.set_baxter_joint_speed(speed)

    def move_left_to_position(self, joint_angles_list):
        self.baxter.initialize_left_arm(joint_angles_list)