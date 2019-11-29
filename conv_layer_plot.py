import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import copy

import sys
from gps.algorithm.policy_opt.config import POLICY_OPT_TF
from gps.algorithm.policy_opt.tf_model_example import multi_modal_network_fp, multi_modal_network
#from python.gps.algorithm.algorithm_badmm import AlgorithmBADMM
from gps.algorithm.policy_opt.policy_opt_tf import PolicyOptTf
from gps.algorithm.policy_opt.tf_utils import TfSolver
#from python.gps.algorithm.policy.policy_prior_gmm import PolicyPriorGMM

import baxter_interface
import rospy
import cv_bridge
import cv2
from sensor_msgs.msg import Image

hyperparams={'network_params': {'image_height': 300, 'obs_include': [1, 2, 11], 'obs_image_data': [11], \
								'num_filters': [64, 32, 32], 'obs_vertor_data': [1, 2], \
								'sensor_dims': {0: 7, 1: 7, 2: 7, 3: 6, 4: 6, 16: 64, 11: 270000, 13: 3},\
								'image_channels': 3, 'image_width': 300}, \
			'random_seed': 1, 'use_gpu': 1, 'gpu_id': 0, 'fc_only_iterations': 0, 'batch_size': 25, \
			'network_model':multi_modal_network, 'ent_reg': 0.0, 'copy_param_scope': 'conv_params',\
			'checkpoint_prefix':'/home/baxter/berkeley/gps-master/python/gps/algorithm/policy_opt/tf_checkpoint/policy_checkpoint_', 
			'iterations': 1 ,'type': PolicyOptTf, 'lr': 0.001, 'lr_policy': 'fixed', 'weight_decay': 0.005, \
			'init_var': 0.1, 'weights_file_prefix': 'python/../experiments/baxter_block_inserting/policy', \
			'momentum': 0.9, 'solver_type': 'Adam'}

'''
hyperparams={'random_seed': 1,
             'network_params': {'image_height': 300, 'obs_include': [1, 2, 11], 'obs_image_data': [11], \
                                'num_filters': [5, 10], 'obs_vertor_data': [1, 2], \
                                'sensor_dims': {0: 7, 1: 7, 2: 7, 3: 6, 4: 6, 11: 270000, 13: 3}, \
                                'image_channels': 3, 'image_width': 300}, \
            'use_gpu': 1, 'gpu_id': 0, \
            'batch_size': 25, 'network_model': multi_modal_network, 'ent_reg': 0.0,'lr': 0.001, \
            'checkpoint_prefix': '/home/baxter/berkeley/gps-ksu/python/gps/algorithm/policy_opt/tf_checkpoint/policy_checkpoint.ckpt',\
            'iterations': 3000, 'type': PolicyOptTf, \
            'weights_file_prefix': 'python/../experiments/baxter_block_inserting/policy', \
            'lr_policy': 'fixed', 'weight_decay': 0.005, 'init_var': 0.1, 'momentum': 0.9, 'solver_type': 'Adam'}
'''
save_dir='./plot_layer/'

class PlotFeatures():
    def __init__(self):
        self._hyperparams = hyperparams
        tf.set_random_seed(self._hyperparams['random_seed'])

        #self.tf_iter = 0
        self.checkpoint_file = self._hyperparams['checkpoint_prefix']
        self.batch_size = self._hyperparams['batch_size']
        self.device_string = "/cpu:0"
        if self._hyperparams['use_gpu'] == 1:
            self.gpu_device = self._hyperparams['gpu_id']
            self.device_string = "/gpu:" + str(self.gpu_device)
        self.act_op = None  # mu_hat
        self.feat_op = None # features
        self.loss_scalar = None
        self.obs_tensor = None
        self.precision_tensor = None
        self.action_tensor = None  # mu true
        self.solver = None
        self.feat_vals = None
##
        self.conv_layer_0 = None
        self.conv_layer_1 = None
        self.conv_layer_2 = None
        self.main_itr = None
##
        self.init_network()
        self.init_solver()
        self.var = self._hyperparams['init_var'] * np.ones(self._dU)
        self.sess = tf.Session()
        #'''
        self.x_idx, self.img_idx, i = [], [], 0
        if 'obs_image_data' not in self._hyperparams['network_params']:
            self._hyperparams['network_params'].update({'obs_image_data': []})
        for sensor in self._hyperparams['network_params']['obs_include']:
            dim = self._hyperparams['network_params']['sensor_dims'][sensor]
            if sensor in self._hyperparams['network_params']['obs_image_data']:
                self.img_idx = self.img_idx + list(range(i, i+dim))
            else:
                self.x_idx = self.x_idx + list(range(i, i+dim))
                i += dim
        #'''
        #print self.x_idx, self.img_idx, i
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        self.baxter_camera_open()



    def init_network(self):
        """ Helper method to initialize the tf networks used """
        tf_map_generator = self._hyperparams['network_model']
        self._dO = 270014
        self._dU = 7
        tf_map, fc_vars, last_conv_vars = tf_map_generator(dim_input=self._dO, dim_output=self._dU, batch_size=self.batch_size,\
                                  network_config=self._hyperparams['network_params'])
        self.obs_tensor = tf_map.get_input_tensor()
        self.precision_tensor = tf_map.get_precision_tensor()
        self.action_tensor = tf_map.get_target_output_tensor()
        self.act_op = tf_map.get_output_op()
        self.feat_op = tf_map.get_feature_op()
        self.loss_scalar = tf_map.get_loss_op()
        self.fc_vars = fc_vars
        self.last_conv_vars = last_conv_vars
        ##
        self.conv_layer_0 = tf_map.get_conv_layer_0()
        self.conv_layer_1 = tf_map.get_conv_layer_1()
        self.conv_layer_2 = tf_map.get_conv_layer_2()
        ##

        # Setup the gradients
        self.grads = [tf.gradients(self.act_op[:,u], self.obs_tensor)[0]
                for u in range(self._dU)]

    def init_solver(self):
        """ Helper method to initialize the solver. """
        self.solver = TfSolver(loss_scalar=self.loss_scalar,
                               solver_name=self._hyperparams['solver_type'],
                               base_lr=self._hyperparams['lr'],
                               lr_policy=self._hyperparams['lr_policy'],
                               momentum=self._hyperparams['momentum'],
                               weight_decay=self._hyperparams['weight_decay'],
                               fc_vars=self.fc_vars,
                               last_conv_vars=self.last_conv_vars)
        self.saver = tf.train.Saver()

    def load_network(self,itr):
        self.itr = itr
        self.fname = self.checkpoint_file + str(itr) + '.ckpt'
        self.saver.restore(self.sess,self.fname)
        print 'Restoring model from: %s', self.fname

    def baxter_camera_open(self):
        self.camera=baxter_interface.CameraController('head_camera')
        self.camera_image = None

        self.camera.resolution = (960, 600)
        self.camera.fps = 120
        self.camera.open()

    def baxter_camera_close(self):
        self.camera.close()

    def get_baxter_camera_image(self):
        
        self.camera_image = None
        self.camera_subscriber = rospy.Subscriber('cameras/head_camera/image', Image, self._get_img)

        while type(self.camera_image) == type(None):
            continue
        self.camera_subscriber.unregister()
        self.camera_image = self.camera_image[300:600, 230:530]
        return self.camera_image

    def _get_img(self, msg):
        self.camera_image = cv_bridge.CvBridge().imgmsg_to_cv2(msg, desired_encoding='rgb8')

    def get_input(self,):
        # Todo: 1. get camera img
        #       2. crop img = obs
        obs = self.get_baxter_camera_image()
        return obs
        '''
        if RGB_IMAGE in self.obs_data_types:
        #self.baxter.get_baxter_camera_open()
            self.img = self.baxter.get_baxter_camera_image()
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
        '''
    def get_features(self,obs):
        if len(obs.shape) == 1:
            obs = np.expand_dims(obs, axis=0)
        # Assume that features don't depend on the robot config, so don't normalize by scale and bias.
        with tf.device(self.device_string):
            feat = self.sess.run(self.feat_op, feed_dict={self.obs_tensor: obs})
        return feat[0]

    def forward(self,obs):

        #obs=[]
        #for i in range(19):
            #img_=copy(img)
        #    obs.append(img)

        if len(obs.shape) == 1:
            obs = np.expand_dims(obs, axis=0)

        self.scale = np.diag(
                1.0 / np.maximum(np.std(obs[:, self.x_idx], axis=0), 1e-3))
        self.bias = - np.mean(
                obs[:, self.x_idx].dot(self.scale), axis=0)
        obs[:, self.x_idx] = obs[:, self.x_idx].dot(self.scale) + self.bias
                                
        with tf.device(self.device_string):
            conv_layer_0, conv_layer_1, conv_layer_2 = \
                self.sess.run([self.conv_layer_0, self.conv_layer_1, self.conv_layer_2],
                    feed_dict={self.obs_tensor: obs})
        feature = self.get_features(obs)
        #print "conv_layer_0:\n", conv_layer_0
        #print "conv_layer_1:\n", conv_layer_1
        #print "conv_layer_2:\n", conv_layer_2
        #print "feature:\n", feature
        #np.savez('conv_layer_11.npz', conv_layer_0=conv_layer_0, conv_layer_1=conv_layer_1, conv_layer_2=conv_layer_2, feature=feature)
        self.conv0 = conv_layer_0
        self.conv1 = conv_layer_1
        self.conv2 = conv_layer_2
        self.feature = feature

    def plot(self,):
        '''
        npfile = np.load('conv_layer_11.npz')
        conv_layer_0 = npfile['conv_layer_0']
        conv_layer_1 = npfile['conv_layer_1']
        conv_layer_2 = npfile['conv_layer_2']
        '''
        feature = self.feature
        #print np.shape(self.conv0[0,0])
        conv_layer_0 = np.array(self.conv0)
        conv_layer_1 = np.array(self.conv1)
        conv_layer_2 = np.array(self.conv2)

        conv_layer_0 = conv_layer_0.reshape(150, 150, 64)
        conv_layer_1 = conv_layer_1.reshape(150, 150, 32)
        conv_layer_2 = conv_layer_2.reshape(150, 150, 32)

        points = []
        for i in range(0, feature.shape[0], 2):
            #points.append(np.array([feature[i], feature[i+1]]))
            points.append(np.array([feature[i] * 150 + 75, feature[i+1] * 150 + 75]))

        # plt.figure()
        # implot = plt.imshow(conv_layer_2[:, :, 1])
        # for i in range(len(points)):
        #     plt.scatter([points[i][0]], [points[i][1]])

        fig,ax = plt.subplots(4,8)
        for j in range(32):
            
            ax[j/8,j%8].imshow(conv_layer_0[:, :, j])
            ax[j/8,j%8].axis('off')
            #plt.subplot.imshow(conv_layer_2[:, :, j])
            for i in range(len(points)):
                plt.scatter([points[i][0]], [points[i][1]])

        plt.savefig(save_dir + 'policy_checkpoint_' + str(self.itr) + '_conv0_' + str(0) + '.jpg')
        plt.close(fig)
        plt.show()


def load_npz(fname):
    npfile = np.load(fname)
    conv_layer_0 = npfile['conv_layer_0']
    conv_layer_1 = npfile['conv_layer_1']
    conv_layer_2 = npfile['conv_layer_2']

    feature = npfile['feature']


    conv_layer_0 = conv_layer_0.reshape(150, 150, 64)
    conv_layer_1 = conv_layer_1.reshape(150, 150, 32)
    conv_layer_2 = conv_layer_2.reshape(150, 150, 32)

    points = []
    for i in range(0, feature.shape[0], 2):
        #points.append(np.array([feature[i], feature[i+1]]))
        points.append(np.array([feature[i] * 150 + 75, feature[i+1] * 150 + 75]))


    #implot = plt.imshow(conv_layer_2[:, :, 0])
    #for i in range(len(points)):
    #    plt.scatter([points[i][0]], [points[i][1]])
    
    '''
    for j in range(32):
        fig=plt.figure()
        plt.subplot.imshow(conv_layer_2[:, :, j])
        for i in range(len(points)):
            plt.scatter([points[i][0]], [points[i][1]])
        plt.savefig(save_dir+fname[:-4]+'_conv2_'+str(0)+'.jpg')
        plt.close(fig)
    '''
    #fig=plt.figure()
    fig,ax = plt.subplots(4,8)
    for j in range(32):
        
        ax[j/8,j%8].imshow(conv_layer_2[:, :, j])
        ax[j/8,j%8].axis('off')
        #plt.subplot.imshow(conv_layer_2[:, :, j])
        for i in range(len(points)):
            plt.scatter([points[i][0]], [points[i][1]])
    plt.savefig(save_dir + fname[:-4] + '_conv2_' + str(0) + '.jpg')
    plt.close(fig)
    #plt.show()

if __name__=='__main__':

    #fname = 'conv_layer_14.npz'
    #load_npz(fname)
    #'''
    rospy.init_node('Plotting')
    #sample_obs = np.load('sample_obs_14.npz')['obs']
    #sample_obs=sample_obs.reshape(300,300,)
    #print sample_obs.shape

    #sample_obs = 

    pf = PlotFeatures()

    #itr_list=[5,8,9,11,14]
    itr_list = [14]
    for itr in itr_list:
        pf.load_network(itr)
    #obs=pf.get_input()
    #plt.plot(obs)

        pf.forward(sample_obs)
        pf.plot()
    #'''

'''
npfile = np.load('conv_layer_11.npz')
conv_layer_0 = npfile['conv_layer_0']
conv_layer_1 = npfile['conv_layer_1']
conv_layer_2 = npfile['conv_layer_2']

feature = npfile['feature']


conv_layer_0 = conv_layer_0.reshape(150, 150, 64)
conv_layer_1 = conv_layer_1.reshape(150, 150, 32)
conv_layer_2 = conv_layer_2.reshape(150, 150, 32)
'''
'''
plt.imshow(conv_layer_0[:, :, 1])
plt.show()

plt.imshow(conv_layer_1[:, :, 1])
plt.show()

plt.imshow(conv_layer_2[:, :, 1])
plt.show()
'''
#filters = conv_layer_0.shape[3]
#print filters
#print conv_layer_0.shape	# (1, 1, 150, 150, 64)

#raw_input('Press Enter to continue...')
#print feature.shape
'''

print feature


points = []
for i in range(0, feature.shape[0], 2):
	#points.append(np.array([feature[i], feature[i+1]]))
	points.append(np.array([feature[i] * 150 + 75, feature[i+1] * 150 + 75]))
import pprint

pprint.pprint(points)

implot = plt.imshow(conv_layer_2[:, :, 1])
for i in range(len(points)):
	plt.scatter([points[i][0]], [points[i][1]])
plt.show()

'''