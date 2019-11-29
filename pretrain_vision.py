import os
import os.path
import sys
import copy
import argparse
import math

import numpy as np
import cv2
from matplotlib import pyplot as plt

sys.path.append('/'.join(str.split(__file__, '/')[:-2]))
sys.path.append('/'.join(str.split(__file__, '/')[:-1]))

sys.path.append(os.path.abspath('/hdd/gps-master/python/gps/pre_train/'))
sys.path.append(os.path.abspath('/hdd/gps-master/python/gps/pre_train/agent'))
sys.path.append(os.path.abspath('/hdd/gps-master/python/gps/pre_train/policy_opt'))

from gps.proto.gps_pb2 import JOINT_ANGLES, JOINT_VELOCITIES, \
		END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES, ACTION, \
		TRIAL_ARM, AUXILIARY_ARM, JOINT_SPACE, \
		RGB_IMAGE, RGB_IMAGE_SIZE, IMAGE_FEAT

#from gps.pre_train.agent.agent_baxter import AgentBaxterPreTrain
#from gps.pre_train.policy_opt.policy_opt_tf import PolicyOptTf

from agent_baxter import AgentBaxterPreTrain
from policy_opt_tf import PolicyOptTf
from tf_model_example import multi_modal_network_fp
from tf_model_example import multi_modal_network_fp_vision
from gps.utility.data_logger import DataLogger

x0s = np.array([0.4866554044006348, -0.885490408795166, 0.5092816209960938, 1.8265876210876466, -0.15339807861328125, 0.4490728751403809, 0.2051699301452637,
                 0., 0., 0., 0., 0., 0., 0.,
                 0.69, -1.46, 0.10, 0., 0., 0.,
                 0., 0., 0., 0., 0., 0.])

initial_left_arm = [np.array([0.32098547949829104, -0.7090826183898926, -0.8417719563903809, 1.3502865869934082, 0.6684321275573731, 1.2225826865478517, -0.607839886505127]),
					np.array([0.32098547949829104, -0.7090826183898926, -0.8417719563903809, 1.3502865869934082, 0.6684321275573731, 1.2225826865478517, -0.607839886505127])]

#CONDITIONS = 20
CONDITIONS = 5

IMAGE_WIDTH = 300
IMAGE_HEIGHT = 300
IMAGE_CHANNELS = 3

NUM_FP = 32

SENSOR_DIMS = {
    JOINT_ANGLES: 7,
    JOINT_VELOCITIES: 7,
    END_EFFECTOR_POINTS: 6,
    END_EFFECTOR_POINT_VELOCITIES: 6,
    ACTION: 6,
    
    RGB_IMAGE: IMAGE_WIDTH * IMAGE_HEIGHT * IMAGE_CHANNELS,
    RGB_IMAGE_SIZE: IMAGE_CHANNELS,

    IMAGE_FEAT: NUM_FP * 2, # affected by num_filters set below.
}


hyperparams_common = {
	'conditions': CONDITIONS,
	'train_conditions': range(CONDITIONS),
	'test_conditions': range(CONDITIONS),
}
hyperparams_agent = {
	'type': AgentBaxterPreTrain,
	'filename': './mjc_models/pr2_arm3d.xml',
	'x0': x0s,
	'initial_left_arm': initial_left_arm,
	'dt': 2.,
	'substeps': 5,
	'conditions': hyperparams_common['conditions'],
	'pos_body_idx': np.array([1]),
	'pos_body_offset': [np.array([0, -0.2, 0]) for i in range(hyperparams_common['conditions'])],
	'T': 5, # 40
	'sensor_dims': SENSOR_DIMS,
	'state_include': [JOINT_ANGLES, JOINT_VELOCITIES, END_EFFECTOR_POINTS,
					  END_EFFECTOR_POINT_VELOCITIES, IMAGE_FEAT],
	
	# 'obs_include': [END_EFFECTOR_POINTS, RGB_IMAGE],
	'obs_include' : [RGB_IMAGE],

	'image_width': IMAGE_WIDTH,
	'image_height': IMAGE_HEIGHT,
	'image_channels': IMAGE_CHANNELS,

	'camera_pos': np.array([0.81, 0.49, 0.31] + [0., 0., 0.])   # IN FACT NOT USED
}

hyperparams_policy_opt = {
	'type': PolicyOptTf,
	'network_params': {
		'num_filters': [64, 32, 32],
		'obs_include': hyperparams_agent['obs_include'],
		'obs_image_data': [RGB_IMAGE],
		
		'image_width': IMAGE_WIDTH,
		'image_height': IMAGE_HEIGHT,
		'image_channels': IMAGE_CHANNELS,
		'sensor_dims': SENSOR_DIMS,
	},
	'network_model': multi_modal_network_fp_vision,
	'iterations': 3000,
	#'weights_file_prefix': EXP_DIR + 'policy',
}


num_samples = 1
hyperparams_config = {
	'iterations': 1,
	'common': hyperparams_common,
	'agent': hyperparams_agent,
	'num_samples': num_samples,
}

# The number of targets in `target_list` should match with the number of conditions.
# target list 1
'''
target_list = [
	np.array([0.874833646990908,-0.05300107680655488,-0.1344609069782329]),
	np.array([0.8690296573158752,-0.10037316321567798,-0.14054187924956418]),
	np.array([0.8639267139967123,-0.1459698508490421,-0.1372746782556149]),
	np.array([0.8688888453810787,-0.1974297186948519,-0.14079736559857212]),
	np.array([0.866724523840324,-0.2415489193071388,-0.14196130585411612]),
	np.array([0.7583437177882064,-0.047335182136807666,-0.13981364174352756]),
	np.array([0.7744344171820777,-0.09616857421567543,-0.13499607563534544]),
	np.array([0.7592045302030694,-0.1443541277968847,-0.1378673996429536]),
	np.array([0.7699610402284631,-0.19970652374174006,-0.13422073472496543]),
	np.array([0.7612076450029054,-0.2504906375735094,-0.137919478401654])
]
'''
valid_list = [
	# np.array([0.8618270617872065, -0.06553517893436048, -0.14537300098788317]),
	# np.array([0.8432860333186509, -0.1341647528017661, -0.1381675606181681]),
	# np.array([0.802606191709081, -0.0886639822930204, -0.139119572283541]),
	# np.array([0.827058790647241, -0.21291346963934604, -0.13788469811610118]),
	# np.array([0.7611822816782611, -0.18679669002021024, -0.13540019255001262])
	np.array([0.8193277005533536, -0.11144151837881047, -0.13416202354443418]),
	np.array([0.8170280165099629, -0.1680200771970466, -0.13713923023215283]),
	np.array([0.8206771062099422, -0.23702836622745044, -0.13171385529141766]),
	np.array([0.7354927915876904, -0.1397975894618078, -0.14125722066499377]),
	np.array([0.7598392372956585, -0.16864028051627575, -0.07233640747713906])
]

target_list = [
	np.array([0.8388315220464306,-0.0719265308486322,-0.13622976314355098]),
	np.array([0.8452931019415472,-0.11262303219547652,-0.13365003317710228]),
	np.array([0.8437173749800658,-0.153477249256813,-0.13825369282797823]),
	np.array([0.834539948054292,-0.1960940723491543,-0.14096559571208847]),
	np.array([0.839457654572932,-0.22897286391033378,-0.13870702922569167]),
	np.array([0.7509685609073389,-0.07791890642375895,-0.13722129120716803]),
	np.array([0.7487747001274405,-0.12028984169766495,-0.13905337302154383]),
	np.array([0.743088229833761,-0.1548200499738748,-0.13884694952157617]),
	np.array([0.747314285132605,-0.19311812231612613,-0.13954475139492814]),
	np.array([0.7448773346454708,-0.23601322399025867,-0.1367902387230499])
]

test_list = [
	np.array([0.8126810585938604, -0.10609574479374752, -0.13258477984144643]),
	np.array([0.8155789222180317, -0.16476497112790195, -0.13988016541614456]),
	np.array([0.819531016005123, -0.22957851133340995, -0.13708905356097373]),
	np.array([0.7389456025819094, -0.13185536544050575, -0.13674489019294123]),
	np.array([0.7393834903443697, -0.2131932462872212, -0.13978594211990444])
]

class PreTrainMain(object):
	def __init__(self):
		self._hyperparams = copy.deepcopy(hyperparams_config)
		self._conditions = self._hyperparams['common']['conditions']
		if 'train_conditions' in self._hyperparams['common']:
			self._train_idx = self._hyperparams['common']['train_conditions']
			self._test_idx = self._hyperparams['common']['test_conditions']
		else:
			self._train_idx = range(self._conditions)
			self._hyperparams['common']['train_conditions'] = self._hyperparams['common']['conditions']
			self._test_idx = self._train_idx

		self.iteration_count = 0

		self.dU = 6
		self.dO = 270006
		self.T = hyperparams_agent['T']
		self.M = len(self._train_idx)
		self.resume_training = 27

		self.agent = AgentBaxterPreTrain(hyperparams_agent)
		self.policy_opt = PolicyOptTf(hyperparams_policy_opt, self.dO, self.dU)
		self.data_logger = DataLogger()
		self.save_dir = '/hdd/gps-master/python/gps/pre_train/policy_opt/policy_opt_save/'

	def sampling(self):
		itr_start = 0
		for itr in range(itr_start, self._hyperparams['iterations']):
			for cond in self._train_idx:
				# command_txt = '\nPress Enter after changing target position of condition ' + str(cond) + ': ' + str(target_list[cond]) + '\n'
				command_txt = '\nPress Enter after changing target position of condition ' + str(cond) + ': ' + str(test_list[cond]) + '\n'
				raw_input(command_txt)
				for i in range(self._hyperparams['num_samples']):
					print 'sample: ', i
					self._take_sample(itr, cond)

			fname = './python/gps/pre_train/data/test_dataset.npz'
			self._save_data2(itr, fname)

	def training(self):
		self.sample_list = []
		
		self._load_data()
		obs = self.sample_list[0]
		tgt = self.sample_list[1]
		
		self._policy_iteration()

	def resume(self):
		itr_start = self.resume_training
		self.sample_list = []
		print 'itr_start- ', itr_start
		self._load_data()
		obs = self.sample_list[0]
		tgt = self.sample_list[1]
		
		self._policy_iteration(start_epoch=itr_start)


	def processing(self):
		path = './python/gps/pre_train/data/'
		self.read_list = [path + 'target_position_1/sampled_data_itr_0',
		  	   		   path + 'target_position_1/sampled_data_itr_0(0)',
		  	   		   path + 'target_position_1/sampled_data_itr_1',
		  	   		   path + 'tgt_position_2']

		self.obs_final = None
		self._delete_image(self.read_list)
		#self._generate_points()

		# self._augment_image()
		
		'''
		idx = range(self.obs_final.shape[0])
		np.random.shuffle(idx)
		for i in idx[:5]:
			img = self.obs_final[i, 9:]
			img = img.reshape(3, 300, 300)
			img = np.transpose(img, (2, 1, 0))
			img = img/255.0
			plt.imshow(img)
			plt.show()
		'''
		#print self.obs_final.shape
		np.savez('obs_final_new.npz', obs_final=self.obs_final)

	# def testing(self, test_itr):
	def testing(self):
		# itr = test_itr
		# policy_opt_file = self.save_dir + 'policy_itr_%02d.pkl' % itr
		# self.policy_opt.restore_model(itr)

		## load dataset
		fname = '/hdd/gps-master/python/gps/pre_train/data/test_dataset.npz'
		f = np.load(fname)
		# position = f['tgt_mu']
		position = f['tgt_mu'][:,:3]
		obs = f['obs']
		
		## actual testing
		# test_ls = []
		loss_l = []
		num_epochs = 60
		for epoch in range(num_epochs):
			self.policy_opt.restore_model(epoch)
			avg_loss, label, pred = self.policy_opt.testing_loss(obs, position)
			loss_l.append(avg_loss)

			err = 0
			for i in range(5):
				err += np.linalg.norm(pred[i,:] - label[i,:])
			err /= 5

			print 'label: \n', label
			print 'prediction: \n', pred
			print 'loss: \n', avg_loss
			print 'average err: ', err
		
		# np.save('/hdd/gps-master/python/gps/pre_train/test_err/test_loss.npy', loss_l)
		# self.policy_opt = self.data_logger.unpickle(policy_opt_file)
		# np.save('/hdd/gps-master/python/gps/pre_train/test_err/err_%02d.npy' % itr, test_ls)

	def run(self):
		itr_start = 0
		for itr in range(itr_start, self._hyperparams['iterations']):
			for cond in self._train_idx:
				for i in range(self._hyperparams['num_samples']):
					print('sample: ' + str(i))
					self._take_sample(itr, cond, i)

			traj_sample_lists = [
				self.agent.get_samples(cond, -self._hyperparams['num_samples'])
				for cond in self._train_idx
			]
		
			self.agent.clear_samples()
		
			self._take_iteration(itr, traj_sample_lists)


	def _take_sample(self, itr, cond):
		pol = self.policy_opt.policy
		tgt_pose = target_list[cond]
		self.agent.sample(pol, cond, tgt_pose)

	def _save_data(self, itr):
		self.sample_list = [
			self.agent.get_samples(cond, -self._hyperparams['num_samples'])
			for cond in self._train_idx
		]

		obs_data, tgt_mu = np.zeros((0, self.T, self.dO)), np.zeros((0, self.T, self.dU))

		for m in range(self.M):
			samples = self.sample_list[m]
			obs_data = np.concatenate((obs_data, samples.get_obs()))
			tgt_mu = np.concatenate((tgt_mu, samples.get_U()))

		fname = './python/gps/pre_train/data/sampled_data_itr_' + str(itr) + '.npz'
		np.savez(fname, obs_data=obs_data, tgt_mu=tgt_mu)

	def _save_data2(self, itr, fname):
		self.sample_list = [
			self.agent.get_samples(cond, -self._hyperparams['num_samples'])
			for cond in self._train_idx
		]
		
		obs_data, tgt_mu = np.zeros((0, self.T, self.dO)), np.zeros((0, self.T, self.dU))

		for m in range(self.M):
			samples = self.sample_list[m]
			position = np.tile(np.concatenate((test_list[m], np.zeros(3))), (self.T,1))
			position = position.reshape((1,self.T,6))
			obs = np.c_[position, samples.get_obs()/255.0]
			# obs_data = np.concatenate((obs_data, samples.get_obs()))
			obs_data = np.concatenate((obs_data, obs))
			# tgt_mu = np.concatenate((tgt_mu, samples.get_U()))
			tgt_mu = np.concatenate((tgt_mu, position))

		np.savez(fname, obs_data=obs_data, tgt_mu=tgt_mu)
		
	def save_image(self):
		f = np.load('/hdd/gps-master/python/gps/pre_train/data/test_dataset.npz')
		obs = f['obs_data']
		for m in range(self.M):
			for t in range(self.T):
				img = obs[m,t,6:].reshape(3, 300, 300)
				img = np.transpose(img, (2, 1, 0))
				fname = '/hdd/gps-master/python/gps/pre_train/data/test_figure/cond_' + str(m) + '_' + str(t) + '.png'
				
				plt.imshow(img)
				plt.savefig(fname)
				plt.show()

	def _load_data(self):
		# fname = './python/gps/pre_train/data/sampled_data_itr_' + str(itr) + '.npz'
		fname = './python/gps/pre_train/data/obs_final_new.npz'
		npfile = np.load(fname)
		tgt_mu = npfile['obs_final'][:,:3]
		obs_data = npfile['obs_final'][:,6:]
		#print tgt_mu.shape, obs_data.shape

		# self.sample_list = [np.r_[self.sample_list[0], obs_data], np.r_[self.sample_list[1], tgt_mu]]
		self.sample_list = [obs_data, tgt_mu]

	def _take_iteration(self, itr, sample_lists):
		self._policy_iteration(sample_lists)

	def _policy_iteration(self, start_epoch=0):
		inner_iterations = 530 #10
		self.iteration_count = 0
		# loss_l = []
		if start_epoch:
			self.policy_opt.restore_model(epoch = start_epoch)
		else:
			print 'training stage'
		for inner_itr in range(start_epoch, inner_iterations):
			# if self.iteration_count > 0 or inner_itr > 0:

			loss = self._update_policy(inner_itr)
			# loss_l.append(loss)

			# self._validate_policy(val_idx)
			
			## save network model
			# fname = './python/gps/pre_train/policy_opt/tf_checkpoint/policy_checkpoint_pre_training_' + str(self.iteration_count) + '.ckpt'
			fname = './python/gps/pre_train/policy_opt/tf_checkpoint/policy_checkpoint_pre_training_' + str(inner_itr) + '.ckpt'
			self.policy_opt.save_model(fname, inner_itr)
			# np.save('/hdd/gps-master/python/gps/pre_train/loss/loss_%02d.npy' % inner_itr, loss_l)
			np.save('/hdd/gps-master/python/gps/pre_train/loss/loss_%02d.npy' % inner_itr, loss)

			self.iteration_count += 1

	def _update_policy(self, inner_itr):
		obs_data = self.sample_list[0]
		tgt_mu = self.sample_list[1]

		return self.policy_opt.update(obs_data, tgt_mu, inner_itr)

	def _delete_image(self, read_list):
		self.obs_final = np.zeros((0, 270006))
		
		for i in range(len(read_list)):
			npfile = np.load(read_list[i] + '.npz')
			obs = npfile['obs_data']

			delete_list = self._read_delete_list(i, obs.shape)
			# delete_list = self._read_delete_list(i, obs.shape)
			obs = obs.reshape(obs.shape[0]*obs.shape[1], obs.shape[2])
			obs = np.delete(obs, delete_list, axis=0)
			
			img = obs[:,6:] / 255
			pos = obs[:,:6]
			obs = np.c_[pos, img]
			
			self.obs_final = np.concatenate((self.obs_final, obs))

		# np.savez(path + 'delete_image.npz', final_image=self.obs_final)


	def _augment_image(self):
		alpha_list = [1.0, 2.0]
		beta_list = [1.0, 2.0]

		obs_final_orig = self.obs_final
		for i in range(len(alpha_list)):
			image_list = obs_final_orig[:, 9:]
			
			image_list = image_list.reshape(-1, 3, 300, 300)
			image_list = np.transpose(image_list, (0, 3, 2, 1))

			for j in range(len(beta_list)):
				cvt_image = cv2.convertScaleAbs(image_list, alpha=alpha_list[i], beta=beta_list[j])
				# cvt_image = np.transpose(cvt_image, (0, 3, 2, 1)).flatten()
				cvt_image = np.transpose(cvt_image, (0, 3, 2, 1))
				cvt_image = np.reshape(cvt_image, (cvt_image.shape[0], 270000))
				
				cvt_image = np.c_[obs_final_orig[:, 0:9], cvt_image]
				self.obs_final = np.concatenate((self.obs_final, cvt_image))


	def _read_delete_list(self, i, obs_shape):
		delete_list = []
		# f = open(self.sample_list[i] + '.txt', 'r')
		f = open(self.read_list[i] + '.txt', 'r')
			
		lines = f.readlines()
		for i in range(len(lines)):
			if i == 0:
				if lines[i][0:4] != 'cond':
					break
			else:
				line = lines[i].replace(' ', '').split(',')
				if line[0].isdigit():
					delete_index = int(line[0]) * obs_shape[1] + int(line[-1])
					delete_list.append(delete_index)
	
		return delete_list


	def _generate_points(self, size=0.01):
		position = np.repeat(self.obs_final[:, 0:3], 3, axis=1)
		
		position[:, 1] = position[:, 3] + size / np.sqrt(3)
		position[:, 2] = position[:, 6]

		position[:, 4] = position[:, 3] - size * np.sqrt(3) / 6
		position[:, 5] = position[:, 6]

		position[:, 7] = position[:, 3] - size * np.sqrt(3) / 6
		position[:, 8] = position[:, 6]

		position[:, 3] = position[:, 0] - size / 2
		position[:, 6] = position[:, 0] + size / 2

		self.obs_final = np.c_[position, self.obs_final[:, 6:]]

	def check_pos_and_img(self):
		## load test dataset
		fname = '/hdd/gps-master/python/gps/pre_train/data/test_dataset.npz'
		f = np.load(fname)
		# tgt_mu = f['tgt_mu'][0, :].reshape(1,9)
		tgt_mu = f['tgt_mu'][0, :3].reshape(1,3)
		obs = f['obs'][0, :].reshape(1,270000)

		## 
		num_epochs = 188
		loss_l = []
		pred_l = []
		for epoch in range(num_epochs):
			## laod model for each epoch
			self.policy_opt.restore_model(epoch)

			## get loss, label and prediction
			loss, label, pred = self.policy_opt.testing_loss(obs, tgt_mu)
			loss_l.append(loss)
			pred_l.append(pred)

		loss = np.asarray(loss_l)
		pred = np.asarray(pred_l)

		## 3d plot of label and prediction and 2d plot of image
		fig = plt.figure()
		ax1 = fig.add_subplot(121, projection='3d')
		ax2 = fig.add_subplot(122)

		print label
		# label_x = np.array([tgt_mu[0,0]])
		# label_y = np.array([tgt_mu[0,1]])
		# label_z = np.array([tgt_mu[0,2]])
		[label_x, label_y, label_z] = [tgt_mu[0,0], tgt_mu[0,1], tgt_mu[0,2]]
		ax1.scatter(label_x, label_y, label_z, marker='^', c='red')
		
		[pred_x, pred_y, pred_z] = [pred[:,0,0], pred[:,0,1], pred[:,0,2]]
		# color = np.array([[0 + (2/80)*i, 0 + (2/80)*i, 0 + (2/80)*i] for i in range(80)])
		# color = np.transpose(color, (1,0))
		# print color.shape

		# ax1.scatter(pred_x, pred_y, pred_z, marker='o', color=rgba_colors)
		colors = np.linspace(0, 5, num_epochs)
		ax1.scatter(pred_x, pred_y, pred_z, marker='o', c=colors, cmap=plt.cm.Blues)

		ax1.set_xlabel('x')
		ax1.set_ylabel('y')
		ax1.set_zlabel('z')
		ax1.legend(['label', 'prediction'], fontsize = 14)
		ax1.set_xlim3d(-0.1, 1.2)
		ax1.set_ylim3d(-0.3, 0.1)
		ax1.set_zlim3d(-0.145, 0.05)
		# ax1.gca().set_aspect('equal')
		# ax1.gca().set_aspect('equal', adjustable='box')

		img = obs[0,:].reshape(3, 300, 300)
		img = np.transpose(img, (2, 1, 0))
		ax2.imshow(img)

		plt.show()
		exit()

def main():
	parser = argparse.ArgumentParser(description='PreTrain Vision Layer.')
	parser.add_argument('-s', '--sample', action='store_true', help='sampling')
	parser.add_argument('-t', '--train', action='store_true', help='training')
	parser.add_argument('-p', '--process', action='store_true', help='processing')
	parser.add_argument('-e', '--test', action='store_true', help='testing')
	# parser.add_argument('-e', '--test', metavar='N', type=int, help='testing')
	parser.add_argument('-r', '--resume', action='store_true', help='resuming')
	args = parser.parse_args()

	test_itr = args.test

	pre_train = PreTrainMain()
	if args.sample:
		# pre_train = PreTrainMain()
		pre_train.sampling()
	
	if args.train:
		# pre_train = PreTrainMain()
		pre_train.training()

	if args.resume:
		# pre_train = PreTrainMain()
		pre_train.resume()

	if args.process:
		# pre_train = PreTrainMain()
		pre_train.processing()

	if args.test:
		# pre_train = PreTrainMain()
		# pre_train.testing(test_itr)
		pre_train.testing()


if __name__ == '__main__':
	# main()
	pretrain = PreTrainMain()
	pretrain.check_pos_and_img()