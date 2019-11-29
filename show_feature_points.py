import matplotlib.pyplot as plt
import numpy as np
import pprint
import matplotlib.cm as cm

IMG_WIDTH = 300
IMG_HEIGHT = 300

base_path = '/hdd/gps-master/experiments/'
exp_name = 'block_insert_new'
fname = 'fp_10_0_cover_cam.npz'
fname2 = 'fp_10_0_cover_cam.png'
path = base_path + exp_name + '/data_files/check_fp/' + fname
path2 = base_path + exp_name + '/data_files/check_fp/' + fname2

f = np.load(path)
img = f['img']

colors = cm.rainbow(np.linspace(0, 1, 32))

fig = plt.figure()
for k in range(20):
	f = np.load(path)

	img = f['img']
	feature = f['fp']

	points = []
	for i in range(0, feature.shape[1], 2):
		points.append(np.array([feature[k,i] * IMG_WIDTH + IMG_WIDTH/2, feature[k,i+1] * IMG_HEIGHT + IMG_HEIGHT/2]))
	points = np.asarray(points)

	ax = fig.add_subplot(4, 5, k+1, autoscale_on=True)
	fig.subplots_adjust(top=1)
	img = np.transpose(img[k].reshape(3,300,300),(2,1,0)).astype(np.uint8)
	implot = ax.imshow(img)

	implot = ax.scatter(points[:,0], points[:,1], s=5, c=colors)
	if k == 0:
		title = 'timestep = ' + str(k)
	else:
		title = str(k)
	ax.set_title(title, fontsize=10)
	# plt.axis('off')
	ax.tick_params(
		axis='both',
		bottom=False,
		left=False,
		labelbottom=False,
		labelleft=False)
plt.tight_layout()
plt.savefig(path2)
plt.show()


'''
for i in range(19):
	path = '/hdd/gps-master/experiments/test_obs/data_files/check_obs/' + 'img_%d.npy' % i
	img = np.load(path)
	
	img = np.transpose(img.reshape(3,300,300), (2,1,0))
	img = img.astype(np.uint8)
	print img.dtype
	
	plt.subplot(4, 5, i+1, autoscale_on=True)
	plt.imshow(img)
	plt.axis('off')
plt.show()
'''