import tensorflow as tf
from Model import Cifar
from DataReader import load_data, train_valid_split
import os

def configure():
	flags = tf.app.flags

	### YOUR CODE HERE
	flags.DEFINE_integer('resnet_version', 2, 'the version of ResNet')
	flags.DEFINE_integer('resnet_size', 18, 'n: the size of ResNet-(6n+2) v1 or ResNet-(9n+2) v2')
	flags.DEFINE_integer('batch_size', 128, 'training batch size')
	flags.DEFINE_integer('num_classes', 10, 'number of classes')
	flags.DEFINE_integer('save_interval', 10, 'save the checkpoint when epoch MOD save_interval == 0')
	flags.DEFINE_integer('first_num_filters', 16, 'number of classes')
	flags.DEFINE_float('weight_decay', 2e-4, 'weight decay rate')
	flags.DEFINE_string('modeldir', 'resnet-18v2', 'model directory')
	### END CODE HERE
	
	flags.FLAGS.__dict__['__parsed'] = False
	return flags.FLAGS

def main(_):
	sess = tf.Session()
	print('---Prepare data...')

	### YOUR CODE HERE
	data_dir = '../cifar-10-batches-py'
	### END CODE HERE

	x_train, y_train, x_test, y_test = load_data(data_dir)
	x_train_new, y_train_new, x_valid, y_valid = train_valid_split(x_train, y_train)

	model = Cifar(sess, configure())

	### YOUR CODE HERE
	# from Network import ResNet
	# network = ResNet(1, 3, 10, 16)
	# ips = tf.placeholder(tf.float32, shape=(100, 32, 32, 3))
	# sess.run(tf.global_variables_initializer())
	# sess.run(tf.local_variables_initializer())
	# net = network(ips,training=True)
	# from tensorflow.keras import Model
	# model = Model(inputs=ips, outputs=net)
	
	# print(model.summary)
	# # print(sess.run(network(ips,training=True)))
	# writer = tf.summary.FileWriter('output', sess.graph)
	# writer.close()
	# First step: use the train_new set and the valid set to choose hyperparameters.
	# model.train(x_train_new, y_train_new, 200)
	# while True:
	# model.train(x_train_new, y_train_new, 600)
	# model.test_or_validate(x_valid,y_valid,[i*10 for i in range(1,11)])
	# model.test_or_validate(x_valid,y_valid,[20])
	# model.test_or_validate(x_valid, y_valid, [160, 170, 180, 190, 200])
	# model.test_or_validate(x_valid,y_valid,[10])

	# Second step: with hyperparameters determined in the first run, re-train
	# your model on the original train set.
	# model.train(x_train, y_train, 200)

	# Third step: after re-training, test your model on the test set.
	# Report testing accuracy in your hard-copy report.
	model.test_or_validate(x_test, y_test, [170])
	### END CODE HERE

if __name__ == '__main__':
	os.environ['CUDA_VISIBLE_DEVICES'] = '0'
	tf.app.run()
