import os
import time
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from Network import ResNet
from ImageUtils import parse_record

"""This script defines the training, validation and testing process.
"""

class Cifar(object):

	def __init__(self, sess, conf):
		self.sess = sess
		self.conf = conf

	def setup(self, training):
		print('---Setup input interfaces...')
		self.inputs = tf.placeholder(tf.float32, shape=(None, 32, 32, 3))
		self.labels = tf.placeholder(tf.int32)
		# Note: this placeholder allows us to set the learning rate for each epoch
		self.learning_rate = tf.placeholder(tf.float32)

		print('---Setup the network...')
		network = ResNet(self.conf.resnet_version, self.conf.resnet_size,
					self.conf.num_classes, self.conf.first_num_filters)

		if training:
			print('---Setup training components...')
			# compute logits
			logits = network(self.inputs, True)

			# predictions for validation
			self.preds = tf.argmax(logits, axis=-1)

			# weight decay
			l2_loss = self.conf.weight_decay * tf.add_n(
						[tf.nn.l2_loss(v) for v in tf.trainable_variables()
						if 'kernel' in v.name])

			### YOUR CODE HERE
			# cross entropy
			cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,labels=self.labels[:]))
			# final loss function
			self.losses = tf.add(cross_entropy,l2_loss)
			### END CODE HERE

			# momentum optimizer with momentum=0.9
			optimizer = tf.train.MomentumOptimizer(
							learning_rate=self.learning_rate, momentum=0.9)

			### YOUR CODE HERE
			# train_op
			extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
			self.train_op = optimizer.minimize(self.losses)
			self.train_op = tf.group([self.train_op, extra_update_ops])
			### END CODE HERE

			print('---Setup the Saver for saving models...')
			self.saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=0)

		else:
			print('---Setup testing components...')
			# compute predictions
			logits = network(self.inputs, False)
			self.preds = tf.argmax(logits, axis=-1)

			print('---Setup the Saver for loading models...')
			self.loader = tf.train.Saver(var_list=tf.global_variables())


	def train(self, x_train, y_train, max_epoch):
		print('###Train###')

		self.setup(True)
		self.sess.run(tf.global_variables_initializer())

		# Determine how many batches in an epoch
		num_samples = x_train.shape[0]
		num_batches = int(num_samples / self.conf.batch_size)
		
		print('---Run...')
		for epoch in range(1, max_epoch+1):

			start_time = time.time()
			# Shuffle
			shuffle_index = np.random.permutation(num_samples)
			curr_x_train = x_train[shuffle_index]
			curr_y_train = y_train[shuffle_index]

			### YOUR CODE HERE
			# Set the learning rate for this epoch
			# Usage example: divide the initial learning rate by 10 after several epochs
			learning_rate = 1e-1 if epoch > 2 else 1e-2
			if epoch*num_batches > 48e3:
				learning_rate *= 1e-2
			elif epoch*num_batches > 32e3:
				learning_rate *= 1e-1
				
			num_batches = int(num_samples/batchsize)
			### END CODE HERE

			loss_value = []
			for i in range(num_batches):
				### YOUR CODE HERE
				# Construct the current batch.
				# Don't forget to use "parse_record" to perform data preprocessing.	
				start_ix = i*batchsize
				end_ix = start_ix + batchsize
				if end_ix > curr_x_train.shape[0]: # shouldn't happen bc of the way num_batches is calculated but still.
					continue
				x_batch = [parse_record(curr_x_train[ix],True) for ix in range(start_ix,end_ix)]
				y_batch = curr_y_train[start_ix:end_ix]
				### END CODE HERE

				# Run
				feed_dict = {self.inputs: x_batch,
							self.labels: y_batch,
							self.learning_rate: learning_rate}
				loss, _ = self.sess.run(
							[self.losses, self.train_op], feed_dict=feed_dict)
				print('Batch {:d}/{:d} Loss {:.6f}'.format(i, num_batches, loss),
						end='\r', flush=True)

			duration = time.time() - start_time
			print('Epoch {:d} Loss {:.6f} Duration {:.3f} seconds.'.format(
						epoch, loss, duration))

			if epoch % self.conf.save_interval == 0:
				self.save(self.saver, epoch)


	def test_or_validate(self, x, y, checkpoint_num_list):
		print('###Test or Validation###')

		self.setup(False)
		self.sess.run(tf.global_variables_initializer())

		# load checkpoint
		for checkpoint_num in checkpoint_num_list:
			checkpointfile = self.conf.modeldir+'/model.ckpt-'+str(checkpoint_num)
			self.load(self.loader, checkpointfile)

			preds = []
			for i in tqdm(range(x.shape[0])):
				### YOUR CODE HERE
				data = parse_record(x[i],False).reshape(1,32,32,3)
				label = y[i].reshape(1)
				pred = self.sess.run(self.preds, feed_dict={self.inputs:data, self.labels:label})
				preds.append(pred)
				### END CODE HERE

			preds = np.array(preds).reshape(y.shape)
			print('Test accuracy: {:.4f}'.format(np.sum(preds==y)/y.shape[0]))

	def save(self, saver, step):
		'''Save weights.
		'''
		model_name = 'model.ckpt'
		checkpoint_path = os.path.join(self.conf.modeldir, model_name)
		if not os.path.exists(self.conf.modeldir):
			os.makedirs(self.conf.modeldir)
		saver.save(self.sess, checkpoint_path, global_step=step)
		print('The checkpoint has been created.')


	def load(self, loader, filename):
		'''Load trained weights.
		''' 
		loader.restore(self.sess, filename)
		print("Restored model parameters from {}".format(filename))

