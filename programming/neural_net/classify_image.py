from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os.path
import re
import sys
import tarfile

import numpy as np
from six.moves import urllib
import tensorflow as tf

'''
Source: https://www.tensorflow.org/tutorials/image_recognition
Original Source Copyright 2015 The TensorFlow Authors. All Rights Reserved
Source modified
'''
DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'

def compute_class_vec(image,modeldir):
	classifications = []
	"""
	classifies given image in the 1000 category I
	Args:
		image: Image file name.

	Returns:
		numpy array of shape (1000,)
	"""
	#fetch pretrained imagenet or it already fetched make sure everything is in order
	maybe_download_and_extract(modeldir)
	if not tf.gfile.Exists(image):
		tf.logging.fatal('File does not exist %s', image)
	image_data = tf.gfile.FastGFile(image, 'rb').read()

	# Creates graph from saved GraphDef.
	with tf.gfile.FastGFile(os.path.join(
			modeldir, 'classify_image_graph_def.pb'), 'rb') as f:
		graph_def = tf.GraphDef()
		graph_def.ParseFromString(f.read())
		_ = tf.import_graph_def(graph_def, name='')

	with tf.Session() as sess:
		# Some useful tensors:
		# 'softmax:0': A tensor containing the normalized prediction across
		#   1000 labels.
		# 'pool_3:0': A tensor containing the next-to-last layer containing 2048
		#   float description of the image.
		# 'DecodeJpeg/contents:0': A tensor containing a string providing JPEG
		#   encoding of the image.
		# Runs the softmax tensor by feeding the image_data as input to the graph.
		softmax_tensor = sess.graph.get_tensor_by_name('softmax:0')
		predictions = sess.run(softmax_tensor,
													 {'DecodeJpeg/contents:0': image_data})
		predictions = np.squeeze(predictions)
		print('shout it from the mountain tops')
		print(predictions)
		print(predictions.shape)
		print('aylmao')
		return predictions


def batch_compute_class_vec(images,modeldir):
	classifications = []
	"""
	classifies given images in the 1000 category Imagenet 
	Args:
		image: Image file name.

	Returns:
		numpy array of shape (1008,) FIXME: shape should be 1000,1 and not 1008,1
	"""
	#fetch pretrained imagenet or it already fetched make sure everything is in order
	maybe_download_and_extract(modeldir)

	# Creates graph from saved GraphDef.
	with tf.gfile.FastGFile(os.path.join(
			modeldir, 'classify_image_graph_def.pb'), 'rb') as f:
		graph_def = tf.GraphDef()
		graph_def.ParseFromString(f.read())
		_ = tf.import_graph_def(graph_def, name='')

	with tf.Session() as sess:
		for image in images:
			if not tf.gfile.Exists(image):
				tf.logging.fatal('File does not exist %s', image)
			image_data = tf.gfile.FastGFile(image, 'rb').read()
			# Some useful tensors:
			# 'softmax:0': A tensor containing the normalized prediction across
			#   1000 labels.
			# 'pool_3:0': A tensor containing the next-to-last layer containing 2048
			#   float description of the image.
			# 'DecodeJpeg/contents:0': A tensor containing a string providing JPEG
			#   encoding of the image.
			# Runs the softmax tensor by feeding the image_data as input to the graph.
			softmax_tensor = sess.graph.get_tensor_by_name('softmax:0')
			predictions = sess.run(softmax_tensor,
														{'DecodeJpeg/contents:0': image_data})
			predictions = np.squeeze(predictions)
			print(predictions)
			print(predictions.shape)
			classifications.append(predictions)
		return classifications

def maybe_download_and_extract(modeldir):
	"""Download and extract model tar file."""
	dest_directory = modeldir
	if not os.path.exists(dest_directory):
		os.makedirs(dest_directory)
	filename = DATA_URL.split('/')[-1]
	filepath = os.path.join(dest_directory, filename)
	if not os.path.exists(filepath):
		def _progress(count, block_size, total_size):
			sys.stdout.write('\r>> Downloading %s %.1f%%' % (
					filename, float(count * block_size) / float(total_size) * 100.0))
			sys.stdout.flush()
		filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
		print()
		statinfo = os.stat(filepath)
		print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
	tarfile.open(filepath, 'r:gz').extractall(dest_directory)

#launch at comvi root for paths to work
#TODO: figure out how to do relative paths in comvi
#make sure the test_dataset is fetched before proceeding
#for that just run 'make fetch' in the SSIM directory
def main(_):
	#single
	modeldir = os.path.join(os.getcwd(),'imagenet')
	image = os.path.join(os.getcwd(), 'imagenet','cropped_panda.jpg')

	#batch
	dataset = os.listdir(os.path.join(os.getcwd(),'programming', 'SSIM','test-dataset'))
	dataset_root  = os.path.join(os.getcwd(),'programming', 'SSIM','test-dataset')
	dataset = [os.path.join(dataset_root,file ) for file in dataset]
	dataset= filter(lambda x: x.endswith('jpg'),dataset)

	#performance results:
	#i5 4200U @1900 mhz
	#single ~ 3.58 s
	#batch 100 images: 30.2 sec
	#batch per image: ~0.3 sec

	#compute_class_vec(image,modeldir)
	batch_compute_class_vec(dataset, modeldir)

if __name__ == '__main__':
	tf.app.run(main=main, argv=[sys.argv[0]])
