import tensorflow as tf 
import numpy as np


def block_pixels(input, p=0.5):
	shape = input.get_shape().as_list()
	prob = tf.random_uniform([shape[0], shape[1], 1], minval=0, maxval=1, dtype=tf.float32)
	prob = tf.tile(prob, [1, 1, 3])
	prob = tf.to_int32(prob < p)
	prob = tf.cast(prob, dtype=tf.float32)
	res = tf.multiply(input, prob)
	return res

def conv_noise(input, k_size, noise):
	pass

def block_patch(input, k_size=32):
	shape = input.get_shape().as_list()

	patch = tf.zeros([k_size, k_size, shape[-1]], dtype=tf.float32)
 
	rand_num = tf.random_uniform([2], minval=0, maxval=shape[0]-k_size, dtype=tf.int32)
	h_, w_ = rand_num[0], rand_num[1]

	padding = [[h_, shape[0]-h_-k_size], [w_, shape[1]-w_-k_size], [0, 0]]
	padded = tf.pad(patch, padding, "CONSTANT", constant_values=1)

	res = tf.multiply(input, padded)
	return res



def keep_patch(input, k_size=32):
	shape = input.get_shape().as_list()

	patch = tf.ones([k_size, k_size, shape[-1]], dtype=tf.float32)
 
	rand_num = tf.random_uniform([2], minval=0, maxval=shape[0]-k_size, dtype=tf.int32)
	h_, w_ = rand_num[0], rand_num[1]

	padding = [[h_, shape[0]-h_-k_size], [w_, shape[1]-w_-k_size], [0, 0]]
	padded = tf.pad(patch, padding, "CONSTANT", constant_values=0)

	res = tf.multiply(input, padded)
	return res



def extract_patch(input, k_size):
	pass
