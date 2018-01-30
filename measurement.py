import tensorflow as tf


def block_pixels(input, p=0.5):
	prob = tf.random_uniform(tf.shape(input), minval=0, maxval=1, dtype=tf.float32)
	prob = tf.clip_by_value(prob, 0, p)
	res = tf.multiply(input, prob)
	return res

def conv_noise(input, p=0.5):
	pass

def keep_patch(input, size=32):
	pass

def extract_patch(input, size=32):
	pass

def pad_rotate_proj(input, rot=30):
	pass

def pad_rotate_proj_theta(input, rot=30):
	pass

def gaussian_proj(input, p=0.5):
	pass

