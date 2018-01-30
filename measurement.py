import tensorflow as tf 

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

def block_patch(input, k_size):
	shape = input.get_shape().as_list()

	patch = tf.zeros([k_size,k_size,shape[-1]], dtype=tf.float32)
 
	start_pos = tf.random_uniform([2], minval=0, maxval=shape[1]/2, dtype=tf.float32)

	res = input[start_pos[0]:start_pos[0]+shape[0]/2, start_pos[1]:start_pos[1]+shape[1]/2, :] = patch
	
	return res



def keep_patch(input, k_size):
	pass

def extract_patch(input, k_size):
	pass
