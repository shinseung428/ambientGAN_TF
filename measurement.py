import tensorflow as tf 

def block_pixels(input, p=0.5):
	shape = input.get_shape().as_list()
	prob = tf.random_uniform([shape[0], shape[1], 1], minval=0, maxval=1, dtype=tf.float32)
	prob = tf.tile(prob, [1, 1, 3])
	prob = tf.to_int32(prob < p)
	prob = tf.cast(prob, dtype=tf.float32)
	res = tf.multiply(input, prob)
	return res
