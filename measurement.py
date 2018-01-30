import tensorflow as tf 

def block_pixels(input, p=0.5):
	prob = tf.random_uniform(tf.shape(input), minval=0, maxval=1, dtype=tf.float32)
	prob = tf.clip_by_value(prob, 0, p)
	res = tf.multiply(input, prob)
	return res