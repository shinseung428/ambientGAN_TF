
from glob import glob 
import os
import tensorflow as tf

def load_train_data(args):
	paths = os.path.join(args.data, "img_align_celeba/*.jpg")
	data_count = len(glob(paths))
	
	filename_queue = tf.train.string_input_producer(tf.train.match_filenames_once(paths))

	image_reader = tf.WholeFileReader()
	_, image_file = image_reader.read(filename_queue)
	images = tf.image.decode_jpeg(image_file, channels=3)

	#input image range from -1 to 1
	images = tf.image.resize_images(images ,[args.input_height, args.input_width])
	images = tf.image.convert_image_dtype(images, dtype=tf.float32) / 127.5 - 1

	train_batch = tf.train.shuffle_batch([images],
										 batch_size=args.batch_size,
										 capacity=args.batch_size*2,
										 min_after_dequeue=args.batch_size
										)

	return train_batch, data_count