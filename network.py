import tensorflow as tf
import numpy as np
import os
from PIL import Image
from glob import glob


class ambientGAN():
    def __init__(self, args):
        self.args = args

        self.images = load_train_data(args)
        self.build_model()
        self.build_loss()


    def build_model(self):

        self.genX = generator(batch_z, name="generator")

        fake_d_logits, fake_d_net = discriminator(fake_measureY, name="discriminator")
        real_d_logits, real_d_net = discriminator(self.images, name="discriminator", reuse=True)

        trainable_vars = tf.trainable_variables()
        self.g_vars = []
        self.d_vars = []
        for v in trainable_vars:
            if "generator" in v:
                self.g_vars.append(v)
            else:
                self.d_vars.append(v)


    def build_loss(self):

        self.g_loss = 0
        self.d_loss = 0
        

    def generator(self, z, name="generator"):
        nets = []
        with tf.variable_scope(name=name) as scope:
            deconv1 = tf.layers.conv2d_transpose(input, 512, 5, 2,
                                                 padding="VALID",
                                                 activation=None,
                                                 name="deconv1"
                                                 )
            deconv1 = batch_norm(deconv1, name="bn1")
            deconv1 = tf.nn.relu(deconv1)
            nets.append(deconv1)

            deconv2 = tf.layers.conv2d_transpose(deconv1, 256, 5, 2,
                                                 padding="VALID",
                                                 activation=None,
                                                 name="deconv2"
                                                 )
            deconv2 = batch_norm(deconv2, name="bn2")
            deconv2 = tf.nn.relu(deconv2)
            nets.append(deconv2)

            deconv3 = tf.layers.conv2d_transpose(deconv2, 128, 5, 2,
                                                 padding="VALID",
                                                 activation=None,
                                                 name="deconv3"
                                                 )
            deconv3 = batch_norm(deconv3, name="bn3")
            deconv3 = tf.nn.relu(deconv3)
            nets.append(deconv3)

            deconv4 = tf.layers.conv2d_transpose(deconv3, 3, 5, 2,
                                                 padding="VALID",
                                                 activation=None,
                                                 name="deconv4"
                                                 )
            deconv4 = tf.nn.tanh(deconv4)
            nets.append(deconv4)


            return deconv4

    def discriminator(self, input, name="discriminator", reuse=False):
        with tf.variable_scope(name=name, reuse=reuse) as scope:
            conv1 = tf.layers.conv2d(input, 512, 5, 2,
                                     padding="VALID",
                                     activation=None,
                                     name="conv1")
            conv1 = batch_norm(conv1, name="bn1")
            conv1 = tf.nn.relu(conv1)

            conv2 = tf.layers.conv2d(conv1, 256, 5, 2,
                                     padding="VALID",
                                     activation=None,
                                     name="conv2")
            conv2 = batch_norm(conv2, name="bn2")
            conv2 = tf.nn.relu(conv2)

            conv3 = tf.layers.conv2d(conv2, 128, 5, 2,
                                     padding="VALID",
                                     activation=None,
                                     name="conv3")
            conv3 = batch_norm(conv1, name="bn3")
            conv3 = tf.nn.relu(conv3)

            conv4 = tf.layers.conv2d(conv3, 64, 5, 2,
                                     padding="VALID",
                                     activation=None,
                                     name="conv4")
            conv4 = batch_norm(conv4, name="bn4")                                                                                                                           
            conv4 = tf.nn.relu(conv4)

            flatten = tf.layers.flatten(conv4)

            #add linear layer

            return flatten



    def measurement_fn(self, input):
        pass