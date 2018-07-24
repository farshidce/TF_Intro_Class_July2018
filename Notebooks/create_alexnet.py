import os.path

import tensorflow as tf
import numpy as np

import helpers

def create_alexnet():

  helpers.mkdir('data')
  numpy_data_path = os.path.join('data', 'bvlc_alexnet.npy')
  download_url = 'https://www.dropbox.com/s/gl5wa3uzru555nd/bvlc_alexnet.npy?dl=1'
  print('Downloading pre-trained AlexNet weights.')
  helpers.download(download_url, numpy_data_path)
  print('Weights downloaded.')

  variable_data = np.load(numpy_data_path, encoding='bytes').item()

  conv1_preW = variable_data["conv1"][0]
  conv1_preb = variable_data["conv1"][1]
  conv2_preW = variable_data["conv2"][0]
  conv2_preb = variable_data["conv2"][1]
  conv3_preW = variable_data["conv3"][0]
  conv3_preb = variable_data["conv3"][1]
  conv4_preW = variable_data["conv4"][0]
  conv4_preb = variable_data["conv4"][1]
  conv5_preW = variable_data["conv5"][0]
  conv5_preb = variable_data["conv5"][1]
  fc6_preW = variable_data["fc6"][0]
  fc6_preb = variable_data["fc6"][1]
  fc7_preW = variable_data["fc7"][0]
  fc7_preb = variable_data["fc7"][1]
  fc8_preW = variable_data["fc8"][0]
  fc8_preb = variable_data["fc8"][1]


  pixel_depth = 255.0
  resized_height = 227
  resized_width = 227
  num_channels = 3

  print('Creating AlexNet model.')

  graph = tf.Graph()

  with graph.as_default():
      x = tf.placeholder(tf.uint8, [None, None, None, num_channels],
                         name='input')
      
      to_float = tf.cast(x, tf.float32)
      resized = tf.image.resize_images(to_float, [resized_height, resized_width])
      
      # Convolution 1
      with tf.name_scope('conv1') as scope:
          kernel = tf.Variable(conv1_preW, name='weights')
          biases = tf.Variable(conv1_preb, name='biases')
          conv = tf.nn.conv2d(resized, kernel, [1, 4, 4, 1], padding="SAME")
          bias = tf.nn.bias_add(conv, biases)
          conv1 = tf.nn.relu(bias, name=scope)

      # Local response normalization 2
      radius = 2
      alpha = 2e-05
      beta = 0.75
      bias = 1.0
      lrn1 = tf.nn.local_response_normalization(conv1,
                                                depth_radius=radius,
                                                alpha=alpha,
                                                beta=beta,
                                                bias=bias)

      # Maxpool 1
      pool1 = tf.nn.max_pool(lrn1,
                             ksize=[1, 3, 3, 1],
                             strides=[1, 2, 2, 1],
                             padding='VALID',
                             name='pool1')

      # Convolution 2
      with tf.name_scope('conv2') as scope:

          kernel = tf.Variable(conv2_preW, name='weights')
          biases = tf.Variable(conv2_preb, name='biases')

          input_a, input_b = tf.split(pool1, 2, 3)
          kernel_a, kernel_b = tf.split(kernel, 2, 3)

          with tf.name_scope('A'):
              conv_a = tf.nn.conv2d(input_a, kernel_a, [1, 1, 1, 1], padding="SAME")        

          with tf.name_scope('B'):
              conv_b = tf.nn.conv2d(input_b, kernel_b, [1, 1, 1, 1], padding="SAME")

          conv = tf.concat([conv_a, conv_b], 3)
          bias = tf.nn.bias_add(conv, biases)
          conv2 = tf.nn.relu(bias, name=scope)

      # Local response normalization 2
      radius = 2
      alpha = 2e-05
      beta = 0.75
      bias = 1.0
      lrn2 = tf.nn.local_response_normalization(conv2,
                                                depth_radius=radius,
                                                alpha=alpha,
                                                beta=beta,
                                                bias=bias)

      # Maxpool 2
      pool2 = tf.nn.max_pool(lrn2,
                             ksize=[1, 3, 3, 1],
                             strides=[1, 2, 2, 1],
                             padding='VALID',
                             name='pool2')

      with tf.name_scope('conv3') as scope:
          kernel = tf.Variable(conv3_preW, name='weights')
          biases = tf.Variable(conv3_preb, name='biases')
          conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding="SAME")
          bias = tf.nn.bias_add(conv, biases)
          conv3 = tf.nn.relu(bias, name=scope)


      with tf.name_scope('conv4') as scope:

          kernel = tf.Variable(conv4_preW, name='weights')
          biases = tf.Variable(conv4_preb, name='biases')

          input_a, input_b = tf.split(conv3, 2, 3)
          kernel_a, kernel_b = tf.split(kernel, 2, 3)

          with tf.name_scope('A'):
              conv_a = tf.nn.conv2d(input_a, kernel_a, [1, 1, 1, 1], padding="SAME")        

          with tf.name_scope('B'):
              conv_b = tf.nn.conv2d(input_b, kernel_b, [1, 1, 1, 1], padding="SAME")

          conv = tf.concat([conv_a, conv_b], 3)
          bias = tf.nn.bias_add(conv, biases)
          conv4 = tf.nn.relu(bias, name=scope)


      with tf.name_scope('conv5') as scope:

          kernel = tf.Variable(conv5_preW, name='weights')
          biases = tf.Variable(conv5_preb, name='biases')

          input_a, input_b = tf.split(conv4, 2, 3)
          kernel_a, kernel_b = tf.split(kernel, 2, 3)

          with tf.name_scope('A'):
              conv_a = tf.nn.conv2d(input_a, kernel_a, [1, 1, 1, 1], padding="SAME")        

          with tf.name_scope('B'):
              conv_b = tf.nn.conv2d(input_b, kernel_b, [1, 1, 1, 1], padding="SAME")

          conv = tf.concat([conv_a, conv_b], 3)
          bias = tf.nn.bias_add(conv, biases)
          conv5 = tf.nn.relu(bias, name=scope)


      # Maxpool 2
      pool5 = tf.nn.max_pool(conv5,
                             ksize=[1, 3, 3, 1],
                             strides=[1, 2, 2, 1],
                             padding='VALID',
                             name='pool5')

      # Fully connected 6
      with tf.name_scope('fc6'):
          weights = tf.Variable(fc6_preW, name='fc6_weights')
          bias = tf.Variable(fc6_preb, name='fc6_bias')
          shape = tf.shape(pool5)
          size = shape[1] * shape[2] * shape[3]
          z = tf.matmul(tf.reshape(pool5, [-1, size]), weights) + bias
          fc6 = tf.nn.relu(z, name='relu')

      # Fully connected 7
      with tf.name_scope('fc7'):
          weights = tf.Variable(fc7_preW, name='weights')
          bias = tf.Variable(fc7_preb, name='bias')
          z = tf.matmul(fc6, weights) + bias
          fc7 = tf.nn.relu(z, name='relu')

      # Fully connected 8
      with tf.name_scope('fc8'):
          weights = tf.Variable(fc8_preW, name='weights')
          bias = tf.Variable(fc8_preb, name='bias')
          fc8 = tf.matmul(fc7, weights) + bias

      softmax = tf.nn.softmax(fc8)

      init = tf.global_variables_initializer()

  print('Model created.')

  sess = tf.Session(graph=graph)
  sess.run(init)

  print('Exporting TensorBoard graph to tbout/alexnet')
  writer = tf.summary.FileWriter('tbout/alexnet', graph=graph)
  writer.close()

  print('Exporting TensorFlow model to data/alexnet')
  with graph.as_default():
      saver = tf.train.Saver()
      save_path = saver.save(sess, 'data/alexnet')