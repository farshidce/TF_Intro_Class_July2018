import random
import ntpath
import os.path

import tensorflow as tf
import numpy as np

import helpers

def download_dataset():
    helpers.mkdir('data')
    url = 'https://www.dropbox.com/s/c452t2dpaq8nm2y/hamsterhare.tar.gz?dl=1'
    path = 'data/hamsterhare.tar.gz'
    print('Downloading hamster hare dataset...')
    helpers.download(url, path)
    print('Dataset downloaded.')
    print('Extracting dataset...')
    helpers.extract_tar(path)
    print('Dataset extracted')


def create_dataset():
    hamster_dir = os.path.join('data','hamsterhare', 'hamster')
    hamster_files = [
        os.path.join(hamster_dir, f)
        for f in os.listdir(hamster_dir)
    ]

    hare_dir = os.path.join('data','hamsterhare', 'hare')
    hare_files = [
        os.path.join(hare_dir, f)
        for f in os.listdir(hare_dir)
    ]

    all_files = hamster_files + hare_files

    random.shuffle(all_files)
    num_files = len(all_files)
    valid_percentage = 0.2
    split = int(num_files * valid_percentage)
    valid_data = all_files[:split]
    train_data = all_files[split:]
    print('Number of training images: {}'.format(len(train_data)))
    print('Number of validation images: {}'.format(len(valid_data)))
    return train_data, valid_data


def batch_generator(data, batch_size, max_epochs, should_distort=False):

    flip_left_right = True
    random_crop = 1
    random_scale = 1
    random_brightness = 1
    num_channels = 3
    height = 227
    width = 227
    pixel_depth = 255.0

    distort_graph = tf.Graph()
    with distort_graph.as_default():
        """
        From https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/image_retraining/retrain.py
        """
        jpeg_name = tf.placeholder(tf.string, name='DistortJPGInput')
        jpeg_data = tf.read_file(jpeg_name)
        decoded_image = tf.image.decode_jpeg(jpeg_data, channels=3)
        resized_image = tf.image.resize_images(decoded_image, [height, width])
        decoded_image_as_float = tf.cast(decoded_image, dtype=tf.float32)
        decoded_image_4d = tf.expand_dims(decoded_image_as_float, 0)
        margin_scale = 1.0 + (random_crop / 100.0)
        resize_scale = 1.0 + (random_scale / 100.0)
        margin_scale_value = tf.constant(margin_scale)
        resize_scale_value = tf.random_uniform([],
                                             minval=1.0,
                                             maxval=resize_scale)
        scale_value = tf.multiply(margin_scale_value, resize_scale_value)
        precrop_width = tf.multiply(scale_value, width)
        precrop_height = tf.multiply(scale_value, width)
        precrop_shape = tf.stack([precrop_height, precrop_width])
        precrop_shape_as_int = tf.cast(precrop_shape, dtype=tf.int32)
        precropped_image = tf.image.resize_bilinear(decoded_image_4d,
                                                  precrop_shape_as_int)
        precropped_image_3d = tf.squeeze(precropped_image, axis=[0])
        cropped_image = tf.random_crop(precropped_image_3d,
                                     [width, width,
                                      num_channels])
        if flip_left_right:
            flipped_image = tf.image.random_flip_left_right(cropped_image)
        else:
            flipped_image = cropped_image
        brightness_min = 1.0 - (random_brightness / 100.0)
        brightness_max = 1.0 + (random_brightness / 100.0)
        brightness_value = tf.random_uniform([],
                                           minval=brightness_min,
                                           maxval=brightness_max)
        brightened_image = tf.multiply(flipped_image, brightness_value)
        distort_result = tf.expand_dims(brightened_image, 0, name='DistortResult')

    distort_sess = tf.Session(graph=distort_graph)
    
    epoch = 0
    idx = 0
    while epoch < max_epochs: 
        batch = []
        labels = []
        for i in range(batch_size):
            if idx + i >= len(data):
                random.shuffle(data)
                epoch += 1
                idx = 0
            image_path = data[idx + i].encode()
            try:
                if should_distort:
                    val = distort_sess.run(distort_result, 
                                           feed_dict={jpeg_name: image_path})
                else:
                    val = distort_sess.run(resized_image, 
                                           feed_dict={jpeg_name: image_path})
            except tf.errors.InvalidArgumentError as e:
                print('skipping file {}'.format(data[idx+i]))
                continue
            if b'n02342885' in ntpath.basename(image_path):
                labels.append(1)
            else:
                labels.append(0)
            batch.append(val)
        idx += batch_size
        yield batch, labels, epoch