from pathlib import Path
from typing import Tuple

import os, urllib
import requests
from urllib.parse import urlparse
from hashlib import md5
from subprocess import check_call
import gzip
import shutil
import numpy as np
import struct
import tensorflow as tf
from matplotlib import pyplot as plt

from tqdm import tqdm

INPUT = Path('../data')

FILES_GZ = [
    ('http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
     '6bbc9ace898e44ae57da46a324031adb'),
    ('http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz',
     'a25bea736e30d166cdddb491f175f624'),
    ('http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
     '2646ac647ad5339dbf082846283269ea'),
    ('http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz',
     '27ae3e4e09519cfbb04c329615203637')
]

IMAGES = {'train': INPUT / 'train-images.idx3-ubyte',
          'val': INPUT / 't10k-images.idx3-ubyte'}
LABELS = {'train': INPUT / 'train-labels.idx1-ubyte',
          'val': INPUT / 't10k-labels.idx1-ubyte'}


def md5sum(file: Path):
    data = file.open('rb').read()
    return md5(data).hexdigest()


def get_data(**kwargs):
    """
    Get MNIST data from Yann LeCun site. Check for existence first.
    """
    for raw_url, file_hash in FILES_GZ:
        url = urlparse(raw_url)
        # store data in INPUT
        dest = INPUT / Path(url.path).name

        # check if we already have the unpacked data
        dest_unpacked = dest.with_suffix('')
        if dest_unpacked.exists() and md5sum(dest_unpacked) == file_hash:
            tqdm.write('Already downloaded {dest_unpacked}')
            continue

        # do download with neat progress bars
        r = requests.get(raw_url, stream=True)
        file_size = int(r.headers.get('content-length', 0))
        tqdm.write('Downloading {raw_url}')
        if file_size:
            bar = tqdm(total=file_size)
        else:
            bar = tqdm()
        with dest.open('wb') as f:
            for chunk in r.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
                    f.flush()
                    bar.update(len(chunk))
        bar.close()

        # use gzip module to unpack downloaded files
        tqdm.write('Unpacking {dest}')
        with gzip.open(str(dest), 'rb') as gz_src:
            with dest_unpacked.open('wb') as gz_dst:
                shutil.copyfileobj(gz_src, gz_dst)

        dest.unlink()


def read_mnist_images(split) -> np.ndarray:
    """
    Create tf.data.Dataset out of MNIST images data
    :param split: one of 'train' or 'val' for training or validation data
    """
    assert split in ['train', 'val']

    # read data as numpy array. The data structure is specified in Yann LeCun
    # site.
    fd = IMAGES[split].open('rb')
    magic, size, h, w = struct.unpack('>iiii', fd.read(4 * 4))
    data = np.frombuffer(fd.read(), 'u1').reshape(size, h, w, 1)
    fd.close()

    return data


def read_mnist_labels(split) -> np.ndarray:
    """
    Create tf.data.Dataset out of MNIST labels data
    :param split: one of 'train' or 'val' for training or validation data
    """
    assert split in ['train', 'val']

    # read data as numpy array. The data structure is specified in Yann LeCun
    # site.
    fd = LABELS[split].open('rb')
    magic, size, = struct.unpack('>ii', fd.read(2 * 4))
    data = np.frombuffer(fd.read(), 'u1').reshape(size, 1)
    fd.close()

    return data


def normalize(images):
    """
    Normalize images to [-1,1]
    """

    images = tf.cast(images, tf.float32)
    images /= 255.
    images -= 0.5
    images *= 2
    return images


def transform_train(images, labels):
    """
    Apply transformations to MNIST data for use in training.

    To images: random zoom and crop to 28x28, then normalize to [-1, 1]
    To labels: one-hot encode.
    """
    zoom = 0.9 + np.random.random() * 0.2  # random between 0.9-1.1
    size = int(round(zoom * 28))
    images = tf.image.resize_bilinear(images, (size, size))
    images = tf.image.resize_image_with_crop_or_pad(images, 28, 28)
    images = normalize(images)
    labels = tf.one_hot(labels, 10)
    labels = tf.squeeze(labels, 1)
    return images, labels


def transform_val(images, labels):
    """
    Normalize MNIST images and one-hot encode labels.
    """
    images = normalize(images)
    labels = tf.one_hot(labels, 10)
    labels = tf.squeeze(labels, 1)
    return images, labels


def create_mnist_dataset(batch_size, split):
    """
    Creates an Dataset for MNIST Data.

    This function create the correct tf.data.Dataset for a given split, transforms and
    batch inputs.
    """
    images = read_mnist_images(split)
    labels = read_mnist_labels(split)

    def gen():
        for image, label in zip(images, labels):
            yield image, label

    ds = tf.data.Dataset.from_generator(gen, (tf.uint8, tf.uint8), ((28, 28, 1), (1,)))

    if split == 'train':
        return ds.batch(batch_size).map(transform_train), len(labels)
    elif split == 'val':
        return ds.batch(batch_size).map(transform_val), len(labels)

def huber_loss(labels, predictions, delta=14.0):
    residual = tf.abs(labels - predictions)
    def f1(): return 0.5 * tf.square(residual)
    def f2(): return delta * residual - 0.5 * tf.square(delta)
    return tf.cond(residual < delta, f1, f2)

def safe_mkdir(path):
    """ Create a directory if there isn't one already. """
    try:
        os.mkdir(path)
    except OSError:
        pass

def read_birth_life_data(filename):
    """
    Read in birth_life_2010.txt and return:
    data in the form of NumPy array
    n_samples: number of samples
    """
    text = open(filename, 'r').readlines()[1:]
    data = [line[:-1].split('\t') for line in text]
    births = [float(line[1]) for line in data]
    lifes = [float(line[2]) for line in data]
    data = list(zip(births, lifes))
    n_samples = len(data)
    data = np.asarray(data, dtype=np.float32)
    return data, n_samples

def download_one_file(download_url, 
                    local_dest, 
                    expected_byte=None, 
                    unzip_and_remove=False):
    """ 
    Download the file from download_url into local_dest
    if the file doesn't already exists.
    If expected_byte is provided, check if 
    the downloaded file has the same number of bytes.
    If unzip_and_remove is True, unzip the file and remove the zip file
    """
    if os.path.exists(local_dest) or os.path.exists(local_dest[:-3]):
        print('%s already exists' %local_dest)
    else:
        print('Downloading %s' %download_url)
        local_file, _ = urllib.request.urlretrieve(download_url, local_dest)
        file_stat = os.stat(local_dest)
        if expected_byte:
            if file_stat.st_size == expected_byte:
                print('Successfully downloaded %s' %local_dest)
                if unzip_and_remove:
                    with gzip.open(local_dest, 'rb') as f_in, open(local_dest[:-3],'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                    os.remove(local_dest)
            else:
                print('The downloaded file has unexpected number of bytes')

def download_mnist(path):
    """ 
    Download and unzip the dataset mnist if it's not already downloaded 
    Download from http://yann.lecun.com/exdb/mnist
    """
    safe_mkdir(path)
    url = 'http://yann.lecun.com/exdb/mnist'
    filenames = ['train-images-idx3-ubyte.gz',
                'train-labels-idx1-ubyte.gz',
                't10k-images-idx3-ubyte.gz',
                't10k-labels-idx1-ubyte.gz']
    expected_bytes = [9912422, 28881, 1648877, 4542]

    for filename, byte in zip(filenames, expected_bytes):
        download_url = os.path.join(url, filename)
        local_dest = os.path.join(path, filename)
        download_one_file(download_url, local_dest, byte, True)

def parse_data(path, dataset, flatten):
    if dataset != 'train' and dataset != 't10k':
        raise NameError('dataset must be train or t10k')

    label_file = os.path.join(path, dataset + '-labels-idx1-ubyte')
    with open(label_file, 'rb') as file:
        _, num = struct.unpack(">II", file.read(8))
        labels = np.fromfile(file, dtype=np.int8) #int8
        new_labels = np.zeros((num, 10))
        new_labels[np.arange(num), labels] = 1
    
    img_file = os.path.join(path, dataset + '-images-idx3-ubyte')
    with open(img_file, 'rb') as file:
        _, num, rows, cols = struct.unpack(">IIII", file.read(16))
        imgs = np.fromfile(file, dtype=np.uint8).reshape(num, rows, cols) #uint8
        imgs = imgs.astype(np.float32) / 255.0
        if flatten:
            imgs = imgs.reshape([num, -1])

    return imgs, new_labels

def read_mnist(path, flatten=True, num_train=55000):
    """
    Read in the mnist dataset, given that the data is stored in path
    Return two tuples of numpy arrays
    ((train_imgs, train_labels), (test_imgs, test_labels))
    """
    imgs, labels = parse_data(path, 'train', flatten)
    indices = np.random.permutation(labels.shape[0])
    train_idx, val_idx = indices[:num_train], indices[num_train:]
    train_img, train_labels = imgs[train_idx, :], labels[train_idx, :]
    val_img, val_labels = imgs[val_idx, :], labels[val_idx, :]
    test = parse_data(path, 't10k', flatten)
    return (train_img, train_labels), (val_img, val_labels), test

def get_mnist_dataset(batch_size):
    # Step 1: Read in data
    mnist_folder = 'data/mnist'
    download_mnist(mnist_folder)
    train, val, test = read_mnist(mnist_folder, flatten=False)

    # Step 2: Create datasets and iterator
    train_data = tf.data.Dataset.from_tensor_slices(train)
    train_data = train_data.shuffle(10000) # if you want to shuffle your data
    train_data = train_data.batch(batch_size)

    test_data = tf.data.Dataset.from_tensor_slices(test)
    test_data = test_data.batch(batch_size)

    return train_data, test_data
    
def show(image):
    """
    Render a given numpy.uint8 2D array of pixel data.
    """
    plt.imshow(image, cmap='gray')
    plt.show()