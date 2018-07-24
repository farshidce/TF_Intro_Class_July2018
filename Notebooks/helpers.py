import pickle
import gzip
import os
import os.path
import sys
import tarfile
from six.moves.urllib.request import urlretrieve, urlopen

import numpy as np

# Helpers to easily make directories and download files

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def download(url, path):
    if not os.path.exists(path):
        urlretrieve(url, path)
    return path


def create_unique_dir(path):
    i = 0
    tb_path = os.path.join(path, str(i))
    while os.path.exists(tb_path) and os.path.isdir(tb_path):
        i += 1
        tb_path = os.path.join(path, str(i))
    return tb_path


def extract_tar(filename, force=False):
    root = os.path.splitext(os.path.splitext(filename)[0])[0]  # remove .tar.gz
    if os.path.isdir(root) and not force:
        # You may override by setting force=True.
        print('{} already present - don\'t need to extract {}.'.format(root, filename))
    else:
        print('Extracting data for {}. This may take a while. Please wait.'.format(root))
        tar = tarfile.open(filename)
        sys.stdout.flush()
        tar.extractall(root[0:root.rfind('/') + 1])
        tar.close()
    return root

def shuffle(*args):
    """
    Shuffles list of NumPy arrays in unison
    """
    state = np.random.get_state()
    for array in args:
        np.random.set_state(state)
        np.random.shuffle(array)

        
def batch_generator(data, labels, batch_size):
    """
    Generator function that continuously returns batches of data and labels from two datasets, 
    as well as the current epoch (how many times we've looped through the dataset).
    
    The `data` and `labels` arrays must have the same number of entries. 
    
    `batch_size` defines the number of examples that will be returned with each batch.
    
    When the generator can no longer return a batch without reusing previous data, it 
    shuffles the datasets, increments the epoch, and continues.
    """
    if len(data) != len(labels):
        raise ValueError('Image data and label data must be same size')
    if batch_size > len(data):
        raise ValueError('Batch size cannot be larger than size of datasets')
    epoch = 0
    i = 0
    while True:
        if i + batch_size > len(data):
            shuffle(data, labels)
            epoch += 1
            i = 0
        yield data[i:i+batch_size], labels[i:i+batch_size], epoch
        i += batch_size
        
        
def _read32(bytestream):
    # Data type is big-endian, 32-bit unsigned integer
    dtype = np.dtype(np.uint32).newbyteorder('>')
    return np.frombuffer(bytestream.read(4), dtype=dtype)[0]


def mnist_load_images(filename):
    with open(filename, 'rb') as f:
        with gzip.GzipFile(fileobj=f) as bytestream:
            magic = _read32(bytestream)
            if magic != 2051:
                raise ValueError("Encountered invalid magic number {} in image file {}".format(magic, f.name))
            num_images = _read32(bytestream)
            rows = _read32(bytestream)
            cols = _read32(bytestream)
            buffer = bytestream.read(num_images * rows * cols)
            data = np.frombuffer(buffer, dtype=np.uint8)
            data = np.float32(data.reshape(num_images, rows, cols, 1))
            return data


def mnist_load_labels(filename):
    with open(filename, 'rb') as f:
        with gzip.GzipFile(fileobj=f) as bytestream:
            magic = _read32(bytestream)
            if magic != 2049:
                raise ValueError("Encountered invalid magic number {} in label file {}".format(magic, f.name))
            num_labels = _read32(bytestream)
            buffer = bytestream.read(num_labels)
            labels = np.int32(np.frombuffer(buffer, dtype=np.uint8))
            return labels
        

def cifar_unpickle(file):
    fo = open(file, 'rb')
    data = pickle.load(fo, encoding='latin1')
    fo.close()
    return data


def mnist_numpy_paths():
    return {'train_data': 'data/mnist-train-images.npy', 
            'train_labels': 'data/mnist-train-labels.npy',
            'test_data': 'data/mnist-test-images.npy', 
            'test_labels': 'data/mnist-test-labels.npy'
            }


def create_mnist_dataset(save_numpy=True):
    mkdir('data')
    train_data_url = 'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz'
    train_labels_url = 'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz'
    test_data_url = 'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz'
    test_labels_url = 'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz'

    is_down = False

    try:
        urlopen('http://yann.lecun.com/', timeout=5)
    except IOError:
        print('http://yann.lecun.com/ is down. Using internet archive as alternative.')
        is_down = True
    if is_down:
        archive_prefix = 'https://web.archive.org/web/20160828233817/'
        train_data_url = archive_prefix + train_data_url
        train_labels_url = archive_prefix + train_labels_url
        test_data_url = archive_prefix + test_data_url
        test_labels_url = archive_prefix + test_labels_url
        
    train_data_path = download(train_data_url, 'data/mnist-train-images.gz')
    train_labels_path = download(train_labels_url, 'data/mnist-train-labels.gz')
    test_data_path = download(test_data_url, 'data/mnist-test-images.gz')
    test_labels_path = download(test_labels_url, 'data/mnist-test-labels.gz')

    train_data = mnist_load_images('data/mnist-train-images.gz')
    train_labels = mnist_load_labels('data/mnist-train-labels.gz')
    test_data = mnist_load_images('data/mnist-test-images.gz')
    test_labels = mnist_load_labels('data/mnist-test-labels.gz')

    if save_numpy:
        np.save('data/mnist-train-images.npy', train_data)
        np.save('data/mnist-train-labels.npy', train_labels)
        np.save('data/mnist-test-images.npy', test_data)
        np.save('data/mnist-test-labels.npy', test_labels)

    return train_data, train_labels, test_data, test_labels


def get_mnist_dataset(save_numpy=True):
    paths = mnist_numpy_paths()
    for key in paths:
        if not os.path.exists(paths[key]):
            return create_mnist_dataset(save_numpy)
    train_data = np.load(paths['train_data'])
    train_labels = np.load(paths['train_labels'])
    test_data = np.load(paths['test_data'])
    test_labels = np.load(paths['test_labels'])
    return train_data, train_labels, test_data, test_labels

