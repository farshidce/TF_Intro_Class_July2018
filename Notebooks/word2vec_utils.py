import os
import gzip
import random
import shutil
import string 
import struct
import sys
sys.path.append('..')
import urllib
import zipfile

from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf
from collections import Counter

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

def safe_mkdir(path):
    """ Create a directory if there isn't one already. """
    try:
        os.mkdir(path)
    except OSError:
        pass

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


def read_data(file_path):
    """ Read data into a list of tokens 
    There should be 17,005,207 tokens
    """
    with zipfile.ZipFile(file_path) as f:
        # tf.compat.as_str returns the given argument as a unicode string
        words = tf.compat.as_str(f.read(f.namelist()[0])).split()
        print("Read tokens. There are %d tokens" % len(words))
    return words

def build_vocab(words, vocab_size, visual_fld):
    """ Build vocabulary of VOCAB_SIZE most frequent words and write it to
    visualization/vocab.tsv
    """
    safe_mkdir(visual_fld)
    file = open(os.path.join(visual_fld, 'vocab.tsv'), 'w')
    
    dictionary = dict()
    count = [('UNK', -1)]
    index = 0
    count.extend(Counter(words).most_common(vocab_size - 1))
    
    for word, _ in count:
        dictionary[word] = index
        index += 1
        file.write(word + '\n')
    
    index_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    file.close()
    return dictionary, index_dictionary

def convert_words_to_index(words, dictionary):
    """ Replace each word in the dataset with its index in the dictionary """
    return [dictionary[word] if word in dictionary else 0 for word in words]

def generate_sample(index_words, context_window_size):
    """ Form training pairs according to the skip-gram model. """
    for index, center in enumerate(index_words):
        context = random.randint(1, context_window_size)
        # get a random target before the center word
        for target in index_words[max(0, index - context): index]:
            yield center, target
        # get a random target after the center wrod
        for target in index_words[index + 1: index + context + 1]:
            yield center, target

def most_common_words(visual_fld, num_visualize):
    """ create a list of num_visualize most frequent words to visualize on TensorBoard.
    saved to visualization/vocab_[num_visualize].tsv
    """
    words = open(os.path.join(visual_fld, 'vocab.tsv'), 'r').readlines()[:num_visualize]
    words = [word for word in words]
    file = open(os.path.join(visual_fld, 'vocab_' + str(num_visualize) + '.tsv'), 'w')
    for word in words:
        file.write(word)
    file.close()

def read_booksintxt():
    bookspath = '../../../../AA_data/booksintextflat/'
    allwordsarray = []

    for root, dirs, files in os.walk(bookspath):
        for fname in files:
            with open(bookspath+fname,'rt',encoding="utf-8") as myfile:
                curfile = myfile.read().replace('\n',' ')
                table = str.maketrans({key: None for key in string.punctuation})
                new_s = curfile.translate(table)     # remove punctuation
                allwordsarray.extend(new_s.lower().split())

    total_words = len(allwordsarray)
    unique_words = len(set(allwordsarray))

    print("Compiled word list from booksintext with {} total words and {} unique words.".format(total_words,unique_words))  
    return allwordsarray

def batch_gen(download_url, expected_byte, vocab_size, batch_size, 
                skip_window, visual_fld):

    if 'booksintext' in download_url:
        words = read_booksintxt()
    elif download_url == 'both':
        download_url = 'http://mattmahoney.net/dc/text8.zip'
        local_dest = 'data/text8.zip'
        download_one_file(download_url, local_dest, expected_byte)
        words = read_data(local_dest)
        words.extend(read_booksintxt())
    else:   
        local_dest = 'data/text8.zip'
        download_one_file(download_url, local_dest, expected_byte)
        words = read_data(local_dest)
    dictionary, _ = build_vocab(words, vocab_size, visual_fld)
    index_words = convert_words_to_index(words, dictionary)
    del words           # to save memory
    single_gen = generate_sample(index_words, skip_window)
    
    while True:
        center_batch = np.zeros(batch_size, dtype=np.int32)
        target_batch = np.zeros([batch_size, 1])
        for index in range(batch_size):
            center_batch[index], target_batch[index] = next(single_gen)
        yield center_batch, target_batch

  