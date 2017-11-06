from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import random
import glob
import h5py
import os
import pdb

import scipy.misc
import scipy.ndimage
from PIL import Image

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

# Global constants describing the model and the dataset.
IMAGE_SIZE = 11
LABEL_SIZE = 19
UP_SCALE = 3
STRIDE = 4
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 1433344
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 412


def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def read_images(filename_queue):
  """Reads and parses examples from data files.

  Recommendation: if you want N-way read parallelism, call this function
  N times.  This will give you N independent Readers reading different
  files & positions within those files, which will give better mixing of
  examples.

  Args:
    filename_queue: A queue of strings with the filenames to read from.

  Returns:
    An object representing a single example, with the following fields:
      image: a [IMAGE_SIZE, IMAGE_SIZE, 1] float Tensor with the image data.
      label: a [LABEL_SIZE, LABEL_SIZE, 1] float Tensor with the label data.
  """

  class ImageRecord(object):
    pass
  result = ImageRecord()


  # Read a record, getting filenames from the filename_queue.
  reader = tf.TFRecordReader()
  key, value = reader.read(filename_queue)

  # Convert from a reader to a features of example.
  features = tf.parse_single_example(
    value,
    features={
      'input': tf.FixedLenFeature([], tf.string),
      'label': tf.FixedLenFeature([], tf.string),
    })
  image = tf.decode_raw(features['input'], tf.float32)
  label = tf.decode_raw(features['label'], tf.float32)

  result.image = tf.reshape(image, [IMAGE_SIZE, IMAGE_SIZE, 1])
  result.label = tf.reshape(label, [LABEL_SIZE, LABEL_SIZE, 1])
   

  return result


def _generate_image_and_label_batch(image, label, min_queue_examples,
                                    batch_size, shuffle):
  """Construct a queued batch of images and labels.

  Args:
    image: 3-D Tensor of [IMAGE_SIZE, IMAGE_SIZE, 1] of type.float32.
    label: 3-D Tensor of [LABEL_SIZE, LABEL_SIZE, 1] of type.float32.
    min_queue_examples: int32, minimum number of samples to retain
      in the queue that provides of batches of examples.
    batch_size: Number of images per batch.
    shuffle: boolean indicating whether to use a shuffling queue.

  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 1] size.
    labels: Labels. 4D tensor of [batch_size, LABEL_SIZE, LABEL_SIZE, 1] size.
  """
  # Create a queue that shuffles the examples, and then
  # read 'batch_size' images + labels from the example queue.
  num_preprocess_threads = 16
  if shuffle:
    images, labels = tf.train.shuffle_batch(
        [image, label],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * batch_size,
        min_after_dequeue=min_queue_examples)
  else:
    images, labels = tf.train.batch(
        [image, label],
        batch_size=batch_size,
        num_threads=1,
        capacity=min_queue_examples + 3 * batch_size)

  return images, labels


def distorted_inputs(data_dir, batch_size):
  """Construct distorted input for training using the Reader ops.

  Args:
    data_dir: Path to the train data directory.
    batch_size: Number of images per batch.

  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 1] size.
    labels: Labels. 4D tensor of [batch_size, LABEL_SIZE, LABEL_SIZE, 1] size.
  """
  filenames = [data_dir]
  for f in filenames:
    if not tf.gfile.Exists(f):
      print ('It may take a few minutes to generate the data...')
      input_setup()

  # Create a queue that produces the filenames to read.
  filename_queue = tf.train.string_input_producer(filenames)

  # Read examples from files in the filename queue.
  read_input = read_images(filename_queue)

  # Ensure that the random shuffling has good mixing properties.
  min_fraction_of_examples_in_queue = 0.4
  min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
                           min_fraction_of_examples_in_queue)
  print ('Filling queue with %d images before starting to train. '
         'This will take a few minutes.' % min_queue_examples)

  # Generate a batch of images and labels by building up a queue of examples.
  return _generate_image_and_label_batch(read_input.image, read_input.label,
                                         min_queue_examples, batch_size,
                                         shuffle=True)


def inputs(eval_data, data_dir, test_dir, batch_size):
  """Construct input for evaluation using the Reader ops.

  Args:
    eval_data: bool, indicating if one should use the train or eval data set.
    data_dir: Path to the eval data directory.
    batch_size: Number of images per batch.

  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 1] size.
    labels: Labels. 4D tensor of [batch_size, LABEL_SIZE, LABEL_SIZE, 1] size.
  """
  if not eval_data:
    filenames = [data_dir]
    num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
    FLAGS.train = True
  else:
    filenames = [test_dir]
    num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_EVAL
    FLAGS.train = False

  for f in filenames:
    if not tf.gfile.Exists(f):
      print ('It may take a few minutes to generate the data...')
      input_setup()

  # Create a queue that produces the filenames to read.
  filename_queue = tf.train.string_input_producer(filenames)

  # Read examples from files in the filename queue.
  read_input = read_images(filename_queue)

  # Ensure that the random shuffling has good mixing properties.
  min_fraction_of_examples_in_queue = 0.4
  min_queue_examples = int(num_examples_per_epoch *
                           min_fraction_of_examples_in_queue)

  # Generate a batch of images and labels by building up a queue of examples.
  return _generate_image_and_label_batch(read_input.image, read_input.label,
                                         min_queue_examples, batch_size,
                                         shuffle=False)


def preprocess(path):
  """
  Preprocess single image file 
    (1) Read original image as YCbCr format (and grayscale as default)
    (2) Normalize
    (3) Apply image file with bicubic interpolation

  Args:
    path: file path of desired file
    input_: image applied bicubic interpolation (low-resolution)
    label_: image with original resolution (high-resolution)
  """

  im = Image.open(path).convert('YCbCr')
  im = np.array(im)[:,:,0].astype(np.float32)
  label_ = modcrop(im)

  # Must be normalized
  label_ /= 255.

  input_ = scipy.ndimage.interpolation.zoom(label_, (1./UP_SCALE), prefilter=False)

  return input_, label_

def prepare_data(dataset, filename):
  """
  Args:
    dataset: choose train dataset or test dataset
    
    For train dataset, output data would be ['.../t1.bmp', '.../t2.bmp', ..., '.../t99.bmp']
  """
  # if FLAGS.train:
  #   data = glob.glob(os.path.join(dataset, "*.bmp"))
  # else:
  #   data = glob.glob(os.path.join(dataset, "*.bmp"))
  f = os.path.join(dataset, filename)
  with h5py.File(f, 'r') as hf:
    data = np.array(hf.get('data')).transpose((0, 2, 3, 1))
    label = np.array(hf.get('label')).transpose((0, 2, 3, 1))

  return data, label

def make_data(data, label):
  """
  Make input data as h5 file format
  Depending on 'is_train' (flag value), savepath would be changed.
  """
  h, w, d = data.shape[1:4]

  if FLAGS.train:
    filename = os.path.join(FLAGS.data_dir, 'train_data.tfrecords')
  else:
    filename = os.path.join(FLAGS.test_dir, 'test_data.tfrecords')
  writer = tf.python_io.TFRecordWriter(filename)
  for i in xrange(len(data)):
    image_raw = data[i].tostring()
    label_raw = label[i].tostring()

    example = tf.train.Example(features=tf.train.Features(feature={
      'input': _bytes_feature(image_raw),
      'label': _bytes_feature(label_raw)}))
    writer.write(example.SerializeToString())
  writer.close()

def modcrop(image):
  """
  To scale down and up the original image, first thing to do is to have no remainder while scaling operation.
  
  We need to find modulo of height (and width) and scale factor.
  Then, subtract the modulo from height (and width) of original image size.
  There would be no remainder even after scaling operation.
  """
  h, w = image.shape[0:2]
  h -= np.mod(h, UP_SCALE)
  w -= np.mod(w, UP_SCALE)
  image = image[0:h, 0:w]
  return image

def input_setup():
  """
  Read image files and make their sub-images and saved them as a TFRecord file.
  """
  # Load data path
  if FLAGS.train:
    data, label = prepare_data(FLAGS.data_dir, 'train.h5')
  else:
    data, label = prepare_data(FLAGS.test_dir, 'test.h5')

  make_data(data, label)
