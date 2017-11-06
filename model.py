"""Builds the network.
"""
# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import sys
import pdb

from six.moves import urllib
import tensorflow as tf

import utils

FLAGS = tf.app.flags.FLAGS

# Basic model parameters.
tf.app.flags.DEFINE_string('data_dir', '/home/dongchao/Desktop/ruicheng/FSRCNN_TF/Train/General-100-aug',
                        	"""Path to data directory.""")
tf.app.flags.DEFINE_string('test_dir', '/home/dongchao/Desktop/ruicheng/FSRCNN_TF/Test/Set5',
							"""Path to test data directory.""")
tf.app.flags.DEFINE_boolean('use_fp16', False,
                            """Train the model using fp16.""")

# Global constants describing the data set.
IMAGE_SIZE = utils.IMAGE_SIZE
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = utils.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = utils.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

# Global constants for setting the model.
# bitW and bitA are the fractional part of weights and activations, respectively.
bitW = 12
bitA = 12
model_d, model_s, model_m = 32, 5, 1

# If a model is trained with multiple GPUs, prefix all Op names with tower_name
# to differentiate the operations. Note that this prefix is removed from the
# names of the summaries when visualizing a model.
TOWER_NAME = 'tower'

def _activation_summary(x):
  """Helper to create summaries for activations.

  Creates a summary that provides a histogram of activations.

  Args:
    x: Tensor
  Returns:
    nothing
  """
  # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
  # session. This helps the clarity of presentation on tensorboard.
  tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
  tf.summary.histogram(tensor_name + '/activations', x)


def _variable_on_cpu(name, shape, initializer):
  """Helper to create a Variable stored on CPU memory.
  Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable
  Returns:
    Variable Tensor
  """
  with tf.device('/cpu:0'):
    dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
    var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
  return var


def _variable_with_weight_decay(name, shape, stddev, wd):
  """Helper to create an initialized Variable with weight decay.

  Note that the Variable is initialized with a truncated normal distribution.
  A weight decay is added only if one is specified.

  Args:
    name: name of the variable
    shape: list of ints
    stddev: standard deviation of a truncated Gaussian
    wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.

  Returns:
    Variable Tensor
  """
  dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
  var = _variable_on_cpu(
      name,
      shape,
      tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))

  if wd is not None:
    weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
  return var


def quantize(x, k, is_var=True):
    """Quantize tensors to simulate fixed-point arithmetic.

	Note that mathematically quantization always produce zero gradient. Hence
	the gradients are overridden with Identity as STE is adopted.

    Args:
      x: Tensor to be quantized.
      k: bit width of fractional part.
      is_var: boolean indicating of tensors or variables.

    Returns:
      Tensor with the same shape of x.
    """
    n = 2**k
    with tf.name_scope('quantize'):
      with tf.get_default_graph().gradient_override_map({'Round':'Identity'}):
        x = tf.round(x * n) / n
        x = x + tf.stop_gradient(tf.clip_by_value(x, -8, 8-1/n) - x)
        if is_var:
          tf.add_to_collection('quantize', x)
        return x


def prelu(x, channel_shared=False, name='PReLU'):
  """PReLU activation.

  Args:
    x: Tensor of the bottom.
    channel_shared: boolean indicating of sharing params across the channels.
    name: name of variable scope.

  Returns:
    Tensor with the same shape of x.
  """
  if channel_shared:
    w_shape=(1,)
  else:
    w_shape= x.get_shape()[-1]
  with tf.name_scope(name) as scope:
    alphas = _variable_on_cpu(
             'alphas',
             shape=w_shape,
             initializer=tf.constant_initializer(0.0))
    if FLAGS.quantize:
      alphas = quantize(alphas, bitW, is_var=True)
    x = tf.nn.relu(x) + tf.multiply(alphas, (x - tf.abs(x))) * 0.5
    if FLAGS.quantize:
      x = quantize(x, bitA, is_var=False)
  return x


def conv2d(bottom, kernel_size, stddev, strides, padding, scope, wd=0.0):
  """Construct a conv layer with prelu activation.

  Args:
    bottom: the output tensor of last layer.
    kernel_size: list of ints, shape of kernel.
    stddev: standard deviation of a truncated Gaussian.
    strides: list of ints, the stride of the sliding window for each
        dimension of 'input'.
    padding: A string of 'SAME' or 'VALID'.
    scope: the scope name for the operation.
    wd: float, indicating the coef of weight decay.

  Returns:
    conv: 4D tensor of the output.
  """
  kernel = _variable_with_weight_decay('weights',
                                      shape=kernel_size,
                                      stddev=stddev,
                                      wd=wd)
  biases = _variable_on_cpu('biases',
                            shape=[kernel_size[3]],
                            initializer=tf.constant_initializer(0.0))

  if FLAGS.quantize:
    kernel = quantize(kernel, bitW, is_var=True)
    biases = quantize(biases, bitW, is_var=True)

  conv = tf.nn.conv2d(bottom, kernel, strides, padding=padding)
  if FLAGS.quantize:
    conv = quantize(conv, bitA, is_var=False)

  conv = prelu(tf.nn.bias_add(conv, biases),
                      channel_shared=True,
                      name=scope.name)
  _activation_summary(conv)
  return conv


def deconv(bottom, kernel_size, output_size, stddev, strides, padding, scope, wd=0.0):
  """Construct a deconv layer.

  Args:
    bottom: the output tensor of last layer.
    kernel_size: list of ints, shape of kernel.
    output_size: list of ints, shape of output.
    stddev: standard deviation of a truncated Gaussian.
    strides: list of ints, the stride of the sliding window for each
        dimension of 'input'.
    padding: A string of 'SAME' or 'VALID'.
    scope: the scope name for the operation.
    wd: float, indicating the coef of weight decay.

  Returns:
    conv: 4D tensor of the output.
  """
  kernel = _variable_with_weight_decay('weights',
                                       shape=kernel_size,
                                       stddev=stddev,
                                       wd=wd)
  biases = _variable_on_cpu('biases',
                            shape=[1],
                            initializer=tf.constant_initializer(0.0))
  if FLAGS.quantize:
    kernel = quantize(kernel, bitW, is_var=True)
    biases = quantize(biases, bitW, is_var=True)

  conv = tf.nn.conv2d_transpose(bottom, kernel, output_size,
                      strides, padding)
  if FLAGS.quantize:
    conv = quantize(conv, bitA, is_var=False)
  conv = tf.nn.bias_add(conv, biases)
  _activation_summary(conv)
  return conv


def distorted_inputs():
  """Construct distorted input for training using the Reader ops.

  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 1] size.
    labels: Labels. 4D tensor of [batch_size, LABEL_SIZE, LABEL_SIZE, 1] size.

  Raises:
    ValueError: If no data_dir
  """
  if not FLAGS.data_dir:
    raise ValueError('Please supply a data_dir')
  data_dir = os.path.join(FLAGS.data_dir, 'train_data.tfrecords')
  images, labels = utils.distorted_inputs(data_dir=data_dir,
                                                  batch_size=FLAGS.batch_size)
  if FLAGS.use_fp16:
    images = tf.cast(images, tf.float16)
    labels = tf.cast(labels, tf.float16)
  return images, labels


def inputs(eval_data):
  """Construct input for evaluation using the Reader ops.

  Args:
    eval_data: bool, indicating if one should use the train or eval data set.

  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 1] size.
    labels: Labels. 4D tensor of [batch_size, LABEL_SIZE, LABEL_SIZE, 1] size.

  Raises:
    ValueError: If no data_dir
  """
  if not FLAGS.data_dir:
    raise ValueError('Please supply a data_dir')
  data_dir = os.path.join(FLAGS.data_dir, 'train_data.tfrecords')
  test_dir = os.path.join(FLAGS.test_dir, 'test_data.tfrecords')
  images, labels = utils.inputs(eval_data=eval_data,
                                data_dir=data_dir,
                                test_dir=test_dir,
                                batch_size=FLAGS.batch_size)
  if FLAGS.use_fp16:
    images = tf.cast(images, tf.float16)
    labels = tf.cast(labels, tf.float16)
  return images, labels


def inference(images):
  """Build the model.

  Args:
    images: Images returned from distorted_inputs() or inputs().

  Returns:
    SR images.
  """
  # We instantiate all variables using tf.get_variable() instead of
  # tf.Variable() in order to share variables across multiple GPU training runs.
  # If we only ran this model on a single GPU, we could simplify this function
  # by replacing all instances of tf.get_variable() with tf.Variable().
  #
  # conv1
  with tf.variable_scope('conv1') as scope:
    conv1 = conv2d(images, kernel_size=[5, 5, 1, model_d], wd=0.0, stddev=0.05,
                            strides=[1, 1, 1, 1], scope=scope, padding='VALID')

  # conv2
  with tf.variable_scope('conv2') as scope:
    conv2 = conv2d(conv1, kernel_size=[1, 1, model_d, model_s], wd=0.0, stddev=0.6325,
                            strides=[1, 1, 1, 1], scope=scope, padding='VALID')

  # conv2i
  for i in xrange(2, 2+model_m):
    with tf.variable_scope('conv2'+str(i)) as scope:
      conv2 = conv2d(conv2, kernel_size=[3, 3, model_s, model_s], wd=0.0, stddev=0.2108,
                            strides=[1, 1, 1, 1], scope=scope, padding='SAME')

  # conv26
  with tf.variable_scope('conv2'+str(i+1)) as scope:
    conv2 = conv2d(conv2, kernel_size=[1, 1, model_s, model_d], wd=0.0, stddev=0.25,
                          strides=[1, 1, 1, 1], scope=scope, padding='VALID')

  # conv3
  with tf.variable_scope('deconv') as scope:
    conv3 = deconv(conv2, kernel_size=[9, 9, 1, model_d], wd=0.0, stddev=0.001,
                          output_size=[FLAGS.batch_size, 19, 19, 1],
                          strides=[1, 3, 3, 1], scope=scope, padding='SAME')

  return conv3


def loss(resize_images, labels):
  """Add L2Loss to all the trainable variables.

  Add summary for "Loss" and "Loss/avg".
  Args:
    resize_images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 1] size.
    labels: Labels. 4D tensor of [batch_size, LABEL_SIZE, LABEL_SIZE, 1] size.

  Returns:
    Loss tensor of type float.
  """
  # Calculate the average cross entropy loss across the batch.
  mse = tf.reduce_sum(tf.square(resize_images - labels),axis=(1,2))/2
  tf.add_to_collection('losses', tf.reduce_mean(mse))

  # The total loss is defined as the cross entropy loss plus all of the weight
  # decay terms (L2 loss).
  return tf.add_n(tf.get_collection('losses'), name='total_loss')