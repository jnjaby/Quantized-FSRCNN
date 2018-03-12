"""A binary to train FSRCNN using multiple GPUs with synchronous updates.

Speed: With batch_size 128.

Usage:
run "FSRCNN.py --train True --gpu 0 --quantize True" for training FSRCNN.
run "FSRCNN.py --train False --gpu 0 --quantize True" for testing.
More flags are available for different usages.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
from six.moves import xrange  # pylint: disable=redefined-builtin
import re
import os
import pdb
import time
import math
import pprint

from scipy.io import savemat
from scipy.io import loadmat
from scipy.misc import imsave
import numpy as np
import tensorflow as tf

import model
import utils

FLAGS = tf.app.flags.FLAGS

pp = pprint.PrettyPrinter()

tf.app.flags.DEFINE_string('train_dir', '/tmp/FSRCNN_train',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_string('eval_dir', '/tmp/FSRCNN_eval',
                           """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('eval_data', 'test',
                           """Either 'test' or 'train_eval'.""")
tf.app.flags.DEFINE_string('gpu', '0',
                           """Which GPU would be used.""")
tf.app.flags.DEFINE_integer('batch_size', 128,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_integer('max_steps', 15000000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_integer('eval_interval_secs', 60,
                            """How often to run the eval.""")
tf.app.flags.DEFINE_float('lr', 1e-3,
                          """Learning rate for training phase.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_boolean('reload', False,
                            """To restore the model and retrain.""")
tf.app.flags.DEFINE_boolean('train', True,
                            """Whether training or not.""")
tf.app.flags.DEFINE_boolean('save', False,
                            """Whether to save the filters as a mat file.""")
tf.app.flags.DEFINE_boolean('run_once', False,
                            """Whether to run eval only once.""")
tf.app.flags.DEFINE_boolean('quantize', False,
                            """Indicating to use quantized version.""")


def save_filters(sess, step):
  """Save filters as .mat for FSRCNN_test released by Dong Chao.
  """
  if FLAGS.quantize:
    v = tf.get_collection('quantize')
  else:
    v = tf.trainable_variables()

  num_layer = (len(v) + 1)//3
  weights = np.zeros((num_layer, 1), dtype=np.object)
  biases = np.zeros((num_layer, 1), dtype=np.object)
  prelu = np.zeros((num_layer, 1), dtype=np.object)
  for i in xrange(num_layer):
    weight = sess.run(v[3*i])
    shape = weight.shape
    weight = weight.transpose(2, 0, 1, 3).reshape((shape[-2], -1, shape[-1]))
    weights[i, 0] = weight.astype(np.double)

    bias = sess.run(v[3*i + 1])
    biases[i,0] = bias[np.newaxis].T.astype(np.single)

    # No activation for the last layer.
    if i < num_layer - 1:
      alpha = sess.run(v[3*i +2])
      prelu[i,0] = alpha.astype(np.double)

  savemat(os.path.join(FLAGS.train_dir,'X%s.mat'%str(step)), {
            'weights_conv': weights,
            'biases_conv': biases,
            'prelu_conv': prelu})


def eval_once(sess, saver, summary_writer, psnr, summary_op, images, labels):
  """Run Eval once.

  Args:
    saver: Saver.
    summary_writer: Summary writer.
    psnr: PSNR op.
    summary_op: Summary op.
  """
  ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
  if ckpt and ckpt.model_checkpoint_path:
    # Restores from checkpoint
    saver.restore(sess, ckpt.model_checkpoint_path)
    # Assuming model_checkpoint_path looks something like:
    #   /my-favorite-path/cifar10_train/model.ckpt-0,
    # extract global_step from it.
    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
  else:
    print('No checkpoint file found')
    return

  # Start the queue runners.
  coord = tf.train.Coordinator()
  try:
    threads = []
    for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
      threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                       start=True))

    num_iter = int(math.ceil(
      utils.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL / FLAGS.batch_size))
    predictions = []
    step = 0
    while step < num_iter and not coord.should_stop():
      predictions.append(sess.run(psnr))
      step += 1

    # Compute PSNR.
    predictions = sum(predictions) / len(predictions)
    print('%s: Average PSNR = %.4f dB' % (datetime.now(), predictions))

    if FLAGS.save:
      save_filters(sess=sess, step=global_step)

    summary = tf.Summary()
    summary.ParseFromString(sess.run(summary_op))
    summary_writer.add_summary(summary, global_step)
  except Exception as e:  # pylint: disable=broad-except
    coord.request_stop(e)

  coord.request_stop()
  coord.join(threads, stop_grace_period_secs=10)


def evaluate():
  """Eval FSRCNN for Set5 dataset."""
  os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
  with tf.Graph().as_default() as g:
    # Get images and labels.
    eval_data = FLAGS.eval_data == 'test'
    # images, labels = model.inputs(eval_data=eval_data)
    images = tf.placeholder(tf.float32)
    labels = tf.placeholder(tf.float32)
    avg_psnr = tf.placeholder(tf.float32)

    # Build a Graph that resize the images from the
    # inference model.
    resized_images = model.inference(images)

    # Calculate the PSNR.
    rmse = tf.sqrt(tf.reduce_mean(tf.square((resized_images - labels)), axis=(1,2)))
    num = 20*tf.log(1./rmse)
    den = tf.log(tf.constant(10, dtype=num.dtype))
    psnr = tf.reduce_mean(num / den)

    # Restore the learned variables for eval.
    saver = tf.train.Saver(tf.global_variables())

    # Build the summary operation based on the TF collection of Summaries.
    tf.summary.scalar('Avg_PSNR', avg_psnr)
    tf.summary.image('HR', resized_images, max_outputs=5)
    summary_op = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(FLAGS.eval_dir, g)

    config = tf.ConfigProto(allow_soft_placement=True,
                            log_device_placement=FLAGS.log_device_placement)
    config.gpu_options.per_process_gpu_memory_fraction = 0.1
    config.gpu_options.allow_growth = True
    # RUn evaluation once.
    while True:
      sess = tf.Session(config=config)

      ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
      if ckpt and ckpt.model_checkpoint_path:
        # Restores from checkpoint
        saver.restore(sess, ckpt.model_checkpoint_path)
        # Assuming model_checkpoint_path looks something like:
        #   /my-favorite-path/cifar10_train/model.ckpt-0,
        # extract global_step from it.
        global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
      else:
        print('No checkpoint file found')
        continue

      # Start the queue runners.
      coord = tf.train.Coordinator()
      try:
        threads = []
        for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
          threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                           start=True))

        filepath = FLAGS.test_dir
        f = [x for x in os.listdir(filepath) if '.mat' in x]
        image = [loadmat(filepath+x)['data'][np.newaxis,:,:,np.newaxis].astype(np.float32) for x in f]
        label = [loadmat(filepath+x)['label'][np.newaxis, 7:-7, 7:-7,np.newaxis].astype(np.float32) for x in f]

        predictions = []
        for i in xrange(len(image)):
          prediction, im = sess.run([psnr, resized_images], feed_dict={images:image[i],labels:label[i]})
          predictions.append(prediction)
          imsave(os.path.join(FLAGS.eval_dir, str(i) + '.png'), (255*im[0, :, :, 0]).astype(np.uint8))

        # Compute PSNR.
        avg_pred = sum(predictions)/len(predictions)
        print('%s: Average PSNR = %.4f dB' % (datetime.now(), avg_pred))

        if FLAGS.save:
          save_filters(sess=sess, step=global_step)

        summary = tf.Summary()
        summary.ParseFromString(sess.run(summary_op, feed_dict={images:image[0],labels:label[0],avg_psnr:avg_pred}))
        summary_writer.add_summary(summary, global_step)
      except Exception as e:  # pylint: disable=broad-except
        coord.request_stop(e)

      coord.request_stop()
      coord.join(threads, stop_grace_period_secs=10)

      if FLAGS.run_once:
        break
      time.sleep(FLAGS.eval_interval_secs)


def tower_loss(scope, images, labels):
  """Calculate the total loss on a single tower running the model.

  Args:
    scope: unique prefix string identifying the tower, e.g. 'tower_0'
    images: Images. 4D tensor of shape [batch_size, height, width, 1].
    labels: Labels. 4D tensor of shape [batch_size, height, width, 1].

  Returns:
     Tensor of shape [] containing the total loss for a batch of data
  """

  # Build inference Graph.
  resize_images = model.inference(images)

  # Build the portion of the Graph calculating the losses. Note that we will
  # assemble the total_loss using a custom function below.
  _ = model.loss(resize_images, labels)

  # Assemble all of the losses for the current tower only.
  losses = tf.get_collection('losses', scope)

  # Calculate the total loss for the current tower.
  total_loss = tf.add_n(losses, name='total_loss')

  # Attach a scalar summary to all individual losses and the total loss; do the
  # same for the averaged version of the losses.
  tf.summary.scalar('total_loss', total_loss)
  return total_loss


def average_gradients(tower_grads):
  """Calculate the average gradient for each shared variable across all towers.

  Note that this function provides a synchronization point across all towers.

  Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
      is over individual gradients. The inner list is over the gradient
      calculation for each tower.
  Returns:
     List of pairs of (gradient, variable) where the gradient has been averaged
     across all towers.
  """
  average_grads = []
  for grad_and_vars in zip(*tower_grads):
    # Note that each grad_and_vars looks like the following:
    #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
    grads = []
    for g, _ in grad_and_vars:
      # Add 0 dimension to the gradients to represent the tower.
      expanded_g = tf.expand_dims(g, 0)

      # Append on a 'tower' dimension which we will average over below.
      grads.append(expanded_g)

    # Average over the 'tower' dimension.
    grad = tf.concat(axis=0, values=grads)
    grad = tf.reduce_mean(grad, 0)

    # Keep in mind that the Variables are redundant because they are shared
    # across towers. So .. we will just return the first tower's pointer to
    # the Variable.
    v = grad_and_vars[0][1]

    # To scale the grads of biases and deconv layer by 0.1. Refer to the paper
    # and the code released by Dong Chao upon request.
    if 'biases' in v.name or 'deconv' in v.name:
      grad *= 0.1

    grad_and_var = (grad, v)
    average_grads.append(grad_and_var)
  return average_grads


def train():
  """Train FSRCNN for a number of steps."""
  os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu
  with tf.Graph().as_default(), tf.device('/cpu:0'):
    # Create a variable to count the number of train() calls. This equals the
    # number of batches processed * FLAGS.num_gpus.
    global_step = tf.get_variable(
        'global_step', [],
        initializer=tf.constant_initializer(0), trainable=False)

    # Create an optimizer that performs gradient descent.
    opt = tf.train.MomentumOptimizer(FLAGS.lr, 0.9)

    # Determine number of GPUs to use.
    num_gpus = len(FLAGS.gpu.split(','))

    # Get images and labels for FSRCNN.
    images, labels = model.distorted_inputs()
    batch_queue = tf.contrib.slim.prefetch_queue.prefetch_queue(
          [images, labels], capacity=2 * num_gpus)
    # Calculate the gradients for each model tower.
    tower_grads = []
    with tf.variable_scope(tf.get_variable_scope()):
      for i in xrange(num_gpus):
        with tf.device('/gpu:%d' % i):
          with tf.name_scope('%s_%d' % (model.TOWER_NAME, i)) as scope:
            # Dequeues one batch for the GPU
            image_batch, label_batch = batch_queue.dequeue()
            # Calculate the loss for one tower of the model. This function
            # constructs the entire model but shares the variables across
            # all towers.
            loss = tower_loss(scope, image_batch, label_batch)

            # Reuse variables for the next tower.
            tf.get_variable_scope().reuse_variables()

            # Retain the summaries from the final tower.
            summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)

            # Calculate the gradients for the batch of data on this tower.
            if FLAGS.quantize:
              tensor_list = tf.get_collection('quantize', scope)
              var_list = tf.trainable_variables()
              grads = zip(tf.gradients(loss, tensor_list), var_list)
            else:
              grads = opt.compute_gradients(loss)

            # Keep track of the gradients across all towers.
            tower_grads.append(grads)

    # We must calculate the mean of each gradient. Note that this is the
    # synchronization point across all towers.
    grads = average_gradients(tower_grads)

    # Add a summary to track the learning rate.
    summaries.append(tf.summary.scalar('learning_rate', FLAGS.lr))

    # Apply the gradients to adjust the shared variables.
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    # Add histograms for trainable variables.
    for var in tf.trainable_variables():
      summaries.append(tf.summary.histogram(var.op.name, var))

    # Group all updates to into a single train op.
    train_op = tf.group(apply_gradient_op)

    # Create a saver.
    saver = tf.train.Saver(tf.global_variables())

    # Build the summary operation from the last tower summaries.
    summary_op = tf.summary.merge(summaries)
    # summary_op = tf.summary.merge_all()

    # Build an initialization operation to run below.
    init = tf.global_variables_initializer()

    global_step = 0

    # Start running operations on the Graph. allow_soft_placement must be set to
    # True to build towers on GPU, as some of the ops do not have GPU
    # implementations.
    config=tf.ConfigProto(allow_soft_placement=True,
                      log_device_placement=FLAGS.log_device_placement)
    config.gpu_options.per_process_gpu_memory_fraction = 0.8
    # config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(init)

    # Restore the model if reload is True.
    if FLAGS.reload:
      ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
      if ckpt and ckpt.model_checkpoint_path:
        # Restores from checkpoint
        saver.restore(sess, ckpt.model_checkpoint_path)
        # Assuming model_checkpoint_path looks something like:
        #   /my-favorite-path/cifar10_train/model.ckpt-0,
        # extract global_step from it.
        global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]

    # Start the queue runners.
    tf.train.start_queue_runners(sess=sess)

    summary_writer = tf.summary.FileWriter(FLAGS.train_dir, sess.graph)

    for step in xrange(int(global_step), FLAGS.max_steps):
      start_time = time.time()
      _, loss_value = sess.run([train_op, loss])
      duration = time.time() - start_time

      assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

      if step % 10 == 0:
        num_examples_per_step = FLAGS.batch_size * num_gpus
        examples_per_sec = num_examples_per_step / duration
        sec_per_batch = duration / num_gpus

        format_str = ('%s: step %d, loss = %.4f (%.1f examples/sec; %.3f '
                      'sec/batch)')
        print (format_str % (datetime.now(), step, loss_value,
                             examples_per_sec, sec_per_batch))

      if step % 100 == 0:
        summary_str = sess.run(summary_op)
        summary_writer.add_summary(summary_str, step)

      # Save the model checkpoint periodically.
      if step % 5000 == 0 or (step + 1) == FLAGS.max_steps:
        checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)


def main(argv=None):  # pylint: disable=unused-argument
  pp.pprint(FLAGS.__flags)
  if FLAGS.train:
    if not os.path.exists(FLAGS.data_dir):
      raise IOError('No such directory: %s' %(FLAGS.data_dir))
    if tf.gfile.Exists(FLAGS.train_dir):
      if FLAGS.reload:
        train()
        exit()
      else:
        tf.gfile.DeleteRecursively(FLAGS.train_dir)
    tf.gfile.MakeDirs(FLAGS.train_dir)
    train()
  else:
    FLAGS.batch_size = 1
    if tf.gfile.Exists(FLAGS.eval_dir):
      if FLAGS.reload:
        evaluate()
      else:
        tf.gfile.DeleteRecursively(FLAGS.eval_dir)
    tf.gfile.MakeDirs(FLAGS.eval_dir)
    evaluate()


if __name__ == '__main__':
  tf.app.run()
