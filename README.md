Quantized-FSRCNN
===================

TensorFlow implementation of FSRCNN with quantized version. This implements illustrates that FSRCNN adopted with 16-bit fixed-point representation delivers performance nearly identical as a full precision one.

## Prerequisites
- Python 2.7
- TensorFlow version> 1.2
- numpy
- Scipy version > 0.18
- h5py
- PIL

## Usage
- Open MATLAB and run `generate_train.m` and `generate_test.m` to generate training and test data. You can also run `data_aug.m` to do data augmentation first.

- Modify the flags `data_dir` and `test_dir` in `model.py` as paths to the directory.

- Set `NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN` in `utils.py` as the number of samples for training.

- To train a new model, run `FSRCNN.py --train True --gpu 0,1 --quantize True` for training. And simultaneously run `FSRCNN.py --train False --gpu 0 --quantize True` for testing. Note that `reload` flag can be set to `True` for reloading a pre-train model. More flags are available for different usages. Check `FSRCNN.py` for all the possible flags.

- After training, you can evaluate performance of the model with `TensorBoard`. Run `tensorboard --logdir=/tmp/FSRCNN_eval`. Also, you can extract parameters and save them in the format `.mat` by set the flag `save` to `True`.

##References
- [FSRCNN](http://mmlab.ie.cuhk.edu.hk/projects/FSRCNN.html)
- [CIFAR10-tutorial](https://github.com/tensorflow/models/tree/master/tutorials/image/cifar10)
- [drakelevy/FSRCNN-TensorFlow](https://github.com/drakelevy/FSRCNN-TensorFlow)
- [liliumao/Tensorflow-srcnn](https://github.com/liliumao/Tensorflow-srcnn)