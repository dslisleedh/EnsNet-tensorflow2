import gin
import tensorflow as tf
import tensorflow_datasets as tfds
from keras_cv.layers import preprocessing as kcv_preprocessing

import numpy as np

from functools import partial
from typing import Optional, Sequence, Callable


@gin.configurable
def load_mnist_dataset():
    train_ds, valid_ds, test_ds = tfds.load(
        'mnist', as_supervised=True, split=['train[:80%]', 'train[80%:]', 'test'])
    return [train_ds, valid_ds, test_ds]


@gin.configurable
def load_fashion_mnist_dataset():
    train_ds, valid_ds, test_ds = tfds.load(
        'fashion_mnist', as_supervised=True, split=['train[:80%]', 'train[80%:]', 'test'])
    return [train_ds, valid_ds, test_ds]


@gin.configurable
def load_cifar10_dataset():
    train_ds, valid_ds, test_ds = tfds.load(
        'cifar10', as_supervised=True, split=['train[:80%]', 'train[80%:]', 'test'])
    return [train_ds, valid_ds, test_ds]


def preprocessing(
        x: tf.Tensor | np.ndarray, y: tf.Tensor | np.ndarray,
        preprocessing_layer: Optional[Callable] = None):
    x = tf.cast(x, tf.float32) / 255.0
    if preprocessing_layer is not None:
        x = preprocessing_layer(x, training=True)
    return x, y


@gin.configurable
def return_augmentation_layer(
        rotation_range: Optional[Sequence[float]],
        scale_range: Optional[Sequence[Sequence[float]]], # [height, width]
        translation_range: Optional[Sequence[Sequence[float]]], # [height, width]
        shear_range: Optional[Sequence[float]], # [x, y]
):
    aug_layers = []
    if rotation_range is not None:
        aug_layers.append(tf.keras.layers.RandomRotation(
            rotation_range, fill_mode='constant', fill_value=0.0))
    if scale_range is not None:
        aug_layers.append(tf.keras.layers.RandomZoom(
            height_factor=scale_range[0], width_factor=scale_range[1], fill_mode='constant', fill_value=0.0))
    if translation_range is not None:
        aug_layers.append(tf.keras.layers.RandomTranslation(
            height_factor=scale_range[0], width_factor=scale_range[1], fill_mode='constant', fill_value=0.0))
    if shear_range is not None:
        aug_layers.append(
            kcv_preprocessing.RandomShear(
                x_factor=shear_range[0], y_factor=shear_range[1], fill_mode='constant', fill_value=0.0))
    return tf.keras.Sequential(aug_layers)