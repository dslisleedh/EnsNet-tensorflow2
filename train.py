import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_datasets as tfds

import gin
from gin.tf import external_configurables
import hydra
from hydra.utils import get_original_cwd
from omegaconf import OmegaConf

from src.model import EnsNet
from utils.system import *
from utils.configure import *
from utils.preprocessing import *

from typing import Callable, List
from functools import partial
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')


def train(
        model: tf.keras.models, optimizer: Callable, loss: Callable,
        metrics: List[Callable], patience: int, epochs: int, batch_size: int,
        datasets: List[tf.data.Dataset], preprocessing_layer: tf.keras.Sequential
):
    train_ds, valid_ds, test_ds = datasets
    preprocessing_layer.build((1,) + train_ds.element_spec[0].shape)
    preprocessing_augmentation = partial(preprocessing, preprocessing_layer=preprocessing_layer)
    train_ds = train_ds.shuffle(100000).batch(batch_size, drop_remainder=True).map(preprocessing_augmentation) \
        .prefetch(tf.data.experimental.AUTOTUNE)
    valid_ds = valid_ds.batch(batch_size, drop_remainder=False).map(preprocessing) \
        .prefetch(tf.data.experimental.AUTOTUNE)
    test_ds = test_ds.batch(batch_size, drop_remainder=False).map(preprocessing) \
        .prefetch(tf.data.experimental.AUTOTUNE)

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    callbacks = [
        tf.keras.callbacks.TensorBoard('./logs'),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=patience, restore_best_weights=True, mode='min'
        )
    ]
    model.fit(
        train_ds, validation_data=valid_ds, epochs=epochs, batch_size=batch_size,
        callbacks=callbacks
    )

    evaluation_result = model.evaluate(test_ds, batch_size=batch_size)
    result = dict(zip(model.metrics_names, evaluation_result))
    OmegaConf.save(OmegaConf.create(result), './result.yaml')
    return evaluation_result


@hydra.main(config_path='configs', config_name='config.yaml', version_base=None)
def main(main_config):
    @Runner
    def _main():
        load_external_config()

        check_dataset_config(main_config.dataset)
        config_files = [
            get_original_cwd() + f'/configs/model/{main_config["dataset"]}.gin',
            get_original_cwd() + f'/configs/data/{main_config["dataset"]}.gin',
            get_original_cwd() + f'/configs/optimizer.gin',
            get_original_cwd() + f'/configs/others.gin'
        ]
        gin.parse_config_files_and_bindings(config_files, None)
        subconfig_save = gin.operative_config_str()
        # Main config automatically saved by hydra
        with open(f'./sub_configs.gin', 'w') as f:
            f.write(subconfig_save)

        train_kwargs = load_train_config()
        train(**train_kwargs)

    _main()


if __name__ == '__main__':
    main()