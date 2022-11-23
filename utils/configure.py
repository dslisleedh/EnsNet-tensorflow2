import gin
import tensorflow as tf
import tensorflow_addons as tfa


def load_external_config():
    def _register_losses(module):
        gin.config.external_configurable(module, module='tf.keras.losses')
    _register_losses(tf.keras.losses.MeanSquaredError)
    _register_losses(tf.keras.losses.MeanAbsoluteError)
    _register_losses(tf.keras.losses.BinaryCrossentropy)
    _register_losses(tf.keras.losses.SparseCategoricalCrossentropy)
    _register_losses(tf.keras.losses.CategoricalCrossentropy)

    def _register_metrics(module):
        gin.config.external_configurable(module, module='tf.keras.metrics')
    _register_metrics(tf.keras.metrics.MeanSquaredError)
    _register_metrics(tf.keras.metrics.MeanAbsoluteError)
    _register_metrics(tf.keras.metrics.SparseCategoricalAccuracy)
    gin.config.external_configurable(tfa.metrics.F1Score, 'tfa.metrics.F1Score')

    @gin.configurable
    class SparseF1Score(tfa.metrics.F1Score):
        def __init__(self, *args, **kwargs):
            super(SparseF1Score, self).__init__(*args, **kwargs)
            self._num_classes = kwargs['num_classes']

        def update_state(self, y_true, y_pred, sample_weight=None):
            y_true = tf.one_hot(y_true, self._num_classes)
            y_true = tf.gather(y_true, 0, axis=1)
            return super(SparseF1Score, self).update_state(y_true, y_pred, sample_weight)


@gin.configurable
def list_wrapper(**kwargs):
    return list(kwargs.values())


@gin.configurable(name_or_fn='train_config')
def load_train_config(
        **kwargs
):
    '''
    Configs to be used in training.

    Expected kwargs:
    model, optimizer, loss, metrics, patience, epochs, batch_size,
    dataset, preprocessing_layer
    '''
    return kwargs
