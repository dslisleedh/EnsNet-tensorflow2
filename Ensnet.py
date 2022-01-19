import tensorflow as tf
from layers import *

class Cnn(tf.keras.layers.Layer):
    def __init__(self, dataset):
        super(Cnn, self).__init__()
        self.dataset = dataset

        self.Conv = tf.keras.Sequential([
            Conv(filters=64,
                 padding='valid',
                 dropout_rate=.25 if self.dataset == 'cifar10' else .35,
                 ),
            Conv(filters=128,
                 dropout_rate=.25 if self.dataset == 'cifar10' else .35
                 ),
            Conv(filters=256,
                 padding='valid',
                 pool=True,
                 dropout_rate=.25 if self.dataset == 'cifar10' else .35
                 ),
            Conv(filters=512,
                 padding='valid',
                 dropout_rate=.25 if self.dataset == 'cifar10' else .35
                 ),
            Conv(filters=1024,
                 dropout_rate=.25 if self.dataset == 'cifar10' else .35
                 ),
            Conv(filters=2048 if self.dataset == 'cifar10' else 2000,
                 padding='valid',
                 pool=True,
                 dropout_rate=.25 if self.dataset == 'cifar10' else .35
                 )
        ])
        if self.dataset == 'cifar10':
            self.Conv.add(Conv(filters=3000,
                               padding='valid'
                               )
                          )
            self.Conv.add(Conv(filters=3500,
                               padding='valid'
                               )
                          )
            self.Conv.add(Conv(filters=4000,
                               padding='valid'
                               )
                          )

    def call(self, inputs, **kwargs):
        y = self.Conv(inputs)
        return y


class Subnetwork(tf.keras.layers.Layer):
    def __init__(self, n_nodes, drop_rate, n_labels):
        super(Subnetwork, self).__init__()
        self.n_nodes = n_nodes
        self.drop_rate = drop_rate
        self.n_labels = n_labels

        self.classifier = tf.keras.Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(self.n_nodes,
                                  activation='relu',
                                  ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(self.drop_rate),
            Dropconnectdense(units=self.n_nodes,
                             prob=self.drop_rate
                             ),
            tf.keras.layers.Dense(self.n_labels,
                                  activation='softmax'
                                  )
        ])

    def call(self, inputs, **kwargs):
        return self.classifier(inputs)

class Subnetworks(tf.keras.layers.Layer):
    def __init__(self, n_labels, dataset):
        super(Subnetworks, self).__init__()
        self.n_labels = n_labels
        self.dataset = dataset

        self.subnetworks = [Subnetwork(n_nodes=512,
                                       drop_rate=.3 if self.dataset == 'cifar10' else .5,
                                       n_labels=self.n_labels
                                       ) for _ in range(10)
                            ]

    def call(self, inputs, **kwargs):
        features = tf.split(inputs,
                     num_or_size_splits = 10,
                     axis = -1)
        y = [subnet(divition, **kwargs) for divition, subnet in zip(features, self.subnetworks)]
        return y


class Ensnet(tf.keras.models.Model):
    def __init__(self, n_labels, dataset = 'mnist'):
        super(Ensnet, self).__init__()
        self.n_labels = n_labels
        if dataset not in ['mnist','fashion_mnist','cifar10']:
            raise ValueError('Unsupported dataset')
        else:
            self.dataset = dataset

        self.augumentation = tf.keras.Sequential([
            tf.keras.layers.experimental.preprocessing.Resizing(height=30, width=30),
            tf.keras.layers.experimental.preprocessing.RandomCrop(height=28, width=28),
            tf.keras.layers.experimental.preprocessing.RandomRotation(factor=.015)
        ])
        self.cnn = Cnn(self.dataset)
        self.cnn_classifier = tf.keras.Sequential([
            tf.keras.layers.GlobalAvgPool2D(),
            Subnetwork(n_nodes = 512,
                       drop_rate = .3 if self.dataset == 'cifar10' else .5,
                       n_labels = self.n_labels
                       )
        ])
        self.subnets = Subnetworks(self.n_labels, self.dataset)

    def compile(self, loss_fn, optimizer, metrics):
        super(Ensnet, self).compile()
        self.loss_fn = loss_fn
        self.cnn_optimizer = optimizer
        self.sub_optimizer = optimizer
        self.cm = metrics

    @tf.function
    def get_mode(self, X):
        y, idx, count = tf.unique_with_counts(X)
        return y[tf.argmax(count)]

    @tf.function
    def train_step(self, data):
        X, y = data
        X = self.augumentation(X)

        # 1. Update CNN
        with tf.GradientTape() as tape:
            features = self.cnn(X, training=True)
            preds_cnn = self.cnn_classifier(features, training=True)
            loss_c = self.loss_fn(y, preds_cnn)
        grads = tape.gradient(loss_c, self.cnn.trainable_variables + self.cnn_classifier.trainable_variables)
        self.cnn_optimizer.apply_gradients(
            zip(grads, self.cnn.trainable_variables + self.cnn_classifier.trainable_variables)
        )

        # 2. Update subnetworks
        with tf.GradientTape() as tape:
            features = self.cnn(X, training=False)
            preds_subs = self.subnets(features, training=True)
            loss_s = []
            for y_hat_ in preds_subs:
                loss_s.append(self.loss_fn(y, y_hat_))
        grads = tape.gradient(loss_s, self.subnets.trainable_variables)
        self.sub_optimizer.apply_gradients(
            zip(grads, self.subnets.trainable_variables)
        )
        preds = tf.argmax(tf.stack([preds_cnn] + preds_subs, axis=1),
                          axis=-1)
        preds = tf.map_fn(fn=self.get_mode, elems=preds)
        m = self.cm(y, preds)
        return {'mean_loss': (loss_c + sum(loss_s)) / 11, f'{self.cm.name}': m}

    @tf.function
    def test_step(self, Input):
        X, y = Input
        features = self.cnn(X, training=False)
        preds = self.subnets(features, training=False)
        preds.append(self.cnn_classifier(features, training=False))
        loss = []
        for y_hat_ in preds:
            loss.append(self.loss_fn(y, y_hat_))
        preds = tf.argmax(tf.stack(preds, axis=1), -1)
        preds = tf.map_fn(fn=self.get_mode, elems=preds)
        m = self.cm(y, preds)
        return {'mean_loss': sum(loss) / len(loss), f'{self.cm.name}': m}

    @tf.function
    def call(self, X):
        features = self.cnn(X, training = False)
        preds = self.subnets(features, training = False)
        preds.append(self.cnn_classifier(features, training = False))
        preds = tf.argmax(tf.stack(preds, axis=1), -1)
        preds = tf.map_fn(fn=self.get_mode, elems=preds)
        return preds