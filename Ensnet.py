import tensorflow as tf
from layers import *


class Cnn(tf.keras.layers.Layer):
    def __init__(self):
        super(Cnn, self).__init__()

        self.Conv = tf.keras.Sequential([
            Conv(filters=64,
                 padding='valid'
                 ),
            Conv(filters=128),
            Conv(filters=256,
                 padding='valid',
                 pool=True
                 ),
            Conv(filters=512,
                 padding='valid'
                 ),
            Conv(filters=1024),
            Conv(filters=2000,
                 padding='valid',
                 pool=True
                 )
        ])
        self.F = tf.keras.layers.Flatten()

    def call(self, X):
        y = self.Conv(X)
        y = tf.split(y,
                     num_or_size_splits=10,
                     axis=-1
                     )
        y = [self.F(y_) for y_ in y]
        return y


class Subnetworks(tf.keras.layers.Layer):
    def __init__(self, n_nodes, drop_rate, n_labels):
        super(Subnetworks, self).__init__()
        self.n_nodes = n_nodes
        self.drop_rate = drop_rate
        self.n_labels = n_labels

        self.classifier = tf.keras.Sequential([
            tf.keras.layers.Dense(self.n_nodes,
                                  activation='relu'
                                  ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(self.drop_rate),
            Dropconnectdense(units=self.n_nodes,
                             activation='relu',
                             prob=self.drop_rate
                             ),
            tf.keras.layers.Dense(self.n_labels,
                                  activation='softmax'
                                  )
        ])

    def call(self, X):
        return self.classifier(X)


class Ensnet(tf.keras.models.Model):
    def __init__(self, n_labels):
        super(Ensnet, self).__init__()
        self.n_labels = n_labels

        self.cnn = Cnn()
        self.subnetworks = [Subnetworks(n_nodes=512,
                                        drop_rate=.5,
                                        n_labels=self.n_labels
                                        ) for _ in range(10)]

        self.build((None, 28, 28, 1))
        self.subvars = []
        for net in self.subnetworks:
            self.subvars += net.trainable_variables

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
    def train_step(self, Input):
        X, y = Input

        with tf.GradientTape() as tape:
            features = self.cnn(X)
            preds = []
            for divition, subnetwork in zip(features, self.subnetworks):
                y_hat = subnetwork(divition)
                preds.append(y_hat)

            loss = []
            for y_hat_ in preds:
                loss.append(self.loss_fn(y, y_hat_))
        grads = tape.gradient(loss, self.cnn.trainable_variables)
        self.cnn_optimizer.apply_gradients(
            zip(grads, self.cnn.trainable_variables)
        )

        with tf.GradientTape() as tape:
            features = self.cnn(X)
            preds = []
            for divition, subnetwork in zip(features, self.subnetworks):
                y_hat = subnetwork(divition)
                preds.append(y_hat)

            loss = []
            for y_hat_ in preds:
                loss.append(self.loss_fn(y, y_hat_))
        grads = tape.gradient(loss, self.subvars)
        self.sub_optimizer.apply_gradients(
            zip(grads, self.subvars)
        )
        preds = tf.stack([tf.argmax(p, axis=1) for p in preds], axis=1)
        preds = tf.map_fn(fn=self.get_mode, elems=preds)
        m = self.cm(y, preds)
        return {'mean loss': sum(loss) / len(loss), f'{self.cm.name}': m}

    @tf.function
    def test_step(self, Input):
        X, y = Input
        features = self.cnn(X)
        preds = []
        for divition, subnetwork in zip(features, self.subnetworks):
            y_hat = subnetwork(divition)
            preds.append(y_hat)
        loss = []
        for y_hat_ in preds:
            loss.append(self.loss_fn(y, y_hat_))
        preds = tf.stack([tf.argmax(p, axis=1) for p in preds], axis=1)
        preds = tf.map_fn(fn=self.get_mode, elems=preds)
        m = self.cm(y, preds)
        return {'mean loss': sum(loss) / len(loss), f'{self.cm.name}': m}

    @tf.function
    def call(self, X):
        features = self.cnn(X)
        preds = tf.stack([tf.argmax(subnet(divition), 1) for divition, subnet in zip(features, self.subnetworks)],
                         axis=0)
        preds = tf.map_fn(fn=self.get_mode, elems=preds)
        return preds