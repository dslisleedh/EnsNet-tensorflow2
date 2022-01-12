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

    def call(self, X):
        y = self.Conv(X)
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
                             activation='relu',
                             prob=self.drop_rate
                             ),
            tf.keras.layers.Dense(self.n_labels,
                                  activation='softmax'
                                  )
        ])

    def call(self, X):
        return self.classifier(X)

class Subnetworks(tf.keras.layers.Layer):
    def __init__(self, n_labels):
        super(Subnetworks, self).__init__()
        self.n_labels = n_labels

        self.subnetworks = [Subnetwork(n_nodes=512,
                                       drop_rate=.5,
                                       n_labels=self.n_labels
                                       ) for _ in range(10)
                            ]

    def call(self, X):
        X = tf.split(X,
                     num_or_size_splits = 10,
                     axis = -1)
        y = [subnet(divition) for divition, subnet in zip(X, self.subnetworks)]
        return y


class Ensnet(tf.keras.models.Model):
    def __init__(self, n_labels):
        super(Ensnet, self).__init__()
        self.n_labels = n_labels

        self.augumentation = tf.keras.layers.experimental.preprocessing.RandomRotation(factor=.025)
        self.cnn = Cnn()
        self.cnn_classifier = tf.keras.Sequential([
            tf.keras.layers.GlobalAvgPool2D(),
            tf.keras.layers.Dense(512,
                                  activation='relu',
                                  ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(.5),
            Dropconnectdense(units=512,
                             activation='relu',
                             prob=.5
                             ),
            tf.keras.layers.Dense(10,
                                  activation='softmax'
                                  )
        ])
        self.subnets = Subnetworks(self.n_labels)

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
        X = self.augumentation(X)

        with tf.GradientTape() as tape:
            features = self.cnn(X)
            preds_cnn = self.cnn_classifier(features)
            loss_c = self.loss_fn(y, preds_cnn)
        grads = tape.gradient(loss_c, self.cnn.trainable_variables + self.cnn_classifier.trainable_variables)
        self.cnn_optimizer.apply_gradients(
            zip(grads, self.cnn.trainable_variables + self.cnn_classifier.trainable_variables)
        )

        with tf.GradientTape() as tape:
            features = self.cnn(X)
            preds_subs = self.subnets(features)
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
        return {'mean loss': loss_c + sum(loss_s) / 11, f'{self.cm.name}': m}

    @tf.function
    def test_step(self, Input):
        X, y = Input
        features = self.cnn(X)
        preds = self.subnets(features)
        preds.append(self.cnn_classifier(features))
        loss = []
        for y_hat_ in preds:
            loss.append(self.loss_fn(y, y_hat_))
        preds = tf.argmax(tf.stack(preds, axis=1), -1)
        preds = tf.map_fn(fn=self.get_mode, elems=preds)
        m = self.cm(y, preds)
        return {'mean loss': sum(loss) / len(loss), f'{self.cm.name}': m}

    @tf.function
    def call(self, X):
        features = self.cnn(X)
        preds = self.subnets(features)
        preds.append(self.cnn_classifier(features))
        preds = tf.argmax(tf.stack(preds, axis=1), -1)
        preds = tf.map_fn(fn=self.get_mode, elems=preds)
        return preds