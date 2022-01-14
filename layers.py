import tensorflow as tf

class Conv(tf.keras.layers.Layer):
    def __init__(self, filters,
                 padding = 'same',
                 strides = (1,1),
                 activation = 'relu',
                 dropout_rate = .25,
                 pool = False
                 ):
        super(Conv, self).__init__()
        self.filters = filters
        self.strides = strides
        self.padding = padding
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.pool = pool

        self.C = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters = self.filters,
                                   kernel_size = (3,3),
                                   padding = self.padding,
                                   strides = self.strides,
                                   activation = self.activation
                                   ),
            tf.keras.layers.BatchNormalization()
        ])
        if self.pool:
            self.C.add(tf.keras.layers.MaxPool2D(pool_size = (2,2),
                                                 strides = (2,2),
                                                 padding = 'same'
                                                 )
                       )
        self.C.add(tf.keras.layers.Dropout(self.dropout_rate))

    def call(self, X):
        return self.C(X)

class Dropconnectdense(tf.keras.layers.Dense):
    def __init__(self, *args, **kwargs):
        self.prob = kwargs.pop('prob', 0.5)
        super(Dropconnectdense, self).__init__(*args, **kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight('kernel',
                                      shape=[int(input_shape[-1]),
                                             self.units],
                                      trainable=True)
        if self.use_bias:
            self.bias = self.add_weight('bias',
                                        shape=[self.units, ],
                                        trainable=True
                                        )

    def call(self, X):
        y = tf.matmul(X, tf.nn.dropout(self.kernel, self.prob))
        if self.use_bias:
            y = y + tf.nn.dropout(self.bias, self.prob)
        return self.activation(y)