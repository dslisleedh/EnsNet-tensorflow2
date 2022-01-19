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

    def call(self, inputs, **kwargs):
        return self.C(inputs)


class Dropconnectdense(tf.keras.layers.Layer):
    def __init__(self, units, prob):
        self.prob = prob
        self.units = units
        super(Dropconnectdense, self).__init__()

        self.dropout = tf.keras.layers.Dropout(self.prob)

    def build(self, input_shape):
        self.kernel = self.add_weight('kernel',
                                      shape=[int(input_shape[-1]),
                                             self.units],
                                      trainable=True)
        self.bias = self.add_weight('bias',
                                    shape=[self.units, ],
                                    trainable=True
                                    )

    def call(self, inputs, **kwargs):
        y = tf.matmul(inputs, self.dropout(self.kernel),) + self.dropout(self.bias)
        return tf.nn.relu(y)