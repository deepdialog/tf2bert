import tensorflow as tf


class Pool(tf.keras.Model):
    def __init__(self, type_vocab_size, **kwargs):
        self.type_vocab_size = type_vocab_size
        super(Pool, self).__init__(**kwargs)

    def build(self, input_shape):
        self.dense = tf.keras.layers.Dense(
            units=self.type_vocab_size)
        self.activtion = tf.keras.layers.Activation('linear', name='pool')

    def call(self, inputs):
        x = inputs
        x = self.dense(x)
        x = tf.nn.softmax(x)
        x = self.activtion(x)
        return x
