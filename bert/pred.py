import tensorflow as tf

from .similarity import Similarity


class Pred(tf.keras.Model):
    def __init__(self, hidden_size, **kwargs):
        self.hidden_size = hidden_size
        super(Pred, self).__init__(**kwargs)

    def build(self, input_shape):
        self.dense = tf.keras.layers.Dense(
            units=self.hidden_size, name='cls/predictions/transform/dense')
        # epsilon is important be same with tf.contrib.layers.layer_norm
        # https://github.com/tensorflow/tensorflow/blob/r1.8/tensorflow/contrib/layers/python/layers/layers.py
        self.layer_norm = tf.keras.layers.LayerNormalization(
            epsilon=1e-12,
            name='cls/predictions/transform/LayerNorm')
        self.similarity = Similarity()
        self.activtion = tf.keras.layers.Activation('linear', name='pred')

    def call(self, inputs):
        x, embedding = inputs
        x = self.dense(x)
        x = self.layer_norm(x)
        x = self.similarity([x, embedding])
        x = tf.nn.softmax(x)
        x = self.activtion(x)
        return x
