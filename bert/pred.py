import tensorflow as tf

from .similarity import Similarity


class Pred(tf.keras.Model):
    def __init__(self, **kwargs):
        super(Pred, self).__init__(**kwargs)

    def build(self, input_shape):
        self.dense = tf.keras.layers.Dense(
            units=768, name='cls/predictions/transform/dense')
        self.layer_norm = tf.keras.layers.LayerNormalization(
            name='cls/predictions/transform/LayerNorm')
        self.similarity = Similarity()

    def call(self, inputs):
        x, embedding = inputs
        x = self.dense(x)
        x = self.layer_norm(x)
        x = self.similarity([x, embedding])
        x = tf.nn.softmax(x)
        return x
