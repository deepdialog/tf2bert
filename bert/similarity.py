import tensorflow as tf


class Similarity(tf.keras.layers.Layer):
    def build(self, input_shape):
        self.weight = self.add_weight(name='weight',
                                      shape=(input_shape[1][0], ),
                                      dtype=tf.keras.backend.floatx(),
                                      initializer='zeros',
                                      trainable=True)
        super(Similarity, self).build(input_shape)

    def call(self, inputs):
        x, embeddings = inputs
        x = tf.matmul(x, tf.transpose(embeddings)) + self.weight
        return x

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1], input_shape[1][0])
