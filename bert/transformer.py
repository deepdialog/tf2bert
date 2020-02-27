import tensorflow as tf

from .attention import Attention
from .utils import get_activation


class Projection(tf.keras.layers.Layer):
    def __init__(self, hidden_size, hidden_dropout_prob, initializer_range,
                 **kwargs):
        self.hidden_size = hidden_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.initializer_range = initializer_range
        self.dense = None
        self.dropout = None
        self.layer_norm = None
        super(Projection, self).__init__(**kwargs)

    def build(self, input_shape):
        assert isinstance(input_shape, list) and 2 == len(input_shape)
        out_shape, residual_shape = input_shape
        self.input_spec = [
            tf.keras.layers.InputSpec(shape=out_shape),
            tf.keras.layers.InputSpec(shape=residual_shape)
        ]

        self.dense = tf.keras.layers.Dense(
            units=self.hidden_size,
            kernel_initializer=tf.keras.initializers.TruncatedNormal(
                stddev=self.initializer_range),
            name="dense")
        self.dropout = tf.keras.layers.Dropout(rate=self.hidden_dropout_prob)
        # epsilon is important be same with tf.contrib.layers.layer_norm
        # https://github.com/tensorflow/tensorflow/blob/r1.8/tensorflow/contrib/layers/python/layers/layers.py
        # L2174
        self.layer_norm = tf.keras.layers.LayerNormalization(
            epsilon=1e-12, name="LayerNorm")

        super(Projection, self).build(input_shape)

    def call(self, inputs, mask=None, training=None, **kwargs):
        output, residual = inputs
        output = self.dense(output)
        output = self.dropout(output, training=training)
        output = self.layer_norm(tf.add(output, residual))
        return output


class TransformerSelfAttention(tf.keras.layers.Layer):
    def __init__(self, hidden_size, num_attention_heads, hidden_dropout_prob,
                 initializer_range, attention_probs_dropout_prob, **kwargs):
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range

        self.size_per_head = hidden_size // num_attention_heads
        self.attention_layer = None
        self.attention_projector = None

        super(TransformerSelfAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = tf.keras.layers.InputSpec(shape=input_shape)

        self.attention_layer = Attention(
            num_attention_heads=self.num_attention_heads,
            size_per_head=self.size_per_head,
            initializer_range=self.initializer_range,
            attention_probs_dropout_prob=self.attention_probs_dropout_prob,
            name="self",
        )
        self.attention_projector = Projection(
            hidden_size=self.hidden_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            initializer_range=self.initializer_range,
            name="output")

        super(TransformerSelfAttention, self).build(input_shape)

    def call(self, inputs, mask=None, training=None):
        x = inputs
        attention_head = self.attention_layer(x, mask=mask, training=training)
        x = self.attention_projector([attention_head, x],
                                     mask=mask,
                                     training=training)

        return x


class SingleTransformerEncoder(tf.keras.layers.Layer):
    def __init__(self, hidden_size, num_attention_heads, intermediate_size,
                 hidden_act, initializer_range, hidden_dropout_prob,
                 attention_probs_dropout_prob, **kwargs):
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob

        self.size_per_head = hidden_size // num_attention_heads
        self.self_attention_layer = None
        self.intermediate_layer = None
        self.output_projector = None
        super(SingleTransformerEncoder, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = tf.keras.layers.InputSpec(
            shape=input_shape)  # [B, seq_len, hidden_size]

        self.self_attention_layer = TransformerSelfAttention(
            hidden_size=self.hidden_size,
            num_attention_heads=self.num_attention_heads,
            hidden_dropout_prob=self.hidden_dropout_prob,
            initializer_range=self.initializer_range,
            attention_probs_dropout_prob=self.attention_probs_dropout_prob,
            name="attention")

        self.intermediate_layer = tf.keras.layers.Dense(
            name="intermediate/dense",
            units=self.intermediate_size,
            activation=get_activation(self.hidden_act),
            kernel_initializer=tf.keras.initializers.TruncatedNormal(
                stddev=self.initializer_range))

        self.output_projector = Projection(
            hidden_size=self.hidden_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            initializer_range=self.initializer_range,
            name="output")

        super(SingleTransformerEncoder, self).build(input_shape)

    def call(self, inputs, mask=None, training=None):
        layer_input = inputs

        attention_output = self.self_attention_layer(layer_input,
                                                     mask=mask,
                                                     training=training)
        # intermediate
        intermediate_output = self.intermediate_layer(attention_output)
        # output
        layer_output = self.output_projector(
            [intermediate_output, attention_output],
            mask=mask,
            training=training)

        return layer_output


class TransformerEncoder(tf.keras.layers.Layer):
    def __init__(self, num_hidden_layers, hidden_size, num_attention_heads,
                 intermediate_size, hidden_act, initializer_range,
                 hidden_dropout_prob, attention_probs_dropout_prob, **kwargs):
        self.num_hidden_layers = num_hidden_layers
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.encoder_layers = []
        super(TransformerEncoder, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = tf.keras.layers.InputSpec(shape=input_shape)
        # create all transformer encoder sub-layers
        # BERT
        for layer_ndx in range(self.num_hidden_layers):
            encoder_layer = SingleTransformerEncoder(
                hidden_size=self.hidden_size,
                num_attention_heads=self.num_attention_heads,
                intermediate_size=self.intermediate_size,
                hidden_act=self.hidden_act,
                initializer_range=self.initializer_range,
                hidden_dropout_prob=self.hidden_dropout_prob,
                attention_probs_dropout_prob=self.attention_probs_dropout_prob,
                name="layer_{}".format(layer_ndx),
            )
            self.encoder_layers.append(encoder_layer)

        super(TransformerEncoder, self).build(input_shape)

    def call(self, inputs, mask=None, training=None):
        x = inputs
        for layer_ndx in range(self.num_hidden_layers):
            encoder_layer = self.encoder_layers[layer_ndx]
            x = encoder_layer(x,
                              mask=mask,
                              training=training)
        return x
