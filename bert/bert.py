import tensorflow as tf

from .embedding import BertEmbedding
from .transformer import TransformerEncoder


class Pooler(tf.keras.layers.Layer):

    def __init__(self, pooler_fc_size, **kwargs):
        self.pooler_fc_size = pooler_fc_size
        super(Pooler, self).__init__(**kwargs)

    def build(self, input_shape):
        self.dense = tf.keras.layers.Dense(
            self.pooler_fc_size, activation='tanh', name='dense')

    def call(self, inputs):
        return self.dense(inputs)


class Bert(tf.keras.Model):
    """
    Bert of TF2

    {
        "attention_probs_dropout_prob": 0.1,
        "directionality": "bidi",
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "hidden_size": 768,
        "initializer_range": 0.02,
        "intermediate_size": 3072,
        "max_position_embeddings": 512,
        "num_attention_heads": 12,
        "num_hidden_layers": 12,
        "pooler_fc_size": 768,
        "pooler_num_attention_heads": 12,
        "pooler_num_fc_layers": 3,
        "pooler_size_per_head": 128,
        "pooler_type": "first_token_transform",
        "type_vocab_size": 2,
        "vocab_size": 21128
    }

    """
    def __init__(self, vocab_size, type_vocab_size, hidden_size,
                 hidden_dropout_prob, initializer_range,
                 max_position_embeddings, num_hidden_layers,
                 num_attention_heads, intermediate_size, hidden_act,
                 attention_probs_dropout_prob, pooler_fc_size, **kwargs):

        self.vocab_size = vocab_size
        self.type_vocab_size = type_vocab_size
        self.hidden_size = hidden_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.initializer_range = initializer_range
        self.max_position_embeddings = max_position_embeddings
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.pooler_fc_size = pooler_fc_size

        self.pooler = None
        super(Bert, self).__init__(**kwargs)

    def build(self, input_shape):
        # Get merged embedding from 3 embeddings:
        # position embedding
        # word embedding
        # word-type embedding
        self.embedding = BertEmbedding(
            vocab_size=self.vocab_size,
            type_vocab_size=self.type_vocab_size,
            hidden_size=self.hidden_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            initializer_range=self.initializer_range,
            max_position_embeddings=self.max_position_embeddings,
            name='bert/embeddings')
        # Run multiple transformer layers
        self.encoder = TransformerEncoder(
            num_hidden_layers=self.num_hidden_layers,
            hidden_size=self.hidden_size,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            hidden_act=self.hidden_act,
            initializer_range=self.initializer_range,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attention_probs_dropout_prob=self.attention_probs_dropout_prob,
            name='bert/encoder')
        self.pooler = Pooler(
            pooler_fc_size=self.pooler_fc_size,
            name='pooler')
        super(Bert, self).build(input_shape)

    def call(self, inputs, pooler=False, training=None):
        x = inputs
        x = self.embedding(x, training=None)
        x = self.encoder(x, training=None)
        if pooler:
            p = self.pooler(x[:, 0, :])
            return x, p
        return x
