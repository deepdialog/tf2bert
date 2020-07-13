from collections import OrderedDict

import tensorflow as tf

from .embedding import BertEmbedding
from .transformer import TransformerEncoder
from .pred import Pred


class Pooler(tf.keras.layers.Layer):

    def __init__(self, pooler_fc_size, **kwargs):
        self.pooler_fc_size = pooler_fc_size
        super(Pooler, self).__init__(**kwargs)

    def build(self, input_shape):
        self.dense = tf.keras.layers.Dense(
            self.pooler_fc_size, activation='tanh', name='dense')

    def call(self, inputs):
        return self.dense(inputs)


class SeqRelationship(tf.keras.Model):
    def __init__(self, hidden_size, type_vocab_size,
                 initializer_range, **kwargs):
        self.hidden_size = hidden_size
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        super(SeqRelationship, self).__init__(**kwargs)

    def build(self, input_shape):
        self.seq_relationship_weights = self.add_weight(
            name='output_weights',
            shape=(self.type_vocab_size, self.hidden_size),
            dtype=tf.keras.backend.floatx(),
            initializer=tf.keras.initializers.TruncatedNormal(
                stddev=self.initializer_range))
        self.seq_relationship_bias = self.add_weight(
            name='output_bias',
            shape=(self.type_vocab_size, ),
            dtype=tf.keras.backend.floatx(),
            initializer='zeros')

    def call(self, inputs):
        x = inputs

        x = tf.matmul(x, self.seq_relationship_weights, transpose_b=True)
        x = tf.nn.bias_add(x, self.seq_relationship_bias)

        return x


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
                 attention_probs_dropout_prob,
                 pooler_fc_size=None,
                 embedding_size=None,
                 shared_layer=False, **kwargs):

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
        self.embedding_size = embedding_size
        self.shared_layer = shared_layer

        if embedding_size is None:
            self.embedding_size = hidden_size
        if pooler_fc_size is None:
            self.pooler_fc_size = hidden_size

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
            embedding_size=self.embedding_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            initializer_range=self.initializer_range,
            max_position_embeddings=self.max_position_embeddings,
            name='bert/embeddings')
        # Run multiple transformer layers
        self.encoder = TransformerEncoder(
            num_hidden_layers=self.num_hidden_layers,
            hidden_size=self.hidden_size,
            embedding_size=self.embedding_size,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            hidden_act=self.hidden_act,
            initializer_range=self.initializer_range,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attention_probs_dropout_prob=self.attention_probs_dropout_prob,
            shared_layer=self.shared_layer,
            name='bert/encoder')
        # Official use hidden_size but not pooler_fc_size
        self.pooler = Pooler(
            pooler_fc_size=self.hidden_size,
            name='bert/pooler')
        self.pred = Pred(
            hidden_size=self.embedding_size,
            vocab_size=self.vocab_size,
            hidden_act=self.hidden_act,
            name='cls/predictions'
        )
        self.seq_relationship = SeqRelationship(
            hidden_size=self.hidden_size,
            type_vocab_size=self.type_vocab_size,
            initializer_range=self.initializer_range,
            name='cls/seq_relationship'
        )

        super(Bert, self).build(input_shape)

    def call(self, inputs, training=None):
        input_ids, segment_ids, mask = inputs
        emb = self.embedding(
            [input_ids, segment_ids], training=training)
        encoder_output = self.encoder(emb, mask=mask, training=training)

        pool = self.pooler(encoder_output[:, 0, :])
        rel = self.seq_relationship(pool)

        emb_vec = tf.identity(self.embedding.word_embeddings)
        # if self.encoder.embedding_hidden_mapping_in:
        #     emb_vec = self.encoder.embedding_hidden_mapping_in(emb_vec)
        pred = self.pred([encoder_output, emb_vec])

        ret = OrderedDict((
            ('sequence_output', encoder_output),
            ('pooled_output', pool),
            ('pred_output', pred),
            ('seq_relationship', rel)
        ))

        return ret
