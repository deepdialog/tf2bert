import tensorflow as tf


class BertEmbedding(tf.keras.layers.Layer):
    def __init__(self, vocab_size, type_vocab_size, hidden_size,
                 hidden_dropout_prob, initializer_range,
                 max_position_embeddings, **kwargs):
        self.vocab_size = vocab_size
        self.type_vocab_size = type_vocab_size
        self.hidden_size = hidden_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.initializer_range = initializer_range
        self.max_position_embeddings = max_position_embeddings

        self.word_embeddings_layer = None
        self.token_type_embeddings_layer = None
        self.position_embeddings_layer = None
        self.layer_norm_layer = None
        self.dropout_layer = None

        super(BertEmbedding, self).__init__(**kwargs)

    def build(self, input_shape):
        if isinstance(input_shape, list):
            assert len(input_shape) == 2
            input_ids_shape, token_type_ids_shape = input_shape
            self.input_spec = [
                tf.keras.layers.InputSpec(shape=input_ids_shape),
                tf.keras.layers.InputSpec(shape=token_type_ids_shape)
            ]
        else:
            input_ids_shape = input_shape
            self.input_spec = tf.keras.layers.InputSpec(shape=input_ids_shape)

        self.word_embeddings_layer = tf.keras.layers.Embedding(
            input_dim=self.vocab_size,
            output_dim=self.hidden_size,
            mask_zero=True,
            name="word_embeddings")

        self.token_type_embeddings_layer = tf.keras.layers.Embedding(
            input_dim=self.type_vocab_size,
            output_dim=self.hidden_size,
            mask_zero=False,
            name="token_type_embeddings")

        # self.position_embeddings_layer = PositionEmbedding(
        #     name="position_embeddings", hidden_size=self.hidden_size)
        self.position_embeddings = self.add_weight(
            name="position_embeddings_layer",
            dtype=tf.keras.backend.floatx(),
            shape=[self.max_position_embeddings, self.hidden_size],
            initializer=tf.keras.initializers.TruncatedNormal(
                stddev=self.initializer_range))

        self.layer_norm_layer = tf.keras.layers.LayerNormalization(
            name="LayerNorm")
        self.dropout_layer = tf.keras.layers.Dropout(
            rate=self.hidden_dropout_prob)

        super(BertEmbedding, self).build(input_shape)

    def call(self, inputs, mask=None, training=None):
        input_ids, token_type_ids = inputs
        input_ids = tf.cast(input_ids, dtype=tf.int32)
        embedding_output = self.word_embeddings_layer(input_ids)

        token_type_ids = tf.cast(token_type_ids, dtype=tf.int32)
        embedding_output += self.token_type_embeddings_layer(token_type_ids)

        shape = tf.shape(input_ids)
        embedding_output = tf.concat([
            embedding_output[:, :self.max_position_embeddings, :] +
            tf.expand_dims(self.position_embeddings[:shape[1]], 0),
            embedding_output[:, self.max_position_embeddings:, :]
        ], axis=1)

        embedding_output = self.layer_norm_layer(embedding_output)
        embedding_output = self.dropout_layer(embedding_output,
                                              training=training)

        return embedding_output  # [B, seq_len, hidden_size]

    def compute_mask(self, inputs, mask=None):
        input_ids, token_type_ids = inputs
        return tf.not_equal(input_ids, 0)
