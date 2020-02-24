import tensorflow as tf
from bert.embedding import BertEmbedding


def test():

    params = {
        'vocab_size': 1000,
        'type_vocab_size': 2,
        'hidden_size': 768,
        'hidden_dropout_prob': 0.1,
        'initializer_range': 0.02,
        'max_position_embeddings': 512
    }

    be = BertEmbedding(**params)

    batch_size = 32
    max_length = 20
    x = tf.random.uniform((batch_size, max_length),
                          minval=0,
                          maxval=100,
                          dtype=tf.int32)
    t = tf.random.uniform((batch_size, max_length),
                          minval=0,
                          maxval=2,
                          dtype=tf.int32)

    # import pdb; pdb.set_trace()
    assert be([x, t]).shape == (batch_size, max_length, params['hidden_size'])


if __name__ == "__main__":
    test()
