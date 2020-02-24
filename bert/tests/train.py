import tensorflow as tf
from bert.bert import Bert
from bert.pred import Pred
from bert.pool import Pool
from .config import params


def build_model():

    bert = Bert(**params)
    pred = Pred()
    pool = Pool(type_vocab_size=params.get('type_vocab_size'))
    input_ids = tf.keras.layers.Input(shape=(None,), dtype=tf.int32)
    input_ids_type = tf.keras.layers.Input(shape=(None,), dtype=tf.int32)

    x = input_ids
    t = input_ids_type
    x = bert([x, t])
    pool_out = pool(x)
    emb = tf.identity(bert.embedding.word_embeddings_layer.weights[0])
    pred_out = pred([x, emb])
    model = tf.keras.Model(
        inputs=[input_ids, input_ids_type],
        outputs=[pred_out, pool_out])
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=tf.keras.optimizers.Adam()
    )
    return model


def test():

    n_samples = 320
    max_length = 20

    x = tf.random.uniform((n_samples, max_length),
                          minval=0,
                          maxval=params.get('vocab_size'),
                          dtype=tf.int32)
    t = tf.random.uniform((n_samples, max_length),
                          minval=0,
                          maxval=params.get('type_vocab_size'),
                          dtype=tf.int32)

    y = tf.random.uniform((n_samples, max_length),
                          minval=0,
                          maxval=params.get('vocab_size'),
                          dtype=tf.int32)

    yt = tf.random.uniform((n_samples, max_length),
                           minval=0,
                           maxval=params.get('type_vocab_size'),
                           dtype=tf.int32)

    x = tf.data.Dataset.from_tensor_slices(x)
    t = tf.data.Dataset.from_tensor_slices(t)
    y = tf.data.Dataset.from_tensor_slices(y)
    yt = tf.data.Dataset.from_tensor_slices(yt)

    data = tf.data.Dataset.zip(((x, t), (y, yt)))
    data = data.batch(32)

    model = build_model()
    model.fit(data)

    model.save('/tmp/bert')
    m2 = tf.keras.models.load_model('/tmp/bert')


if __name__ == "__main__":
    test()
