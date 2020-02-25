import tensorflow as tf
from bert.bert import Bert
from bert.pred import Pred
from bert.pool import Pool
from .config import params_small as params


def build_model(params):
    input_ids = tf.keras.layers.Input(shape=(None, ), dtype=tf.int32)
    input_ids_type = tf.keras.layers.Input(shape=(None, ), dtype=tf.int32)
    bert_layer = Bert(**params)
    bert_pred, bert_pool = bert_layer([input_ids, input_ids_type], pooler=True)
    bert_model = tf.keras.Model(inputs=[input_ids, input_ids_type],
                                outputs=[bert_pred, bert_pool])

    input_ids = tf.keras.layers.Input(shape=(None, ), dtype=tf.int32)
    input_ids_type = tf.keras.layers.Input(shape=(None, ), dtype=tf.int32)
    x = input_ids
    t = input_ids_type
    x, p = bert_model([x, t])
    pool_out = Pool(type_vocab_size=params.get('type_vocab_size'))(p)
    pool_out = tf.keras.layers.Activation('linear', name='sen')(pool_out)

    pred_out = Pred(
        params.get('hidden_size'))([x, bert_layer.embedding.word_embeddings])
    pred_out = tf.keras.layers.Activation('linear', name='mlm')(pred_out)

    model = tf.keras.Model(inputs=[input_ids, input_ids_type],
                           outputs=[pred_out, pool_out])
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=tf.keras.optimizers.Adam(),
                  metrics=['acc'])
    return model, bert_model


def test():

    n_samples = 320
    max_length = 20

    xr = tf.random.uniform((n_samples, max_length),
                           minval=0,
                           maxval=params.get('vocab_size'),
                           dtype=tf.int32)
    tr = tf.random.uniform((n_samples, max_length),
                           minval=0,
                           maxval=params.get('type_vocab_size'),
                           dtype=tf.int32)

    yr = tf.random.uniform((n_samples, max_length),
                           minval=0,
                           maxval=params.get('vocab_size'),
                           dtype=tf.int32)

    ytr = tf.random.uniform((n_samples, ),
                            minval=0,
                            maxval=params.get('type_vocab_size'),
                            dtype=tf.int32)

    x = tf.data.Dataset.from_tensor_slices(xr)
    t = tf.data.Dataset.from_tensor_slices(tr)
    y = tf.data.Dataset.from_tensor_slices(yr)
    yt = tf.data.Dataset.from_tensor_slices(ytr)

    data = tf.data.Dataset.zip(((x, t), (y, yt)))
    data = data.batch(32)

    model, bert_model = build_model(params)
    model.fit(data)

    print('save')
    model.save('/tmp/model', include_optimizer=False)
    bert_model.save('/tmp/bert', include_optimizer=False)
    print('load')
    m2 = tf.keras.models.load_model('/tmp/bert')
    print(m2([xr[:1], tr[:1]]))


if __name__ == "__main__":
    test()
