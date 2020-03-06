import tensorflow as tf
import tensorflow_addons as tfa
from bert.pred import Pred
from bert.pool import Pool
from .config import params_small as params
from bert.bert_model import BertModel


def loss(one_hot_labels, log_probs):
    per_example_loss = -tf.reduce_sum(log_probs * one_hot_labels,
                                      axis=[-1])
    return tf.reduce_mean(per_example_loss)


def sparse_loss(labels, log_probs):
    labels = tf.cast(labels, tf.int32)
    one_hot_labels = tf.one_hot(labels, tf.shape(log_probs)[-1])
    return loss(one_hot_labels, log_probs)


def build_model(params):
    bert_model = BertModel(**params)

    input_ids = tf.keras.layers.Input(shape=(None, ), dtype=tf.int32)
    input_ids_type = tf.keras.layers.Input(shape=(None, ), dtype=tf.int32)

    x = input_ids
    t = input_ids_type
    x, p = bert_model([x, t])
    pool_out = Pool(type_vocab_size=params.get('type_vocab_size'))(p)
    pool_out = tf.keras.layers.Activation('linear', name='sen')(pool_out)

    pred_out = Pred(params.get('hidden_size'))(
        [x, bert_model.layers[-1].embedding.word_embeddings])
    pred_out = tf.keras.layers.Activation('linear', name='mlm')(pred_out)

    model = tf.keras.Model(inputs=[input_ids, input_ids_type],
                           outputs=[pred_out, pool_out])

    return model, bert_model


@tf.function
def train_step(model, optimizer, batch):
    batch_len = tf.constant(len(batch), dtype=tf.keras.backend.floatx())
    batch_grad = []
    accum_grad = {}
    for (x0, x1), (y0, y1) in batch:
        with tf.GradientTape() as tape0, tf.GradientTape() as tape1:
            p0, p1 = model([x0, x1], training=True)
            loss0 = sparse_loss(y0, p0)
            loss1 = sparse_loss(y1, p1)

        loss0 /= batch_len
        loss1 /= batch_len

        pred_var = [
            x for x in model.trainable_variables
            if 'pool' not in x.name
        ]
        pool_var = [
            x for x in model.trainable_variables
            if 'predictions' not in x.name
        ]
        grad0 = tape0.gradient(loss0, pred_var)
        grad1 = tape1.gradient(loss1, pool_var)
        for grad, var in zip(grad0 + grad1, pred_var + pool_var):
            if grad is None:
                continue
            name = var.name
            if 'embeddings' in name:
                batch_grad.append((grad, var))
            else:
                if name not in accum_grad:
                    accum_grad[name] = [grad, var]
                else:
                    accum_grad[name] = [
                        grad + accum_grad[name][0],
                        var
                    ]
    for x, y in accum_grad.values():
        batch_grad.append((x, y))
    optimizer.apply_gradients(batch_grad)
    return loss0, loss1


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
    data = data.batch(4)

    model, bert_model = build_model(params)

    schedule = tf.optimizers.schedules.PolynomialDecay(
        initial_learning_rate=1e-4, end_learning_rate=1e-6, decay_steps=25000)

    optimizer = tfa.optimizers.AdamW(
        weight_decay=1e-2,
        learning_rate=schedule,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-06,
    )

    batch = []
    for x in data:
        batch.append(x)
        if len(batch) == 4:
            loss0, loss1 = train_step(model, optimizer, batch)
            loss0 = loss0.numpy()
            loss1 = loss1.numpy()
            print(loss0, loss1)
            batch = []

    # model.fit(data)

    # print('save')
    # model.save('/tmp/model', include_optimizer=False)
    # bert_model.save('/tmp/bert', include_optimizer=False)
    # print('load')
    # m2 = tf.keras.models.load_model('/tmp/bert')
    # print(m2([xr[:1], tr[:1]]))


if __name__ == "__main__":
    test()
