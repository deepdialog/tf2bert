"""Test load official model, print difference."""

import tensorflow as tf
import tensorflow_hub as hub

from bert import BertModel, params

tfbert = BertModel(**params)

model_path = '../bert-embs/hub/chinese_L-12_H-768_A-12/'
bert_layer = hub.KerasLayer(model_path,
                            signature="tokens",
                            signature_outputs_as_dict=True,
                            trainable=False)

assert len(tfbert.weights) == len(bert_layer.weights)


weight_of_tfbert = set([x.name for x in tfbert.weights])
weight_of_official = set([x.name for x in bert_layer.weights])

fit_weight = len(weight_of_tfbert & weight_of_tfbert)
assert fit_weight == len(bert_layer.weights)


def get_name_values(model):
    names = [x.name for x in model.weights]
    values = tf.keras.backend.batch_get_value(model.weights)
    return dict(zip(names, values))


official_weights = get_name_values(bert_layer)
tfbert_weights = {w.name: w for w in tfbert.weights}

weight_tuples = []
for name, value in tfbert_weights.items():
    weight_tuples.append((value, official_weights[name]))

tf.keras.backend.batch_set_value(weight_tuples)


def test(ids, ids_types, mask):
    official_ret = bert_layer({
        'token_type_ids': ids_types,
        'input_mask': mask,
        'input_ids': ids
    })
    official_pool_ret = official_ret['pooled_output']
    official_seq_ret = official_ret['sequence_output']

    tfbert_ret = tfbert({
        'token_type_ids': ids_types,
        'input_mask': mask,
        'input_ids': ids
    })
    tfbert_pool_ret = tfbert_ret['pooled_output']
    tfbert_seq_ret = tfbert_ret['sequence_output']

    print(
        'diff',
        tf.reduce_sum(tf.pow(official_pool_ret - tfbert_pool_ret, 2)).numpy())

    print(
        'diff',
        tf.reduce_sum(tf.pow(official_seq_ret - tfbert_seq_ret, 2)).numpy())


ids = tf.constant([
    [3, 4, 5],
    [100, 400, 0]
], dtype=tf.int32)
ids_types = tf.constant([
    [0, 0, 0],
    [0, 0, 0]
], dtype=tf.int32)
mask = tf.constant([
    [1, 1, 1],
    [1, 1, 1]
], dtype=tf.int32)

test(ids, ids_types, mask)


ids = tf.constant([
    [3, 4, 5],
    [100, 400, 100]
], dtype=tf.int32)
ids_types = tf.constant([
    [0, 0, 0],
    [0, 0, 0]
], dtype=tf.int32)
mask = tf.constant([
    [1, 1, 1],
    [1, 1, 1]
], dtype=tf.int32)

test(ids, ids_types, mask)


ids = tf.constant([
    [3, 4, 5],
    [100, 400, 0]
], dtype=tf.int32)
ids_types = tf.constant([
    [0, 0, 0],
    [0, 0, 0]
], dtype=tf.int32)
mask = tf.constant([
    [1, 1, 1],
    [1, 1, 0]
], dtype=tf.int32)

test(ids, ids_types, mask)


ids = tf.constant([
    [3, 4, 5],
    [100, 400, 100]
], dtype=tf.int32)
ids_types = tf.constant([
    [0, 0, 0],
    [0, 0, 0]
], dtype=tf.int32)
mask = tf.constant([
    [1, 1, 1],
    [1, 1, 0]
], dtype=tf.int32)

test(ids, ids_types, mask)
