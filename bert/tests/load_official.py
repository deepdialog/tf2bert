"""Test load official model, print difference."""

import tensorflow as tf
import tensorflow_hub as hub

from bert import BertModel, params

tfbert = BertModel(**params)

model_path = '../bert-embs/hub/chinese_L-12_H-768_A-12/'
pool_layer = hub.KerasLayer(model_path,
                            signature="tokens",
                            output_key='pooled_output',
                            trainable=False)
seq_layer = hub.KerasLayer(model_path,
                           signature="tokens",
                           output_key='sequence_output',
                           trainable=False)

assert len(pool_layer.weights) == len(seq_layer.weights)

assert len(tfbert.weights) == len(pool_layer.weights)

fit_weight = len(
    set([x.name
         for x in tfbert.weights]) & set([x.name for x in pool_layer.weights]))
assert fit_weight == len(pool_layer.weights)


def get_name_values(model):
    names = [x.name for x in model.weights]
    values = tf.keras.backend.batch_get_value(model.weights)
    return dict(zip(names, values))


official_weights = get_name_values(pool_layer)
tfbert_weights = {w.name: w for w in tfbert.weights}

weight_tuples = []
for name, value in tfbert_weights.items():
    weight_tuples.append((value, official_weights[name]))

tf.keras.backend.batch_set_value(weight_tuples)

ids = tf.constant([[3, 4, 5]], dtype=tf.int32)
ids_types = tf.constant([[0, 0, 0]], dtype=tf.int32)
mask = tf.constant([[1, 1, 1]], dtype=tf.int32)

official_pool_ret = pool_layer({
    'segment_ids': ids_types,
    'input_mask': mask,
    'input_ids': ids
})

tfbert_pool_ret = tfbert([ids, ids_types])[1]

print('diff',
      tf.reduce_sum(tf.pow(official_pool_ret - tfbert_pool_ret, 2)).numpy())

official_seq_ret = seq_layer({
    'segment_ids': ids_types,
    'input_mask': mask,
    'input_ids': ids
})

tfbert_seq_ret = tfbert([ids, ids_types])[0]

print('diff',
      tf.reduce_sum(tf.pow(official_seq_ret - tfbert_seq_ret, 2)).numpy())
