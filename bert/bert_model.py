
import tensorflow as tf
from .bert import Bert as BertLayer


def BertModel(**kwargs):
    input_ids = tf.keras.layers.Input(
        shape=(None, ),
        name='input_ids',
        dtype=tf.int32
    )
    token_type_ids = tf.keras.layers.Input(
        shape=(None, ),
        name='token_type_ids',
        dtype=tf.int32
    )
    input_mask = tf.keras.layers.Input(
        shape=(None, ),
        name='input_mask',
        dtype=tf.int32
    )
    out = BertLayer(**kwargs)([input_ids, token_type_ids, input_mask])
    model = tf.keras.Model(
        inputs=[input_ids, token_type_ids, input_mask],
        outputs=out)
    return model
