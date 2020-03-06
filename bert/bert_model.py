
import tensorflow as tf
from .bert import Bert as BertLayer


def BertModel(**kwargs):
    input_ids = tf.keras.layers.Input(shape=(None, ), dtype=tf.int32)
    input_ids_type = tf.keras.layers.Input(shape=(None, ), dtype=tf.int32)
    pred, pool = BertLayer(**kwargs)([input_ids, input_ids_type], pooler=True)
    model = tf.keras.Model(
        inputs=[input_ids, input_ids_type],
        outputs=[pred, pool])
    return model
