import tensorflow as tf
from .gelu import gelu


def get_activation(activation_string):
    if not isinstance(activation_string, str):
        return activation_string
    if not activation_string:
        return None

    act = activation_string.lower()
    if act == "linear":
        return None
    elif act == "relu":
        return tf.nn.relu
    elif act == "gelu":
        return gelu
    elif act == "tanh":
        return tf.tanh
    else:
        raise ValueError("Unsupported activation: %s" % act)
