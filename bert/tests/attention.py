
import tensorflow as tf
from bert.attention import Attention


hidden_size = 768
num_attention_heads = 12
size_per_head = hidden_size // num_attention_heads
initializer_range = 0.02
attention_probs_dropout_prob = 0.1


atten = Attention(num_attention_heads=num_attention_heads,
                  size_per_head=size_per_head,
                  initializer_range=initializer_range,
                  attention_probs_dropout_prob=attention_probs_dropout_prob)


batch_size = 32
max_length = 30

x = tf.random.uniform((batch_size, max_length, hidden_size),
                      minval=0,
                      maxval=1,
                      dtype=tf.float32)

assert atten(x).shape == (batch_size, max_length, hidden_size)
