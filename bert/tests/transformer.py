import tensorflow as tf

from bert.transformer import TransformerEncoder

params = {
    'num_hidden_layers': 12,
    'hidden_size': 768,
    'num_attention_heads': 12,
    'intermediate_size': 768 * 4,
    'hidden_act': 'gelu',
    'initializer_range': 0.02,
    'hidden_dropout_prob': 0.1,
    'attention_probs_dropout_prob': 0.1,
}

batch_size = 32
max_length = 20
hidden_size = params.get('hidden_size')
x = tf.random.uniform((batch_size, max_length, hidden_size),
                      minval=0,
                      maxval=1,
                      dtype=tf.float32)

tsa = TransformerEncoder(**params)
print(tsa(x).shape)
