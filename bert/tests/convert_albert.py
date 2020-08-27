"""Test load official model, print difference."""

import os
import argparse
import json
import tensorflow as tf

from bert import BertModel, albert_params as params
from bert.tests.convert_official import BERT


def load_vocab(vocab_path):
    word_index = {}
    with open(vocab_path) as fp:
        for i, line in enumerate(fp):
            line = line.strip()  # .lower()
            word_index[line] = i
    word_index[''] = -1  # real padding
    # index_word = {v: k for k, v in word_index.items()}
    return word_index


def load_model(model_path, num_hidden_layers=None):
    ckpt_reader = tf.train.load_checkpoint(
        os.path.join(model_path, 'model.ckpt-best'))
    config_file = [x for x in os.listdir(model_path) if x.endswith('.json')][0]
    config = json.load(open(os.path.join(model_path, config_file)))

    loaded_params = {k: config[k] for k in params.keys()}

    tfbert = BertModel(shared_layer=True, **loaded_params)

    tfbert_weights = {w.name: w for w in tfbert.weights}
    official_weights = set(ckpt_reader.get_variable_to_dtype_map().keys())

    good = True
    for x in official_weights - set([x.split(':')[0]
                                     for x in tfbert_weights.keys()]):
        if 'adam' not in x and 'global_step' not in x:
            print('diff offi', x)
            good = False

    for x in set([x.split(':')[0]
                  for x in tfbert_weights.keys()]) - official_weights:
        if 'adam' not in x and 'global_step' not in x:
            print('diff ours', x)
            good = False

    assert good

    weight_tuples = []
    for k, v in tfbert_weights.items():
        name = k[:-2]
        if ckpt_reader.has_tensor(name):
            ckpt_value = ckpt_reader.get_tensor(name)
            weight_tuples.append((v, ckpt_value))
            assert v.shape == ckpt_value.shape, \
                f'{name} shape invalid {v.shape}, {ckpt_value.shape}'
        else:
            print(f'{name} weight not loaded')
    tf.keras.backend.batch_set_value(weight_tuples)
    return tfbert


def main():
    parser = argparse.ArgumentParser(
        description='convert official tf1 bert model to tf2')
    parser.add_argument('--input',
                        required=True,
                        type=str,
                        help='input tf1 bert dir')
    parser.add_argument('--output',
                        required=True,
                        type=str,
                        help='output tf2 bert dir')
    args = parser.parse_args()

    model_path = args.input
    word_index = load_vocab(os.path.join(model_path, 'vocab_chinese.txt'))
    bert = BERT(model_path, word_index, load_model)

    print('save to', args.output)

    strs = tf.TensorSpec(shape=[None, None],
                         dtype=tf.string,
                         name="input_strs")
    ids = tf.TensorSpec(shape=[None, None],
                        dtype=tf.int32,
                        name="input_ids")
    mask = tf.TensorSpec(shape=[None, None],
                         dtype=tf.int32,
                         name="input_mask")
    ttids = tf.TensorSpec(shape=[None, None],
                          dtype=tf.int32,
                          name="token_type_ids")

    tf.saved_model.save(bert, args.output, signatures={
        'serving_default': bert.call.get_concrete_function(
            strs
        ),

        'ids_mask_type': bert.call_ids_mask_type.get_concrete_function(
            ids, mask, ttids
        ),

        'ids_mask': bert.call_ids_mask.get_concrete_function(
            ids, mask
        ),

        'ids_type': bert.call_ids_type.get_concrete_function(
            ids, ttids
        ),

        'ids': bert.call_ids.get_concrete_function(
            ids
        ),

        'strs_mask_type': bert.call_strs_mask_type.get_concrete_function(
            strs, mask, ttids
        ),

        'strs_mask': bert.call_strs_mask.get_concrete_function(
            strs, mask
        ),

        'strs_type': bert.call_strs_type.get_concrete_function(
            strs, ttids
        ),

        'strs': bert.call_strs.get_concrete_function(
            strs
        ),
    })


if __name__ == "__main__":
    main()
