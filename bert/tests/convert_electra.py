"""Test load official model, print difference."""

import os
import argparse
import json
import tensorflow as tf

from bert import BertModel, params
from bert.tests.convert_official import BERT, load_vocab


def load_model(model_path, num_hidden_layers=None):
    ckpt_reader = tf.train.load_checkpoint(
        os.path.join(model_path, 'bert_model.ckpt'))
    config = json.load(open(os.path.join(model_path, 'bert_config.json')))

    loaded_params = {k: config[k] for k in params.keys() if k in config}
    if 'embedding_size' in config:
        loaded_params['embedding_size'] = config['embedding_size']
    # import pdb; pdb.set_trace()
    if num_hidden_layers is not None and num_hidden_layers > 0:
        loaded_params['num_hidden_layers'] = num_hidden_layers

    tfbert = BertModel(**loaded_params)
    tfbert([
        tf.constant([[1]]),
        tf.constant([[1]]),
        tf.constant([[1]])
    ])

    def convert_official_name(x):
        x = x.replace('electra/encoder', 'bert/encoder')
        x = x.replace('discriminator_predictions/dense/kernel',
                      'bert/pooler/dense/kernel')
        x = x.replace('discriminator_predictions/dense/bias',
                      'bert/pooler/dense/bias')
        x = x.replace('electra/embeddings_project/kernel',
                      'bert/encoder/embedding_hidden_mapping_in/kernel')
        x = x.replace('electra/embeddings_project/bias',
                      'bert/encoder/embedding_hidden_mapping_in/bias')
        x = x.replace('electra/embeddings', 'bert/embeddings')
        return x

    skip_tensor = [
        'discriminator_predictions/dense_1/kernel',
        'discriminator_predictions/dense_1/bias',
        'cls/seq_relationship/output_bias',
        'cls/predictions/output_bias',
        'cls/predictions/transform/dense/kernel',
        'cls/predictions/transform/LayerNorm/beta',
        'cls/predictions/transform/LayerNorm/gamma',
        'cls/predictions/transform/dense/bias',
        'cls/seq_relationship/output_weights',
    ]

    tfbert_weights = {
        w.name: w
        for w in tfbert.weights if 'generator' not in w.name
    }
    official_weights = {
        convert_official_name(k): ckpt_reader.get_tensor(k)
        for k in ckpt_reader.get_variable_to_dtype_map().keys()
    }

    good = True
    our_keys = set([x.split(':')[0] for x in tfbert_weights.keys()])
    for x in set(official_weights.keys()) - our_keys:
        if 'adam' not in x and 'global_step' not in x and 'generator' not in x:
            if x not in skip_tensor:
                print('diff offi', x, official_weights[x].shape)
                good = False

    for x in our_keys - set(official_weights.keys()):
        if 'adam' not in x and 'global_step' not in x and 'generator' not in x:
            if x not in skip_tensor:
                print('diff ours', x, tfbert_weights[x + ':0'].shape)
                good = False

    assert good

    weight_tuples = []
    for k, v in tfbert_weights.items():
        name = k[:-2]
        if name in skip_tensor:
            continue
        else:
            off_tensor = None
            for ok in ckpt_reader.get_variable_to_dtype_map().keys():
                if convert_official_name(ok) == name:
                    off_tensor = ckpt_reader.get_tensor(ok)
            if off_tensor is not None:
                weight_tuples.append((v, off_tensor))
                assert v.shape == off_tensor.shape, \
                    f'{name} shape invalid {v.shape}, {off_tensor.shape}'
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
    parser.add_argument('--num_hidden_layers',
                        type=int,
                        default=0,
                        help='num_hidden_layers to keep, default all (0)')
    args = parser.parse_args()

    model_path = args.input
    word_index = load_vocab(os.path.join(model_path, 'vocab.txt'))
    bert = BERT(model_path, word_index, load_model, args.num_hidden_layers)

    print('save to', args.output)

    strs = tf.TensorSpec(shape=[None, None],
                         dtype=tf.string,
                         name="input_strs")

    bert._set_inputs(strs)
    bert.save(args.output)


if __name__ == "__main__":
    main()
