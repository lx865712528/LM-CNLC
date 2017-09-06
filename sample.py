# -*- coding: utf-8 -*-
from __future__ import print_function

import os
import pickle
import sys

import tensorflow as tf

from models.charrnn import CharRNN
from utils import TextLoader, UNK_ID


def main(_):
    if len(sys.argv) < 2:
        print("Please enter a prime")
        sys.exit()

    prime = sys.argv[1]
    prime = prime.decode('utf-8')

    with open("./log/hyperparams.pkl", 'rb') as f:
        config = pickle.load(f)

    if not os.path.exists(config['checkpoint_dir']):
        print(" [*] Creating checkpoint directory...")
        os.makedirs(config['checkpoint_dir'])

    data_loader = TextLoader(os.path.join(config['data_dir'], config['dataset_name']),
                             config['batch_size'], config['seq_length'])
    vocab_size = data_loader.vocab_size

    with tf.variable_scope('model'):
        model = CharRNN(vocab_size, 1, config['rnn_size'],
                        config['layer_depth'], config['num_units'],
                        1, config['keep_prob'],
                        config['grad_clip'],
                        is_training=False)

    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(config['checkpoint_dir'] + '/' + config['dataset_name'])
        tf.train.Saver().restore(sess, ckpt.model_checkpoint_path)

        res = model.sample(sess, data_loader.chars, data_loader.vocab, UNK_ID, 100, prime)

        print(res)


if __name__ == '__main__':
    tf.app.run()
