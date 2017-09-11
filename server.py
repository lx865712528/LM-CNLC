# -*- coding: utf-8 -*-

import math
import pickle

import jieba
import numpy as np
import tensorflow as tf

from models.charrnn import CharRNN
from utils import GO, EOS, UNK_ID, START_VOCAB


class LanguageCorrector():
    '''
    Natural Language Correction Model
    '''

    def __init__(self, fw_hyp_path, bw_hyp_path, fw_vocab_path, bw_vocab_path, fw_model_path, bw_model_path,
                 dictionary_path, threshold=math.exp(-7.0)):
        '''
        Load solver
        :param fw_hyp_path: forward model hyperparam path
        :param bw_hyp_path: backward model hyperparam path
        :param fw_vocab_path: forward model vocab path
        :param bw_vocab_path: backward model vocab path
        :param dictionary_path: dictionary path
        :param threshold: threshold for model
        '''
        jieba.load_userdict(dictionary_path)

        self.threshold = np.log(threshold)
        print(self.threshold)

        # load configs
        with open(fw_hyp_path, 'rb') as f:
            fw_hyp_config = pickle.load(f)
        with open(bw_hyp_path, 'rb') as f:
            bw_hyp_config = pickle.load(f)

        # load vocabularys
        with open(fw_vocab_path, 'rb') as f:
            self.fw_vocab_i2c = pickle.load(f)
            self.fw_vocab_size = len(self.fw_vocab_i2c)
            self.fw_vocab_c2i = dict(zip(self.fw_vocab_i2c, range(self.fw_vocab_size)))
        with open(bw_vocab_path, 'rb') as f:
            self.bw_vocab_i2c = pickle.load(f)
            self.bw_vocab_size = len(self.bw_vocab_i2c)
            self.bw_vocab_c2i = dict(zip(self.bw_vocab_i2c, range(self.bw_vocab_size)))

        # load fwmodel
        g1 = tf.Graph()
        self.fw_sess = tf.Session(graph=g1)
        with self.fw_sess.as_default():
            with g1.as_default():
                with tf.variable_scope(fw_hyp_config['dataset_name']):
                    self.fw_model = CharRNN(vocab_size=self.fw_vocab_size,
                                            batch_size=1,
                                            rnn_size=fw_hyp_config['rnn_size'],
                                            layer_depth=fw_hyp_config['layer_depth'],
                                            num_units=fw_hyp_config['num_units'],
                                            seq_length=1,
                                            keep_prob=fw_hyp_config['keep_prob'],
                                            grad_clip=fw_hyp_config['grad_clip'],
                                            rnn_type=fw_hyp_config['rnn_type'])
                ckpt = tf.train.get_checkpoint_state(fw_model_path +
                                                     '/' +
                                                     fw_hyp_config['dataset_name'])
                tf.train.Saver().restore(self.fw_sess, ckpt.model_checkpoint_path)
        # print("fwmodel done!")

        # load bwmodel
        g2 = tf.Graph()
        self.bw_sess = tf.Session(graph=g2)
        with self.bw_sess.as_default():
            with g2.as_default():
                with tf.variable_scope(bw_hyp_config['dataset_name']):
                    self.bw_model = CharRNN(vocab_size=self.bw_vocab_size,
                                            batch_size=1,
                                            rnn_size=bw_hyp_config['rnn_size'],
                                            layer_depth=bw_hyp_config['layer_depth'],
                                            num_units=bw_hyp_config['num_units'],
                                            seq_length=1,
                                            keep_prob=bw_hyp_config['keep_prob'],
                                            grad_clip=bw_hyp_config['grad_clip'],
                                            rnn_type=bw_hyp_config['rnn_type'])
                ckpt = tf.train.get_checkpoint_state(bw_model_path +
                                                     '/' +
                                                     bw_hyp_config['dataset_name'])
                tf.train.Saver().restore(self.bw_sess, ckpt.model_checkpoint_path)
        # print("bwmodel done!")

        # load dictionary
        with open(dictionary_path, "r", encoding="utf-8") as f:
            self.dictionary = set()
            self.word_max_length = 0
            for line in f:
                line = line.strip()
                if len(line) == 0:
                    continue
                segs = line.split("\t")
                self.dictionary.add(segs[0])
                if self.word_max_length < len(segs[0]):
                    self.word_max_length = len(segs[0])

    def correctify(self, sentence):
        '''
        Requested Method
        :param sentence: input sentence
        :return: corrections
        '''
        chars = [GO] + list(sentence) + [EOS]
        bw_chars = chars[::-1]
        sz = len(chars)
        fw_ints = [self.fw_vocab_c2i.get(c, UNK_ID) for c in chars]
        bw_ints = [self.bw_vocab_c2i.get(c, UNK_ID) for c in bw_chars]

        fw_losses = {}  # 过一个字符后的loss
        bw_losses = {}
        fw_probs = {}
        bw_probs = {}

        # find bad guys
        bads_or_not = []
        bad_pos = set()
        for i in range(sz):
            bads_or_not.append(False)
        # fw side
        with self.fw_sess.as_default():
            with self.fw_sess.graph.as_default():
                for i in range(1, sz - 1):
                    fw_substr_ints = fw_ints[:i + 1]
                    fw_losses[i], fw_probs[i] = self.fw_model.get_loss(self.fw_sess, fw_substr_ints)

        # bw side
        with self.bw_sess.as_default():
            with self.bw_sess.graph.as_default():
                for i in range(1, sz - 1):
                    bw_substr_ints = bw_ints[:i + 1]
                    bw_losses[i], bw_probs[i] = self.bw_model.get_loss(self.bw_sess, bw_substr_ints)

        # first view
        results = []
        for i in range(1, sz - 1):
            # print(fw_losses[i], bw_losses[sz - 1 - i])
            t_loss = fw_losses[i] + bw_losses[sz - 1 - i]
            # print(t_loss)
            # print(chars[:i + 1])
            results.append([i, t_loss])
        results = list(sorted(results, key=lambda x: x[1]))
        for i in range((len(results) + 1) // 2):
            score = results[i][1]
            if score < self.threshold:
                pos = results[i][0]
                bads_or_not[pos] = True
                bad_pos.add(pos)

        # second view
        for p in range(sz):
            if not bads_or_not[p]:
                continue
            for word_len in range(2, self.word_max_length):
                left_p = max(1, p - word_len + 1)
                right_p = min(sz - 2 - word_len + 1, p)
                for left in range(left_p, right_p + 1):
                    subword = sentence[left - 1:left - 1 + word_len]
                    if subword in self.dictionary:
                        # print(subword)
                        bads_or_not[p] = False
                        bad_pos.remove(p)
                    if not bads_or_not[p]:
                        break
                if not bads_or_not[p]:
                    break
        # print(bad_pos)

        # find candidates
        for p in bad_pos:
            if not sz - 2 >= p >= 1:
                continue
            best_ch = ""
            best_score = np.NINF

            left_ints = fw_ints[:p]
            left_sz = len(left_ints)
            with self.fw_sess.as_default():
                with self.fw_sess.graph.as_default():
                    left_loss, left_probs = self.fw_model.get_loss(self.fw_sess, left_ints)

            right_ints = bw_ints[:sz - 1 - p]
            right_sz = len(right_ints)
            with self.bw_sess.as_default():
                with self.bw_sess.graph.as_default():
                    right_loss, right_probs = self.bw_model.get_loss(self.bw_sess, right_ints)

            for ic, ch in enumerate(self.fw_vocab_i2c):
                if ch in START_VOCAB:
                    continue

                loss = (left_loss * left_sz + math.log(left_probs[ic])) / (left_sz + 1) + \
                       (right_loss * right_sz + math.log(right_probs[self.bw_vocab_c2i[ch]])) / (right_sz + 1)

                if loss > best_score + 1e-6:
                    best_score = loss
                    best_ch = ch
            chars[p] = best_ch
        chars = "".join(chars[1:-1])
        assert len(chars) == len(sentence)
        segs = jieba.cut(chars)

        # get requested format
        ans_array = []
        a_p = 0
        for seg in segs:
            s_s = a_p
            t_t = s_s + len(seg)
            if chars[s_s:t_t] != sentence[s_s:t_t]:
                ans_array.append({
                    "sourceValue": sentence[s_s:t_t],
                    "correctValue": chars[s_s:t_t],
                    "startOffset": s_s,
                    "endOffset": t_t
                })
            a_p = t_t
        return ans_array
