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

    def __init__(self, fw_hyp_path, bw_hyp_path, fw_vocab_path, bw_vocab_path, dictionary_path, threshold=-20):
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
        self.threshold = threshold

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
                tf.global_variables_initializer().run()
                ckpt = tf.train.get_checkpoint_state(
                    bw_hyp_config['checkpoint_dir'] + '/' + bw_hyp_config['dataset_name'])
                saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path + ".meta", clear_devices=True)
                saver.restore(self.fw_sess, ckpt.model_checkpoint_path)
        print("fwmodel done!")

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
                tf.global_variables_initializer().run()
                ckpt = tf.train.get_checkpoint_state(
                    fw_hyp_config['checkpoint_dir'] + '/' + fw_hyp_config['dataset_name'])
                saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path + ".meta", clear_devices=True)
                saver.restore(self.bw_sess, ckpt.model_checkpoint_path)
        print("bwmodel done!")

        # load dictionary
        with open(dictionary_path, "r", encoding="utf-8") as f:
            self.dictionary = set()
            self.word_max_length = 0
            for line in f:
                line = line.strip()
                if len(line) == 0:
                    continue
                self.dictionary.add(line)
                if self.word_max_length < len(line):
                    self.word_max_length = len(line)

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
        bw_ints = [self.bw_vocab_c2i.get(c, UNK_ID) for c in chars[::-1]]
        fw_probs = []  # 过一个字符后的概率
        bw_probs = []
        fw_losses = []  # 过一个字符后的loss
        bw_losses = []

        # find bad guys
        bads_or_not = []
        bad_pos = set()
        for i in range(sz):
            bads_or_not.append(False)
        # fw side
        with self.fw_sess.as_default():
            with self.fw_sess.graph.as_default():
                fw_cnt_state = self.fw_sess.run(self.fw_model.cell.zero_state(1, tf.float32))
                for i in range(sz):
                    res = self.fw_model.go_step(self.fw_sess, fw_cnt_state, fw_ints[i])
                    fw_probs.append(res["probs"])
                    if i + 1 < sz:
                        fw_losses.append(math.log(res["probs"][fw_ints[i + 1]]))
                        print("%s --> %s: %lf" % (chars[i], chars[i + 1], res["probs"][fw_ints[i + 1]]))
                    fw_cnt_state = res["state"]
        # bw side
        with self.bw_sess.as_default():
            with self.bw_sess.graph.as_default():
                bw_cnt_state = self.bw_sess.run(self.bw_model.cell.zero_state(1, tf.float32))
                for i in range(sz):
                    res = self.bw_model.go_step(self.bw_sess, bw_cnt_state, bw_ints[i])
                    bw_probs.append(res["probs"])
                    if i + 1 < sz:
                        bw_losses.append(math.log(res["probs"][bw_ints[i + 1]]))
                        print("%s --> %s: %lf" % (bw_chars[i], bw_chars[i + 1], res["probs"][bw_ints[i + 1]]))
                    bw_cnt_state = res["state"]

        # accumulate loss in each steps
        for i in range(1, sz - 1):
            fw_losses[i] += fw_losses[i - 1]
            bw_losses[i] += bw_losses[i - 1]

        # first view
        for i in range(1, sz - 2):
            print(fw_losses[i], bw_losses[sz - 3 - i])
            t_loss = fw_losses[i] / (i) + bw_losses[sz - 3 - i] / (sz - i - 2)
            print(t_loss)
            if t_loss < self.threshold:
                bads_or_not[i + 1] = True
                bad_pos.add(i + 1)

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
                        bads_or_not[p] = False
                        bad_pos.remove(p)
                    if not bads_or_not[p]:
                        break
                if not bads_or_not[p]:
                    break

        # find candidates
        for p in bad_pos:
            if not (sz - 1 > p - 2 >= 0 and sz - 1 > sz - 3 - p >= 0):
                continue
            best_ch = ""
            best_score = np.NINF
            for ch in self.fw_vocab_i2c:
                if ch in START_VOCAB:
                    continue
                loss = (fw_losses[p - 2] + math.log(fw_probs[p - 1][self.fw_vocab_c2i[ch]])) / (p + 1) + \
                       (bw_losses[sz - 3 - p] + math.log(bw_probs[sz - 3 - p + 1][self.fw_vocab_c2i[ch]])) / (sz - p)
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
