# -*- coding: utf-8 -*-

import pickle

import jieba
import tensorflow as tf

from models.charrnn import CharRNN
from utils import GO, EOS, UNK_ID


class LanguageCorrector():
    '''
    Natural Language Correction Model
    '''

    def __init__(self, fw_hyp_path, bw_hyp_path, fw_vocab_path, bw_vocab_path, dictionary_path, threshold=4):
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

        self.threshold = 4

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

        # load models
        with tf.variable_scope(fw_hyp_config['dataset_name']):
            self.fw_model = CharRNN(self.fw_vocab_size, 1, fw_hyp_config['rnn_size'], fw_hyp_config['layer_depth'],
                                    fw_hyp_config['num_units'], 1, fw_hyp_config['keep_prob'],
                                    fw_hyp_config['grad_clip'])
        with tf.Session() as self.fw_lm_sess:
            ckpt = tf.train.get_checkpoint_state(fw_hyp_config['checkpoint_dir'] + '/' + fw_hyp_config['dataset_name'])
            tf.train.Saver().restore(self.fw_lm_sess, ckpt.model_checkpoint_path)
        with tf.variable_scope(bw_hyp_config['dataset_name']):
            self.bw_model = CharRNN(self.bw_vocab_size, 1, bw_hyp_config['rnn_size'], bw_hyp_config['layer_depth'],
                                    bw_hyp_config['num_units'], 1, bw_hyp_config['keep_prob'],
                                    bw_hyp_config['grad_clip'])
        with tf.Session() as self.bw_lm_sess:
            ckpt = tf.train.get_checkpoint_state(bw_hyp_config['checkpoint_dir'] + '/' + bw_hyp_config['dataset_name'])
            tf.train.Saver().restore(self.bw_lm_sess, ckpt.model_checkpoint_path)

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
        sz = len(chars)

        fw_ints = [self.fw_vocab_c2i.get(c, UNK_ID) for c in chars]
        bw_ints = [self.bw_vocab_c2i.get(c, UNK_ID) for c in chars]
        fw_probs = []  # 过一个字符后的概率
        bw_probs = []
        fw_losses = []  # 过一个字符后的loss
        bw_losses = []
        fw_states = []  # 过一个字符后的状态
        bw_states = []

        # find bad guys
        bads_or_not = []
        bad_pos = set()
        for i in range(sz):
            bads_or_not.append(False)
        # fw side
        fw_cnt_state = self.fw_model.cell.zero_state(1, tf.float32)
        for i in range(sz):
            res = self.fw_model.go_step(self.fw_lm_sess, fw_cnt_state, fw_ints[i])
            fw_probs.append(res["probs"])
            fw_losses.append(res["loss"])
            fw_states.append(res["state"])
            fw_cnt_state = res["state"]
        # bw side
        bw_cnt_state = self.bw_model.cell.zero_state(1, tf.float32)
        for i in range(sz):
            res = self.bw_model.go_step(self.bw_lm_sess, bw_cnt_state, bw_ints[i])
            bw_probs.append(res["probs"])
            bw_losses.append(res["loss"])
            bw_states.append(res["state"])
            bw_cnt_state = res["state"]
        # accumulate loss in each steps
        for i in range(1, sz):
            fw_losses[i] += fw_losses[i - 1]
            bw_losses[i] += bw_losses[i - 1]

        # first view
        for i in range(0, sz - 2):
            t_loss = fw_losses[i] + bw_losses[sz - 3 - i]
            if t_loss > self.threshold:
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
            left_probs_dict = {}
            for ch, prob in zip(self.fw_vocab_i2c, fw_probs[p - 1]):
                left_probs_dict[ch] = prob
            right_probs_dict = {}
            for ch, prob in zip(self.bw_vocab_i2c, bw_probs[sz - 3 - p + 1]):
                right_probs_dict[ch] = prob
            best_ch = ""
            best_score = -1
            for ch in self.fw_vocab_i2c:
                prob = left_probs_dict[ch] * right_probs_dict[ch]
                if prob > best_score + 1e-6:
                    best_score = prob
                    best_ch = ch
            chars[p] = best_ch
        chars = str(chars[1:-1])
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
