import numpy as np

from server import LanguageCorrector

me = LanguageCorrector(fw_hyp_path="data/normal_hyperparams.pkl",
                       bw_hyp_path="data/reverse_hyperparams.pkl",
                       fw_vocab_path="data/normal_vocab.pkl",
                       bw_vocab_path="data/reverse_vocab.pkl",
                       fw_model_path="checkpoint",
                       bw_model_path="checkpoint",
                       dictionary_path="data/voc.txt",
                       threshold=np.exp(-7.0))

# [{'sourceValue': '送大', 'correctValue': '送达', 'startOffset': 4, 'endOffset': 6}]
print(me.correctify("本裁定书送大后即发生法律效力"))
