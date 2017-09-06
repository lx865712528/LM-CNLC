from server import LanguageCorrector

me = LanguageCorrector(fw_hyp_path="data/normal_hyperparams.pkl",
                       bw_hyp_path="data/reverse_hyperparams.pkl",
                       fw_vocab_path="data/normal_vocab.pkl",
                       bw_vocab_path="data/reverse_vocab.pkl",
                       dictionary_path="data/voc.txt")

print(me.correctify("人门都要遵守刑去"))
