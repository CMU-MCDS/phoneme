from __future__ import division
from __future__ import print_function

import sentencepiece as spm


spm.SentencePieceTrainer.Train('--input=/Users/yuhsianglin/Dropbox/cmu/phoneme/sample_data/ted-train.orig.spa --model_prefix=spa_eng.spa --vocab_size=32000 --character_coverage=1.0 --model_type=unigram')
