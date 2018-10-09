from __future__ import division
from __future__ import print_function

import sentencepiece as spm


sp = spm.SentencePieceProcessor()
sp.Load('spa_eng.spa.model')

token_count = 0
token_set = set()
#skip_tokens = set({'<unk>', '<s>', '</s>'})

with open('/Users/yuhsianglin/Dropbox/cmu/phoneme/sample_data/ted-train.orig.spa') as file:
    #cnt = 0
    for line in file:
        #cnt += 1
        #if cnt == 10:
        #    break
        #print(line)
        tokens = sp.EncodeAsPieces(line)
        #print(tokens)
        for t in tokens:
            #if t not in skip_tokens:
            token_count += 1
            token_set.add(t)

distinct_token_count = len(token_set)
ttr = distinct_token_count / token_count

print ('distinct_token_count =', distinct_token_count)
print ('token_count =', token_count)
print ('TTR =', ttr)

#print(token_set)
