import csv
import pandas as pd
import os
import csv
import numpy as np
import random
from scipy.stats import rankdata
from sklearn.preprocessing import normalize
from collections import Counter
from collections import defaultdict
root = "/Users/yuhsianglin/Dropbox/cmu/11634A_11632A_capstone/20181029 Jupyter notebook"


### Load baseline experiment result: Get ranking ###
# mt.csv is Prof. Neubig's MT experiment results
baseline = pd.read_csv(os.path.join(root, "mt.csv"))
baseline = baseline.set_index("v extra v \ trg >>")

lang_set = baseline.columns.values
lang_num = 54
lang_set = lang_set[:lang_num]

with open("./language_set.txt", "w") as f:
    for lan in lang_set:
        print(lan, file=f)

table = [["Task lang","Aux lang","BLEU"]]
rank = ["Rank"]
BLEU_level = ["BLEU level"]

for lan_target in lang_set:
    rank_score = []
    for lan_train in lang_set:
        if lan_target == lan_train or baseline.loc[lan_train, lan_target] == 'X':
            continue
        table.append([str(lan_target), str(lan_train), float(baseline.loc[lan_train, lan_target])])
        # Note: use negative BLEU score here
        # scipy.stats.rankdata() gives rank [1, 2, 3, ...] to negative BLEU scores [-0.3, -0.2, -0.1],
        # so higher BLEU score is put in the front (small "rank" integer value) of the ranking result
        rank_score.append(-float(baseline.loc[lan_train, lan_target]))

    rank.extend(rankdata(rank_score, 'min'))
    BLEU_level.extend(rankdata(list(-np.array(rank_score)), 'min'))

table = np.column_stack((np.array(table), np.array(rank), np.array(BLEU_level)))

#####################

# mt_ttr, mt_overlap_word.csv, mt_overlap_subword.csv, mt_datasize.csv
# are the sheets in our feature spreadsheet.
extracted_root = root
ttr_table = pd.read_csv(os.path.join(extracted_root, "mt_ttr.csv"))
ttr_table = ttr_table.set_index("Lang")

overlap_word_table = pd.read_csv(os.path.join(extracted_root, "mt_overlap_word.csv"))
overlap_word_table = overlap_word_table.set_index("lang")

overlap_subword_table = pd.read_csv(os.path.join(extracted_root, "mt_overlap_subword.csv"))
overlap_subword_table = overlap_subword_table.set_index("lang")

datasize_table = pd.read_csv(os.path.join(extracted_root, "mt_datasize.csv"))
datasize_table = datasize_table.set_index("Language")

extracted_type = [["Aux lang TTR", "Overlap word-level", "Overlap subword-level", "Aux lang dataset size", "TTR difference ratio", "Dataset size ratio", "Task lang dataset size"]]


for i in range(1, table.shape[0]):
    lan_target, lan_train, _, _, _ = table[i]
    ttr = ttr_table.loc[lan_train].values[0]
    overlap_word = overlap_word_table.loc[lan_train, lan_target] / overlap_word_table.loc[lan_target, lan_target]
    overlap_subword = overlap_subword_table.loc[lan_train, lan_target] / overlap_subword_table.loc[lan_target, lan_target]
    datasize = datasize_table.loc[lan_train].values[0]
    ttr_target = ttr_table.loc[lan_target].values[0]
    ttr_diff = (ttr - ttr_target) / ttr_target
    datasize_target = datasize_table.loc[lan_target].values[0]
    datasize_ratio = datasize / datasize_target
    extracted_type.append([ttr, overlap_word, overlap_subword, datasize, ttr_diff, datasize_ratio, datasize_target])

table = np.column_stack((table, np.array(extracted_type)))

# Write out raw dataset for ranking
with open(os.path.join(root, "data_ranking_mt.csv"), "w") as f:
    for row in table:
        print(",".join(row), file=f)
