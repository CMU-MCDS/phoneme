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


if __name__ == "__main__":
    # Working directory on clio
    root = "/home/yuhsianl/public/phoneme_common_data/data/mt"

    # Working directory on your local machine
    # root = "/Users/yuhsianglin/Dropbox/cmu/11634A_11632A_capstone/20181029 Jupyter notebook"


    ### Load baseline experiment result: Get ranking ###
    # mt.csv is Prof. Neubig's MT experiment results
    baseline = pd.read_csv(os.path.join(root, "mt.csv"))
    baseline = baseline.set_index("v extra v \\ trg >>")

    lang_set = baseline.columns.values
    lang_num = 54
    lang_set = lang_set[:lang_num]

    with open(os.path.join(root, "language_set.txt"), "w") as f:
        for lan in lang_set:
            print(lan, file=f)

    table = [["Task lang", "Aux lang", "BLEU"]]
    rank = ["Rank"]
    BLEU_level = ["BLEU level"]

    for lan_task in lang_set:
        rank_score = []
        for lan_aux in lang_set:
            if lan_task == lan_aux or baseline.loc[lan_aux, lan_task] == 'X':
                continue
            table.append([str(lan_task), str(lan_aux), float(baseline.loc[lan_aux, lan_task])])
            # Note: use negative BLEU score here
            # scipy.stats.rankdata() gives rank [1, 2, 3, ...] to negative BLEU scores [-0.3, -0.2, -0.1],
            # so higher BLEU score is put in the front (small "rank" integer value) of the ranking result
            rank_score.append(-float(baseline.loc[lan_aux, lan_task]))

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

    # URIEL distance
    distance_root = "/home/yuhsianl/public/phoneme_common_data/data/uriel_v0_2/distances"
    distance_category = [fname.split(".")[0] for fname in os.listdir(distance_root) if fname.split(".")[1] == "csv"]
    distance_tables = []

    for category in distance_category:
        dist = pd.read_csv(os.path.join(distance_root, category + ".csv"))
        dist = dist.set_index("G_CODE")
        dist_sub = dist[dist.columns.intersection(lang_set)]
        dist_sub = dist_sub[dist_sub.index.isin(lang_set)]
        distance_tables.append(dist_sub)

    #####################

    extracted_type = [["Aux lang TTR", "Overlap word-level", "Overlap subword-level", "Aux lang dataset size", "TTR difference ratio", "Dataset size ratio", "Task lang dataset size"] + distance_category]

    for i in range(1, table.shape[0]):
        lan_task, lan_aux, _, _, _ = table[i]
        ttr = ttr_table.loc[lan_aux].values[0]
        overlap_word = overlap_word_table.loc[lan_aux, lan_task] / overlap_word_table.loc[lan_task, lan_task]
        overlap_subword = overlap_subword_table.loc[lan_aux, lan_task] / overlap_subword_table.loc[lan_task, lan_task]
        datasize = datasize_table.loc[lan_aux].values[0]
        ttr_target = ttr_table.loc[lan_task].values[0]
        ttr_diff = (ttr - ttr_target) / ttr_target
        datasize_target = datasize_table.loc[lan_task].values[0]
        datasize_ratio = datasize / datasize_target
        distance_list = [dtable.loc[lan_aux, lan_task] for dtable in distance_tables]

        extracted_type.append([ttr, overlap_word, overlap_subword, datasize, ttr_diff, datasize_ratio, datasize_target] + distance_list)

    table = np.column_stack((table, np.array(extracted_type)))

    # Write out raw dataset for ranking
    with open(os.path.join(root, "data_ranking_mt.csv"), "w") as f:
        for row in table:
            print(",".join(row), file=f)
