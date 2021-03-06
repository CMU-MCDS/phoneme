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
import re

def get_lan_code(label):
    return label.split("(")[0]

def convert(code):
    if code in iso_mapping_dict:
        return iso_mapping_dict[code]
    elif len(code) != 3:
        print(code + ' not in the mapping')
        assert(False)
    return code

if __name__ == "__main__":
    # Working directory on clio
    root = "/home/yuhsianl/public/phoneme_common_data/data/pos"

    # Conversion mapping between ISO 639 1 and 3
    iso_mapping_file = 'ISO_639_1_TO_3.txt'
    iso_mapping_dict = dict()
    for pair in np.loadtxt(os.path.join(root, iso_mapping_file), dtype=str, delimiter="\n"):
        pair = pair.split(":")
        iso_mapping_dict[pair[0]] = pair[1]

    ### Load baseline experiment result: Get ranking ###
    # mt.csv is Prof. Neubig's MT experiment results
    baseline = pd.read_csv(os.path.join(root, "pos_baseline.csv"))
    baseline = baseline.set_index(baseline.columns[0])

    # Task language set and auxiliary language set for the entity linking experiment
    task_lang_set = baseline.columns.values
    aux_lang_set = baseline.index.values

    with open(os.path.join(root, "task_language_set.txt"), "w") as f:
        for lan in task_lang_set:
            lan_ = lan.split("(")
            datasize = int(lan_[1][:-1])
            if(datasize < 1400):
                print(lan, file=f)

    with open(os.path.join(root, "aux_language_set.txt"), "w") as f:
        for lan in aux_lang_set:
            lan_ = lan.split("(")
            datasize = int(lan_[1][:-1])
            if(datasize > 0):
                print(lan, file=f)


    table = [["Task lang", "Aux lang", "Accuracy"]]
    rank = ["Rank"]
    accuracy_level = ["Accuracy level"]

    task_lang_set = np.loadtxt(os.path.join(root, "task_language_set.txt"), dtype=str, delimiter="\n")
    aux_lang_set = np.loadtxt(os.path.join(root, "aux_language_set.txt"), dtype=str, delimiter="\n")

    for lan_task in task_lang_set:
        rank_score = []
        for lan_aux in aux_lang_set:
            if lan_task == lan_aux or baseline.loc[lan_aux, lan_task] == '' or type(baseline.loc[lan_aux, lan_task]) != np.float64:
                continue
            table.append([get_lan_code(str(lan_task)),get_lan_code(str(lan_aux)), float(baseline.loc[lan_aux, lan_task])])
            # Note: use negative BLEU score here
            # scipy.stats.rankdata() gives rank [1, 2, 3, ...] to negative BLEU scores [-0.3, -0.2, -0.1],
            # so higher BLEU score is put in the front (small "rank" integer value) of the ranking result
            rank_score.append(-float(baseline.loc[lan_aux, lan_task]))

        rank.extend(rankdata(rank_score, 'min'))
        accuracy_level.extend(rankdata(list(-np.array(rank_score)), 'max'))

    table = np.column_stack((np.array(table), np.array(rank), np.array(accuracy_level)))

    #####################

    # mt_ttr, mt_overlap_word.csv, mt_overlap_subword.csv, mt_datasize.csv
    # are the sheets in our feature spreadsheet.
    extracted_root = root
    ttr_table = pd.read_csv(os.path.join(extracted_root, "ttr_pos.csv"))
    ttr_table = ttr_table.set_index("lang")

    overlap_word_table = pd.read_csv(os.path.join(extracted_root, "overlap_pos.csv"))
    overlap_word_table = overlap_word_table.set_index("lang")

    datasize_table = pd.read_csv(os.path.join(extracted_root, "datasize_pos.csv"))
    datasize_table = datasize_table.set_index("Language")

    # URIEL distance
    distance_root = "/home/yuhsianl/public/phoneme_common_data/data/uriel_v0_2/distances/"
    distance_category = [fname.split(".")[0] for fname in os.listdir(distance_root) if fname.split(".")[1] == "csv"]
    distance_tables = []

    for category in distance_category:
        print(category)
        dist = pd.read_csv(os.path.join(distance_root, category + ".csv"))
        dist = dist.set_index("G_CODE")
        distance_tables.append(dist)

    #####################

    extracted_type = [["Overlap word-level", "Transfer lang dataset size", "Target lang dataset size", "Transfer over target size ratio", "Transfer lang TTR", "Target lang TTR", "Transfer target TTR distance"] + distance_category]

    for i in range(1, table.shape[0]):
        lan_tg, lan_tf, _, _, _ = table[i]
        overlap_word = overlap_word_table.loc[lan_tf, lan_tg] / overlap_word_table.loc[lan_tg, lan_tg]
        datasize_tf = datasize_table.loc[lan_tf].values[0]
        datasize_tg = datasize_table.loc[lan_tg].values[0]
        datasize_tf_to_tg_ratio = datasize_tf / datasize_tg
        ttr_tf = ttr_table.loc[lan_tf].values[0]
        ttr_tg = ttr_table.loc[lan_tg].values[0]
        ttr_distance = (1 - ttr_tf / ttr_tg) ** 2
        distance_list = [dtable.loc[convert(lan_tf), convert(lan_tg)] for dtable in distance_tables]

        extracted_type.append([overlap_word, datasize_tf, datasize_tg, datasize_tf_to_tg_ratio, ttr_tf, ttr_tg, ttr_distance] + distance_list)

    table = np.column_stack((table, np.array(extracted_type)))

    # Write out raw dataset for ranking
    with open(os.path.join(root, "data_ranking_pos.csv"), "w") as f:
        for row in table:
            print(",".join(row), file=f)
