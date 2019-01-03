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
    root = "/home/yuhsianl/public/phoneme_common_data/data/parsing"
    # URIEL distance
    distance_root = "/home/yuhsianl/public/phoneme_common_data/data/uriel_v0_2/distances"

    # Working directory on your local machine
    # root = "/Users/yuhsianglin/Dropbox/cmu/phoneme_data/parsing"
    # distance_root = "/Users/yuhsianglin/Dropbox/cmu/phoneme_data/uriel_v0_2/distances"

    # Conversion mapping between ISO 639 1 and 3
    iso_mapping_file = 'ISO_639_1_TO_3.txt'
    iso_mapping_dict = dict()
    for pair in np.loadtxt(os.path.join(root, iso_mapping_file), dtype=str, delimiter="\n"):
        pair = pair.split(":")
        iso_mapping_dict[pair[0]] = pair[1]

    def convert(code):
        return iso_mapping_dict[code] if code in iso_mapping_dict else code

    ### Load baseline experiment result: Get ranking ###
    # mt.csv is Prof. Neubig's MT experiment results
    score_table = pd.read_csv(os.path.join(root, "parsing.csv"), index_col=0)

    tg_lang_set = list(score_table.columns.values)
    # Get rid of the additional rows
    TG_LANG_NUM = 30
    tg_lang_set = tg_lang_set[:TG_LANG_NUM]

    tf_lang_set = list(score_table.index.values)
    # Get rid of the additional rows
    TF_LANG_NUM = 30
    tf_lang_set = tf_lang_set[:TF_LANG_NUM]

    with open(os.path.join(root, "tg_language_set.txt"), "w") as f:
        for lan in tg_lang_set:
            print(lan, file=f)

    with open(os.path.join(root, "tf_language_set.txt"), "w") as f:
        for lan in tf_lang_set:
            print(lan, file=f)

    # Format the data table 
    table = [["Target lang", "Transfer lang", "Accuracy"]]
    rank = ["Rank"]
    accuracy_level = ["Accuracy level"]

    for lan_tg in tg_lang_set:
        rank_score = []
        for lan_tf in tf_lang_set:
            if lan_tg == lan_tf:
                continue
            table.append([str(lan_tg), str(lan_tf), float(score_table.loc[lan_tf, lan_tg])])
            # Note: use negative Accuracy score here
            # scipy.stats.rankdata() gives rank [1, 2, 3, ...] to negative Accuracy scores [-0.3, -0.2, -0.1],
            # so higher Accuracy score is put in the front (small "rank" integer value) of the ranking result
            rank_score.append(-float(score_table.loc[lan_tf, lan_tg]))

        rank.extend(rankdata(rank_score, 'min'))
        accuracy_level.extend(rankdata(list(-np.array(rank_score)), 'max'))

    table = np.column_stack((np.array(table), np.array(rank), np.array(accuracy_level)))
    
    #####################
    word_overlap_table = pd.read_csv(os.path.join(root, "word_overlap.csv"), index_col=0)

    tf_dataset_size = pd.read_csv(os.path.join(root, "tf_size.csv"))
    tg_dataset_size = pd.read_csv(os.path.join(root, "tg_size.csv"))

    tf_ttr_table = pd.read_csv(os.path.join(root, "tf_ttr.csv"))
    tg_ttr_table = pd.read_csv(os.path.join(root, "tg_ttr.csv"))


    # URIEL distance
    distance_category = [fname.split(".")[0] for fname in os.listdir(distance_root) if fname.split(".")[1] == "csv"]
    distance_tables = []

    for category in distance_category:
        print(category)
        dist = pd.read_csv(os.path.join(distance_root, category + ".csv"))
        dist = dist.set_index("G_CODE")
        distance_tables.append(dist)

    #####################

    extracted_type = [["Word overlap", "Transfer lang dataset size", "Target lang dataset size", "Transfer over target size ratio", "Transfer lang TTR", "Target lang TTR", "Transfer target TTR distance"] + distance_category]


    for i in range(1, table.shape[0]):
        lan_tg, lan_tf, _, _, _ = table[i]

        word_overlap = word_overlap_table.loc[lan_tf, lan_tg]
        datasize_tf = tf_dataset_size[lan_tf].values[0]
        datasize_tg = tg_dataset_size[lan_tg].values[0]
        datasize_tf_to_tg_ratio = datasize_tf / datasize_tg

        ttr_tf = tf_ttr_table[lan_tf].values[0]
        ttr_tg = tg_ttr_table[lan_tg].values[0]
        ttr_distance = (1 - ttr_tf / ttr_tg) ** 2

        distance_list = [dtable.loc[convert(lan_tf), convert(lan_tg)] for dtable in distance_tables]

        extracted_type.append([word_overlap, datasize_tf, datasize_tg, datasize_tf_to_tg_ratio, ttr_tf, ttr_tg, ttr_distance] + distance_list)

    extracted_type = np.array(extracted_type)
    table = np.concatenate((table, extracted_type), axis=1)

    # Write out raw dataset for ranking
    with open(os.path.join(root, "data_ranking_parsing.csv"), "w") as f:
        for row in table:
            print(",".join(list(row)), file=f)
