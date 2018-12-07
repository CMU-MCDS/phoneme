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

# Working directory on clio
root = "/home/yuhsianl/public/phoneme_common_data/data/el"

# Conversion mapping between ISO 639 1 and 3
iso_mapping_file = 'ISO_639_1_TO_3.txt'
iso_mapping_dict = dict()
for pair in np.loadtxt(os.path.join(root, iso_mapping_file), dtype=str, delimiter="\n"):
    pair = pair.split(":")
    iso_mapping_dict[pair[0]] = pair[1]

def convert(code):
    return iso_mapping_dict[code] if code in iso_mapping_dict else code

if __name__ == "__main__":
    ### Load baseline experiment result: Get ranking ###
    # EL_PanPhon.csv is Shruti's Panphon Entity Linking experiment results
    baseline = pd.read_csv(os.path.join(root, "EL_PanPhon.csv"))
    baseline = baseline.set_index("v pivot v \\ test >>")

    # Target language set and transfer language set for the entity linking experiment
    tg_lang_set = baseline.columns.values
    tf_lang_set = baseline.index.values

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
            if lan_tg == lan_tf or baseline.loc[lan_tf, lan_tg] == 'xxx':
                continue
            table.append([str(lan_tg), str(lan_tf), float(baseline.loc[lan_tf, lan_tg])])
            # Note: use negative Accuracy score here
            # scipy.stats.rankdata() gives rank [1, 2, 3, ...] to negative Accuracy scores [-0.3, -0.2, -0.1],
            # so higher Accuracy score is put in the front (small "rank" integer value) of the ranking result
            rank_score.append(-float(baseline.loc[lan_tf, lan_tg]))

        rank.extend(rankdata(rank_score, 'min'))
        accuracy_level.extend(rankdata(list(-np.array(rank_score)), 'max'))

    table = np.column_stack((np.array(table), np.array(rank), np.array(accuracy_level)))
    
    
    #####################
    # entity_match.csv is our feature spreadsheet of the counts of overlapping entity between two languages
    entity_match_table = pd.read_csv(os.path.join(root, "entity_match.csv"))
    entity_match_table = entity_match_table.set_index("lang")

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

    extracted_type = [["Entity overlap", "Transfer lang dataset size", "Target lang dataset size", "Transfer over target size ratio"] + distance_category]
    

    for i in range(1, table.shape[0]):
        lan_tg, lan_tf, _, _, _ = table[i]

        entity_overlap = entity_match_table.loc[lan_tf, lan_tg] / entity_match_table.loc[lan_tg, lan_tg]
        # The datasize should be the number of overlapping entities between the language and itself
        datasize_tf = entity_match_table.loc[lan_tf, lan_tf]
        datasize_tg = entity_match_table.loc[lan_tg, lan_tg]
        datasize_tf_to_tg_ratio = datasize_tf / datasize_tg
        distance_list = [dtable.loc[convert(lan_tf), convert(lan_tg)] for dtable in distance_tables]

        extracted_type.append([entity_overlap, datasize_tf, datasize_tg, datasize_tf_to_tg_ratio] + distance_list)


    table = np.column_stack((table, np.array(extracted_type)))

    # Write out raw dataset for ranking
    with open(os.path.join(root, "data_ranking_el.csv"), "w") as f:
        for row in table:
            print(",".join(list(row)), file=f)
