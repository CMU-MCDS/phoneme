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
root = "/home/yuhsianl/public/phoneme_common_data/data/el/"

# Conversion mapping between ISO 639 1 and 3
iso_mapping_file = 'ISO_639_1_TO_3.txt'
iso_mapping_dict = dict()
for pair in np.loadtxt(os.path.join(root, iso_mapping_file), dtype=str, delimiter="\n"):
    pair = pair.split(":")
    iso_mapping_dict[pair[0]] = pair[1]


def convert(code):
    if code in iso_mapping_dict:
        return iso_mapping_dict[code]
    return code


if __name__ == "__main__":
    ### Load baseline experiment result: Get ranking ###
    # EL_PanPhon.csv is Shruti's Panphon Entity Linking experiment results
    baseline = pd.read_csv(os.path.join(root, "POS_tagging.csv"))
    baseline = baseline.set_index("v pivot v \ test >>")

    # Task language set and auxiliary language set for the entity linking experiment
    task_lang_set = baseline.columns.values
    aux_lang_set = baseline.index.values

    with open(os.path.join(root, "task_language_set.txt"), "w") as f:
        for lan in task_lang_set:
            print(lan, file=f)

    with open(os.path.join(root, "aux_language_set.txt"), "w") as f:
        for lan in aux_lang_set:
            print(lan, file=f)

    # Format the data table
    table = [["Task lang", "Aux lang", "Accuracy"]]
    rank = ["Rank"]
    accuracy_level = ["Accuracy level"]

    # for lan_task in task_lang_set:
    #     rank_score = []
    #     for lan_aux in aux_lang_set:
    #         if lan_task == lan_aux or baseline.loc[lan_aux, lan_task] == 'xxx':
    #             continue
    #         table.append([str(lan_task), str(lan_aux), float(baseline.loc[lan_aux, lan_task])])
    #         # Note: use negative Accuracy score here
    #         # scipy.stats.rankdata() gives rank [1, 2, 3, ...] to negative Accuracy scores [-0.3, -0.2, -0.1],
    #         # so higher Accuracy score is put in the front (small "rank" integer value) of the ranking result
    #         rank_score.append(-float(baseline.loc[lan_aux, lan_task]))
    #
    #     rank.extend(rankdata(rank_score, 'min'))
    #     accuracy_level.extend(rankdata(list(-np.array(rank_score)), 'min'))

    for lan_task in task_lang_set:
        rank_score = []
        for lan_aux in aux_lang_set:
            if baseline.loc[lan_aux, lan_task] == 'xxx' or len(baseline.loc[lan_aux, lan_task].strip()) == 0:
                continue
            table.append([str(lan_task), str(lan_aux), float(baseline.loc[lan_aux, lan_task])])
            # Note: use negative Accuracy score here
            # scipy.stats.rankdata() gives rank [1, 2, 3, ...] to negative Accuracy scores [-0.3, -0.2, -0.1],
            # so higher Accuracy score is put in the front (small "rank" integer value) of the ranking result
            rank_score.append(-float(baseline.loc[lan_aux, lan_task]))

        rank.extend(rankdata(rank_score, 'min'))
        accuracy_level.extend(rankdata(list(-np.array(rank_score)), 'min'))

    table = np.column_stack((np.array(table), np.array(rank), np.array(accuracy_level)))

    #####################
    # entity_match.csv is our feature spreadsheet of the counts of overlapping entity between two languages
    extracted_root = root
    entity_match_table = pd.read_csv(os.path.join(extracted_root, "entity_match.csv"))
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

    extracted_type = [["Count entity_match", "Aux lang dataset size", "Task lang dataset size",
                       "Dataset size ratio"] + distance_category]

    for i in range(1, table.shape[0]):
        lan_task, lan_aux, _, _, _ = table[i]

        count_entity_match = entity_match_table.loc[lan_aux, lan_task] / entity_match_table.loc[lan_task, lan_task]
        # The datasize should be the number of overlapping entities between the language and itself
        datasize_aux = entity_match_table.loc[lan_aux, lan_aux]
        datasize_task = entity_match_table.loc[lan_task, lan_task]
        datasize_ratio = datasize_aux / datasize_task
        distance_list = [dtable.loc[convert(lan_aux), convert(lan_task)] for dtable in distance_tables]

        extracted_type.append([count_entity_match, datasize_aux, datasize_task, datasize_ratio] + distance_list)

    table = np.column_stack((table, np.array(extracted_type)))

    # Write out raw dataset for ranking
    with open(os.path.join(root, "data_ranking_el.csv"), "w") as f:
        for row in table:
            print(",".join(list(row)), file=f)