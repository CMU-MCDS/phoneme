#!/usr/bin/env python3
import numpy as np
import pandas as pd
import os
from collections import defaultdict
from sklearn.datasets import load_svmlight_file
import lightgbm
import matplotlib.pyplot as plt
import evaluation
import json


# Compute the "relevance exponent" for LightGBM use
#
# It is used in the DCG@N as
# \sum_{i = 1}^N \frac{2^{relevance exponent} - 1}{log_2 (1 + i)}
#
# Transform the BLEU level [cutoff, cutoff + 1, ...] into relevance exponent [1, 2, ...],
# and assign the BLEU levels below cutoff to relevance exponent 0
def lgbm_rel_exp(BLEU_level, cutoff):
    return BLEU_level - cutoff + 1 if BLEU_level >= cutoff else "0"


if __name__ == "__main__":
    # Working directory on clio
    # root = "/home/yuhsianl/public/phoneme_common_data/data/mt"

    # Working directory on your local machine
    root = "/Users/yuhsianglin/Dropbox/cmu/11634A_11632A_capstone/20181029 Jupyter notebook"

    # Create directory for output
    output_dir = os.path.join(root, "output_mt")
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    # Load data for ranking model
    data_file = "data_ranking_mt.csv"
    lang_set_file = "language_set.txt"
    data = np.loadtxt(os.path.join(root, data_file), dtype=str, delimiter=",")
    lang_set = np.loadtxt(os.path.join(root, lang_set_file), dtype=str)

    # Do 53--1 training/test set separation
    NDCG_list = []
    test_data_size_list = []

    # NDCG_output_dict should be:
    # {
    #  "LambdaRank":
    #   {
    #    "task_lang":  ["task lang 1", "task lang 2", ...],
    #    "NDCG_list": [NDCG for task lang 1, NDCG for task lang 2, ...],
    #    "avg":       average NDCG,
    #    "std":       NDCG standard deviation
    #   },
    #  "single feature 1":
    #   {
    #    "task_lang":  ["task lang 1", "task lang 2", ...],
    #    "NDCG_list": [NDCG for task lang 1, NDCG for task lang 2, ...],
    #    "avg":       average NDCG,
    #    "std":       NDCG standard deviation
    #   },
    #  ...
    # }
    NDCG_output_dict = {"LambdaRank": {"task_lang": [], "NDCG_list": [], "avg": -1, "std": -1}}

    single_feature_name_list = ["Aux lang TTR", "Overlap word-level", "Overlap subword-level", "Aux lang dataset size", "TTR difference ratio", "Dataset size ratio", "Task lang dataset size", "GEOGRAPHIC", "GENETIC", "SYNTACTIC", "FEATURAL", "INVENTORY", "PHONOLOGICAL"]
    for feature in single_feature_name_list:
        NDCG_output_dict[feature] = {"task_lang": [], "NDCG_list": [], "avg": -1, "std": -1}

    for task_lang_idx in range(lang_set.shape[0]):
        test_lang_set = [lang_set[task_lang_idx]]
        train_lang_set = np.concatenate((lang_set[:task_lang_idx], lang_set[task_lang_idx + 1:]), axis=0)
        assert(train_lang_set.shape[0] == 53)
        assert(test_lang_set[0] not in list(train_lang_set))

        # Count number of queries in a query group of a task language
        # {language name: number of queries}
        train_qg_size = defaultdict(int)
        test_qg_size = defaultdict(int)

        # Record the order of query language (task language) we see
        train_query_seq = []
        test_query_seq = []

        # Generate training/test data for LightGBM
        rank_train_dir = "."
        rank_train_file = open(os.path.join(rank_train_dir, "rank.train.txt"), "w")
        rank_test_file = open(os.path.join(rank_train_dir, "rank.test.txt"), "w")
        rank_train_lang_pair_file = open(os.path.join(rank_train_dir, "rank.train.langpair.txt"), "w")
        rank_test_lang_pair_file = open(os.path.join(rank_train_dir, "rank.test.langpair.txt"), "w")

        # Transform the BLEU level [cutoff, cutoff + 1, ...] into relevance exponent [1, 2, ...],
        # and assign the BLEU levels below cutoff to relevance exponent 0
        REL_EXP_CUTOFF = 44

        for data_row_idx in range(1, data.shape[0]):
            row = data[data_row_idx]
            task_lang = str(row[0])
            aux_lang = str(row[1])
            rank = int(row[3])
            BLEU_level = int(row[4])
            rel_exp = lgbm_rel_exp(BLEU_level, REL_EXP_CUTOFF)

            # Features are:
            # ["Aux lang TTR", "Overlap word-level", "Overlap subword-level", "Aux lang dataset size", "TTR difference ratio", "Dataset size ratio", "Task lang dataset size", "GEOGRAPHIC", "GENETIC", "SYNTACTIC", "FEATURAL", "INVENTORY", "PHONOLOGICAL"]
            features = row[5:]

            # Here we use BLEU_level as our relevance exponent
            line_out = [str(rel_exp)]

            line_out.extend([str(k) + ":" + str(v) for k, v in enumerate(features)])

            if task_lang in train_lang_set and aux_lang in train_lang_set:
                print(" ".join(line_out), file=rank_train_file)
                print(",".join([task_lang, aux_lang, str(rank)] + list(map(str, features))), file=rank_train_lang_pair_file)
                train_qg_size[task_lang] += 1
                if task_lang not in train_query_seq:
                    train_query_seq.append(task_lang)
            elif task_lang in test_lang_set and aux_lang in train_lang_set:
                print(" ".join(line_out), file=rank_test_file)
                print(",".join([task_lang, aux_lang, str(rank)] + list(map(str, features))), file=rank_test_lang_pair_file)
                test_qg_size[task_lang] += 1
                if task_lang not in test_query_seq:
                    test_query_seq.append(task_lang)

        rank_train_file.close()
        rank_test_file.close()
        rank_train_lang_pair_file.close()
        rank_test_lang_pair_file.close()

        # Generate query group size file for LightGBM
        with open(os.path.join(rank_train_dir, "rank.train.qgsize.txt"), "w") as f:
            for lang in train_query_seq:
                print(train_qg_size[lang], file=f)

        with open(os.path.join(rank_train_dir, "rank.test.qgsize.txt"), "w") as f:
            for lang in test_query_seq:
                print(test_qg_size[lang], file=f)

        X_train, y_train = load_svmlight_file(os.path.join(rank_train_dir, "rank.train.txt"))
        X_test, y_test = load_svmlight_file(os.path.join(rank_train_dir, "rank.test.txt"))
        qgsize_train = np.loadtxt(os.path.join(rank_train_dir, "rank.train.qgsize.txt"))
        qgsize_test = np.loadtxt(os.path.join(rank_train_dir, "rank.test.qgsize.txt")).reshape(-1)
        model = lightgbm.LGBMRanker(boosting_type='gbdt', num_leaves=16,
                                    max_depth=-1, learning_rate=0.1, n_estimators=100,
                                    min_child_samples=5)
        model.fit(X_train, y_train, group=qgsize_train,
                  eval_set=[(X_test, y_test)], eval_group=[qgsize_test], eval_at=3,
                  early_stopping_rounds=40, eval_metric="ndcg", verbose=False)
        model.booster_.save_model(os.path.join(output_dir, "lgbm_model_mt_" + lang_set[task_lang_idx] + ".txt"))

        print("================================")
        print("Features:", data[0, 5:])
        print("Feature importance:", model.feature_importances_)

        #print("Best test NDCG@1 during training =", model.best_score_['valid_0']['ndcg@1'])
        #print("Best test NDCG@2 during training =", model.best_score_['valid_0']['ndcg@2'])
        print("Best test NDCG@3 during training =", model.best_score_['valid_0']['ndcg@3'])
        #print("Best test NDCG@10 during training =", model.best_score_['valid_0']['ndcg@10'])
        print("Best iteration =", model.best_iteration_)
        print("Total number of training iterations =", len(model.evals_result_["valid_0"]["ndcg@3"]))

        NDCG_list.append(model.best_score_['valid_0']['ndcg@3'])

        test_lang_pair = np.loadtxt(os.path.join(rank_train_dir, "rank.test.langpair.txt"), dtype=str, delimiter=",")
        PRINT_TOP_K = 3

        predict_scores = model.predict(X_test)
        qg_start_idx = 0
        # Here we actually assert there are only one task language
        assert(len(qgsize_test) == 1)
        for qg_size in qgsize_test:
            qg_scores = predict_scores[qg_start_idx:qg_start_idx + int(qg_size)]
            best_aux_idx = np.argsort(-qg_scores)   # argsort: ascending
            task_lang = test_lang_pair[qg_start_idx, 0]
            task_size = test_lang_pair[qg_start_idx, 6]

            true_ranking = test_lang_pair[qg_start_idx:qg_start_idx + int(qg_size), 2].astype(int)
            true_best_aux_idx = np.argsort(true_ranking)

            # Here we assert there are only one task language
            test_data_size_list.append(task_size)

            # topK_output_dict should be:
            # {
            #  "task_lang": task lang,
            #  "LambdaRank": [[top 1 aux lang, top 2 aux lang, top 3 aux lang], [true rank 1, true rank 2, true rank 3]],
            #  "single feature 1": [[top 1 aux lang, top 2 aux lang, top 3 aux lang], [true rank 1, true rank 2, true rank 3]],
            #  "single feature 2": [[top 1 aux lang, top 2 aux lang, top 3 aux lang], [true rank 1, true rank 2, true rank 3]],
            #  ...
            # }
            # We will save it as an json file
            topK_output_dict = {"task_lang": lang_set[task_lang_idx]}

            # Extract top-K results
            topK_aux_lang_list = []
            topK_true_rank_list = []
            for topK_k in range(PRINT_TOP_K):
                test_lang_pair_row = test_lang_pair[qg_start_idx + best_aux_idx[topK_k]]
                # The first 3 columns are: task_lang, aux_lang, str(rank)
                assert(test_lang_pair_row[0] == task_lang)
                topK_aux_lang_list.append(test_lang_pair_row[1])
                topK_true_rank_list.append(int(test_lang_pair_row[2]))
            topK_output_dict["LambdaRank"] = [topK_aux_lang_list, topK_true_rank_list]

            # Extract true top-K
            true_topK_aux_lang_list = []
            true_topK_true_rank_list = []
            for topK_k in range(PRINT_TOP_K):
                test_lang_pair_row = test_lang_pair[qg_start_idx + true_best_aux_idx[topK_k]]
                # The first 3 columns are: task_lang, aux_lang, str(rank)
                true_topK_aux_lang_list.append(test_lang_pair_row[1])
                true_topK_true_rank_list.append(int(test_lang_pair_row[2]))
            topK_output_dict["Truth"] = [true_topK_aux_lang_list, true_topK_true_rank_list]

            # Extract top-K results by each single feature
            # single_feature_name_list = ["Aux lang TTR", "Overlap word-level", "Overlap subword-level", "Aux lang dataset size", "TTR difference ratio", "Dataset size ratio", "Task lang dataset size", "GEOGRAPHIC", "GENETIC", "SYNTACTIC", "FEATURAL", "INVENTORY", "PHONOLOGICAL"]

            best_aux_idx_by_single_feature_lists = [[] for _ in range(len(single_feature_name_list))]
            # Smaller value is better (e.g. distance) => sign = +1
            # Larger value is better (e.g. dataset size) => sign = -1
            # 0 means we ignore this feature (don't compute single-feature result of it)
            sort_sign_list = [0, -1, -1, -1, 0, -1, -1, 1, 1, 1, 1, 1, 1]
            assert(len(sort_sign_list) == len(single_feature_name_list))

            topK_aux_lang_by_single_feature_lists = [[] for _ in range(len(single_feature_name_list))]
            topK_true_rank_by_single_feature_lists = [[] for _ in range(len(single_feature_name_list))]

            for single_feature_idx in range(len(single_feature_name_list)):
                if sort_sign_list[single_feature_idx] != 0:
                    single_feature_value_list = test_lang_pair[qg_start_idx:qg_start_idx + int(qg_size), 3 + single_feature_idx].astype(float)
                    best_aux_idx_by_single_feature_lists[single_feature_idx] = np.argsort(sort_sign_list[single_feature_idx] * single_feature_value_list)
                    for topK_k in range(PRINT_TOP_K):
                        aux_lang_got = test_lang_pair[qg_start_idx + best_aux_idx_by_single_feature_lists[single_feature_idx][topK_k], 1]
                        true_rank_got = test_lang_pair[qg_start_idx + best_aux_idx_by_single_feature_lists[single_feature_idx][topK_k], 2]
                        topK_aux_lang_by_single_feature_lists[single_feature_idx].append(aux_lang_got)
                        topK_true_rank_by_single_feature_lists[single_feature_idx].append(int(true_rank_got))

                    topK_output_dict[single_feature_name_list[single_feature_idx]] = [topK_aux_lang_by_single_feature_lists[single_feature_idx], topK_true_rank_by_single_feature_lists[single_feature_idx]]

            with open(os.path.join(output_dir, "topK_" + lang_set[task_lang_idx] + ".json"), "w") as f:
                json.dump(topK_output_dict, f)

            # Compute NDCG
            print("[DEBUG] y_train =", y_train)
            print("[DEBUG] y_test =", y_test)

            true_rel_exp = y_test[qg_start_idx:qg_start_idx + int(qg_size)]
            relevance_sorted_true = -np.sort(-true_rel_exp)

            relevance_sorted_lgbm = y_test[qg_start_idx + best_aux_idx]
            NDCG = evaluation.ndcg(relevance_sorted_lgbm, PRINT_TOP_K, relevance_sorted_true)
            NDCG_output_dict["LambdaRank"]["task_lang"].append(lang_set[task_lang_idx])
            NDCG_output_dict["LambdaRank"]["NDCG_list"].append(NDCG)

            for single_feature_idx in range(len(single_feature_name_list)):
                if sort_sign_list[single_feature_idx] != 0:
                    relevance_sorted_single_feature = y_test[qg_start_idx + best_aux_idx_by_single_feature_lists[single_feature_idx]]
                    NDCG_single_feature = evaluation.ndcg(relevance_sorted_single_feature, PRINT_TOP_K, relevance_sorted_true)
                    NDCG_output_dict[single_feature_name_list[single_feature_idx]]["task_lang"].append(lang_set[task_lang_idx])
                    NDCG_output_dict[single_feature_name_list[single_feature_idx]]["NDCG_list"].append(NDCG_single_feature)
                else:
                    # Remove the un-used feature item placeholder
                    if single_feature_name_list[single_feature_idx] in NDCG_output_dict:
                        del NDCG_output_dict[single_feature_name_list[single_feature_idx]]

            qg_start_idx += int(qg_size)

    # Compute avg/std of NDCG
    NDCG_output_dict["LambdaRank"]["avg"] = np.average(np.array(NDCG_output_dict["LambdaRank"]["NDCG_list"]))
    NDCG_output_dict["LambdaRank"]["std"] = np.std(np.array(NDCG_output_dict["LambdaRank"]["NDCG_list"]))
    for single_feature_idx in range(len(single_feature_name_list)):
        if sort_sign_list[single_feature_idx] != 0:
            NDCG_output_dict[single_feature_name_list[single_feature_idx]]["avg"] = np.average(np.array(NDCG_output_dict[single_feature_name_list[single_feature_idx]]["NDCG_list"]))
            NDCG_output_dict[single_feature_name_list[single_feature_idx]]["std"] = np.std(np.array(NDCG_output_dict[single_feature_name_list[single_feature_idx]]["NDCG_list"]))

    with open(os.path.join(output_dir, "NDCG.json"), "w") as f:
        json.dump(NDCG_output_dict, f)

    """
    plt.plot(test_data_size_list, NDCG_list, "k.")
    plt.xlabel("Task language data size")
    plt.ylabel("Average NDCG@3")
    plt.savefig("./NDCG3_datasize.png")
    plt.clf()
    plt.close()
    """
