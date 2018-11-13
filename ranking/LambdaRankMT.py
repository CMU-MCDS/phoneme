#!/usr/bin/env python3
import numpy as np
import pandas as pd
import os
from collections import defaultdict
from sklearn.datasets import load_svmlight_file
import lightgbm
import matplotlib.pyplot as plt
import evaluation


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
    root = "/home/yuhsianl/public/phoneme_common_data/data/mt"

    # Working directory on your local machine
    # root = "/Users/yuhsianglin/Dropbox/cmu/11634A_11632A_capstone/20181029 Jupyter notebook"

    # Load data for ranking model
    data_file = "data_ranking_mt.csv"
    lang_set_file = "language_set.txt"
    data = np.loadtxt(os.path.join(root, data_file), dtype=str, delimiter=",")
    lang_set = np.loadtxt(os.path.join(root, lang_set_file), dtype=str)

    # Do 53--1 training/test set separation
    NDCG_list = []
    test_data_size_list = []
    my_NDCG_list = []
    NDCG_size_list = []
    for i in range(lang_set.shape[0]):
        test_lang_set = [lang_set[i]]
        train_lang_set = np.concatenate((lang_set[:i], lang_set[i + 1:]), axis=0)
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

        for i in range(1, data.shape[0]):
            row = data[i]
            task_lang = str(row[0])
            aux_lang = str(row[1])
            rank = int(row[3])
            BLEU_level = int(row[4])
            rel_exp = lgbm_rel_exp(BLEU_level, REL_EXP_CUTOFF)
            task_data_size = int(row[11])
            aux_data_size = int(row[8])

            features = row[5:]
            feature_dict = {k: v for k, v in enumerate(features)}

            # Here we use BLEU_level as our relevance exponent
            line_out = [str(rel_exp)]

            line_out.extend([str(k) + ":" + str(v) for k, v in feature_dict.items()])

            if task_lang in train_lang_set and aux_lang in train_lang_set:
                print(" ".join(line_out), file=rank_train_file)
                print(",".join([task_lang, aux_lang, str(rank), str(task_data_size), str(aux_data_size)]), file=rank_train_lang_pair_file)
                train_qg_size[task_lang] += 1
                if task_lang not in train_query_seq:
                    train_query_seq.append(task_lang)
            elif task_lang in test_lang_set and aux_lang in train_lang_set:
                print(" ".join(line_out), file=rank_test_file)
                print(",".join([task_lang, aux_lang, str(rank), str(task_data_size), str(aux_data_size)]), file=rank_test_lang_pair_file)
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
        for qg_size in qgsize_test:
            qg_scores = predict_scores[qg_start_idx:qg_start_idx + int(qg_size)]
            best_aux_idx = np.argsort(-qg_scores)   # argsort: ascending
            task_lang = test_lang_pair[qg_start_idx, 0]
            task_size = test_lang_pair[qg_start_idx, 3]

            # Here we assert there are only one task language
            test_data_size_list.append(task_size)

            aux_lang_list = []
            true_rank_list = []
            aux_size_list = []
            for i in range(PRINT_TOP_K):
                assert(test_lang_pair[qg_start_idx + best_aux_idx[i], 0] == task_lang)
                aux_lang_list.append(test_lang_pair[qg_start_idx + best_aux_idx[i], 1])
                true_rank_list.append(int(test_lang_pair[qg_start_idx + best_aux_idx[i], 2]))
                aux_size_list.append(test_lang_pair[qg_start_idx + best_aux_idx[i], 4])

            test_aux_size_list = test_lang_pair[qg_start_idx:qg_start_idx + int(qg_size), 4].astype(int)
            best_aux_idx_from_size = np.argsort(-test_aux_size_list)
            aux_lang_list_from_size = []
            true_rank_list_from_size = []
            for i in range(PRINT_TOP_K):
                aux_lang_list_from_size.append(test_lang_pair[qg_start_idx + best_aux_idx_from_size[i], 1])
                true_rank_list_from_size.append(int(test_lang_pair[qg_start_idx + best_aux_idx_from_size[i], 2]))

            print("Top", PRINT_TOP_K, "auxiliary language for '%s'" % task_lang, "are:", aux_lang_list, "with true ranks", true_rank_list)
            print("Task language data size = %d, task languages data size =" % int(task_size), aux_size_list)

            print("Using only data size, the top", PRINT_TOP_K, "auxiliary language are:", aux_lang_list_from_size, "with true ranks", true_rank_list_from_size)

            relevance_sorted_lgbm = y_test[qg_start_idx + best_aux_idx]

            print("[DEBUG] y_train =", y_train)
            print("[DEBUG] y_test =", y_test)

            true_rel_exp = y_test[qg_start_idx:qg_start_idx + int(qg_size)]
            relevance_sorted_true = -np.sort(-true_rel_exp)

            NDCG = evaluation.ndcg(relevance_sorted_lgbm, PRINT_TOP_K, relevance_sorted_true)
            print("My calculation of model NDCG@3 =", NDCG)
            my_NDCG_list.append(NDCG)

            # NDCG, using only data size
            relevance_sorted_size = y_test[qg_start_idx + best_aux_idx_from_size]
            NDCG_size = evaluation.ndcg(relevance_sorted_size, PRINT_TOP_K, relevance_sorted_true)
            print("Using only dataset size, NDCG@3 =", NDCG_size)
            NDCG_size_list.append(NDCG_size)

            qg_start_idx += int(qg_size)

    avg_NDCG = np.average(np.array(NDCG_list))
    std_NDCG = np.std(np.array(NDCG_list))
    print("Average NDCG@3 =", avg_NDCG, "and standard deviation =", std_NDCG)

    my_avg_NDCG = np.average(np.array(my_NDCG_list))
    my_std_NDCG = np.std(np.array(my_NDCG_list))
    print("My average NDCG@3 =", my_avg_NDCG, "and standard deviation =", my_std_NDCG)

    avg_NDCG_size = np.average(np.array(NDCG_size_list))
    std_NDCG_size = np.std(np.array(NDCG_size_list))
    print("Using only dataset size, average NDCG@3 =", avg_NDCG_size, "and standard deviation =", std_NDCG_size)

    """
    plt.plot(test_data_size_list, NDCG_list, "k.")
    plt.xlabel("Task language data size")
    plt.ylabel("Average NDCG@3")
    plt.savefig("./NDCG3_datasize.png")
    plt.clf()
    plt.close()
    """
