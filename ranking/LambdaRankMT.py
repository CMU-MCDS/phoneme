#!/usr/bin/env python3
import numpy as np
import pandas as pd
import os
from collections import defaultdict
from sklearn.datasets import load_svmlight_file
import lightgbm


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
    #root = "/home/yuhsianl/public/phoneme_common_data/data/mt"

    # Working directory on your local machine
    root = "/Users/yuhsianglin/Dropbox/cmu/11634A_11632A_capstone/20181029 Jupyter notebook"

    # Load data for ranking model
    data_file = "data_ranking_mt.csv"
    lang_set_file = "language_set.txt"
    data = np.loadtxt(os.path.join(root, data_file), dtype=str, delimiter=",")
    lang_set = np.loadtxt(os.path.join(root, lang_set_file), dtype=str)

    # Do 44--10 training/test set separation
    TRAIN_LANG_NUM = 44
    lang_set_shuffled = lang_set[np.random.permutation(lang_set.shape[0])]
    train_lang_set = lang_set_shuffled[:TRAIN_LANG_NUM]
    test_lang_set = lang_set_shuffled[TRAIN_LANG_NUM:]
    print("train_lang_set =", train_lang_set)
    print("test_lang_set =", test_lang_set)

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
    REL_EXP_CUTOFF = 34
        
    for i in range(1, data.shape[0]):
        row = data[i]
        task_lang = str(row[0])
        aux_lang = str(row[1])
        rank = int(row[3])
        BLEU_level = int(row[4])
        rel_exp = lgbm_rel_exp(BLEU_level, REL_EXP_CUTOFF)

        features = row[5:]
        feature_dict = {k: v for k, v in enumerate(features)}
        line_out = [str(rel_exp)]
        line_out.extend([str(k) + ":" + str(v) for k, v in feature_dict.items()])

        if task_lang in train_lang_set and aux_lang in train_lang_set:
            print(" ".join(line_out), file=rank_train_file)
            print(",".join([task_lang, aux_lang, str(rank)]), file=rank_train_lang_pair_file)
            train_qg_size[task_lang] += 1
            if task_lang not in train_query_seq:
                train_query_seq.append(task_lang)
        elif task_lang in test_lang_set and aux_lang in train_lang_set:
            print(" ".join(line_out), file=rank_test_file)
            print(",".join([task_lang, aux_lang, str(rank)]), file=rank_test_lang_pair_file)
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
    qgsize_test = np.loadtxt(os.path.join(rank_train_dir, "rank.test.qgsize.txt"))
    model = lightgbm.LGBMRanker(boosting_type='gbdt', num_leaves=4,
                                max_depth=-1, learning_rate=0.1, n_estimators=100,
                                min_child_samples=5)
    model.fit(X_train, y_train, group=qgsize_train,
              eval_set=[(X_test, y_test)], eval_group=[qgsize_test], eval_at=3,
              early_stopping_rounds=10, eval_metric="ndcg", verbose=False)

    print("Features:", data[0, 5:])
    print("Feature importance:", model.feature_importances_)

    #print("Best test NDCG@1 during training =", model.best_score_['valid_0']['ndcg@1'])
    #print("Best test NDCG@2 during training =", model.best_score_['valid_0']['ndcg@2'])
    print("Best test NDCG@3 during training =", model.best_score_['valid_0']['ndcg@3'])
    #print("Best test NDCG@10 during training =", model.best_score_['valid_0']['ndcg@10'])
    print("Best iteration =", model.best_iteration_)
    print("Total number of training iterations =", len(model.evals_result_["valid_0"]["ndcg@3"]))

    test_lang_pair = np.loadtxt(os.path.join(rank_train_dir, "rank.test.langpair.txt"), dtype=str, delimiter=",")
    PRINT_TOP_K = 3

    predict_scores = model.predict(X_test)
    qg_start_idx = 0
    for qg_size in qgsize_test:
        qg_scores = predict_scores[qg_start_idx:qg_start_idx + int(qg_size)]
        best_aux_idx = np.argsort(-qg_scores)   # argsort: ascending
        task_lang = test_lang_pair[qg_start_idx, 0]
        aux_lang_list = []
        true_rank_list = []
        for i in range(PRINT_TOP_K):
            assert(test_lang_pair[qg_start_idx + best_aux_idx[i], 0] == task_lang)
            aux_lang_list.append(test_lang_pair[qg_start_idx + best_aux_idx[i], 1])
            true_rank_list.append(int(test_lang_pair[qg_start_idx + best_aux_idx[i], 2]))
        print("Top", PRINT_TOP_K, "auxiliary language for '%s'" % task_lang, "are:", aux_lang_list, "with true ranks", true_rank_list)
        qg_start_idx += int(qg_size)
