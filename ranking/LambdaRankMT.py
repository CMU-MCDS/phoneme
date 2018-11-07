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
    root = "/home/yuhsianl/public/phoneme_common_data/data/mt"

    # Load data for ranking model
    data_file = "data_ranking_mt.csv"
    data = np.loadtxt(os.path.join(root, data_file), dtype=str, delimiter=",")
    lang_set = np.loadtxt("language_set.txt", dtype=str)

    # Do 44--10 training/test set separation
    TRAIN_LANG_NUM = 44
    lang_set_shuffled = lang_set[np.random.permutation(lang_set.shape[0])]
    train_lang_set = lang_set_shuffled[:TRAIN_LANG_NUM]
    test_lang_set = lang_set_shuffled[TRAIN_LANG_NUM:]

    # Count number of queries in a query group of a task language
    # {language name: number of queries}
    train_qg_size = defaultdict(int)
    test_qg_size = defaultdict(int)

    # Record the order of query language (task language) we see
    train_query_seq = []
    test_query_seq = []

    # Generate training/test data for LightGBM
    rank_train_file = open("rank.train.txt", "w")
    rank_test_file = open("rank.test.txt", "w")

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
        row_out = [rank]
        row_out.extend([str(k) + ":" + str(v) for k, v in feature_dict.items()])

        if task_lang in train_lang_set and aux_lang in train_lang_set:
            print(" ".join(line_out), file=rank_train_file)
            train_qg_size[task_lang] += 1
            if task_lang not in train_query_seq:
                train_query_seq.append(task_lang)
        elif task_lang in test_lang_set and aux_lang in train_lang_set:
            print(" ".join(line_out), file=rank_test_file)
            test_qg_size[task_lang] += 1
            if task_lang not in test_query_seq:
                test_query_seq.append(task_lang)

    rank_train_data.close()
    rank_test_data.close()

    # Generate query group size file for LightGBM
    with open("rank.train.qgsize.txt", "w") as f:
        for lang in train_query_seq:
            print(train_qg_size[lang], file=f)

    with open("rank.test.qgsize.txt", "w") as f:
        for lang in test_query_seq:
            print(test_qg_size[lang], file=f)

    X_train, y_train = load_svmlight_file("rank.train.txt")
    X_test, y_test = load_svmlight_file("rank.test.txt")
    qgsize_train = np.loadtxt("rank.train.qgsize.txt")
    qgsize_test = np.loadtxt("rank.test.qgsize.txt")
    lgbm = lightgbm.LGBMRanker()
    lgbm.fit(X_train, y_train, group=qgsize_train,
             eval_set=[(X_test, y_test)], eval_group=[qgsize_test], eval_at=[1, 2, 3, 10],
             early_stopping_rounds=5, eval_metric="ndcg",
             verbose=False, callbacks=[lgb.reset_parameter(learning_rate=lambda x: 0.95 ** x * 0.1)])

    print("Features:", data[0, 5:])
    print("Feature importance:", gbm.feature_importances_)
    print("Average test NDCG@1 =", np.average(lgbm.evals_result_["valid_0"]["ndcg@1"]))
    print("Average test NDCG@2 =", np.average(lgbm.evals_result_["valid_0"]["ndcg@2"]))
    print("Average test NDCG@3 =", np.average(lgbm.evals_result_["valid_0"]["ndcg@3"]))
    print("Average test NDCG@10 =", np.average(lgbm.evals_result_["valid_0"]["ndcg@10"]))
