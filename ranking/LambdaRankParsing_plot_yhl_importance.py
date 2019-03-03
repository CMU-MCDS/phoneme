#!/usr/bin/env python3
import numpy as np
import pandas as pd
import os
from collections import defaultdict
from sklearn.datasets import load_svmlight_file
import lightgbm
### BEGIN FOR FIG 2 ###
import matplotlib
matplotlib.use('agg')
import sys
### END FOR FIG 2 ###
import matplotlib.pyplot as plt
import evaluation
import json


# Compute the "relevance exponent" for LightGBM use
#
# It is used in the DCG@N as
# \sum_{i = 1}^N \frac{2^{relevance exponent} - 1}{log_2 (1 + i)}
#
# Transform the acc level [cutoff, cutoff + 1, ...] into relevance exponent [1, 2, ...],
# and assign the acc levels below cutoff to relevance exponent 0
def lgbm_rel_exp(acc_level, cutoff):
    return acc_level - cutoff + 1 if acc_level >= cutoff else "0"


if __name__ == "__main__":
    # Working directory on clio
    root = "/home/yuhsianl/public/phoneme_common_data/data/parsing"

    # Working directory on your local machine
    # root = "/Users/yuhsianglin/Dropbox/cmu/phoneme_data/parsing"

    # Create directory for output
    output_dir = os.path.join(root, "output_parsing")
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    # Load data for ranking model
    data_file = "data_ranking_parsing.csv" #Dataset-dependent features
    data = np.loadtxt(os.path.join(root, data_file), dtype=str, delimiter=",")

    task_lang_file = 'tg_language_set.txt'
    aux_lang_file = 'tf_language_set.txt'
    task_lang_set = np.loadtxt(os.path.join(root, task_lang_file), dtype=str)
    aux_lang_set = np.loadtxt(os.path.join(root, aux_lang_file), dtype=str)


    # Do leave one out training/test set separation

    ### BEGIN FOR FIG 2 ###
    plot_list = []
    total_importance = [0 for i in range(13)]
    ### END FOR FIG 2 ###

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

    single_feature_name_list = ["Overlap word-level", "Transfer lang dataset size", "Target lang dataset size", "Transfer over target size ratio", "Transfer lang TTR", "Target lang TTR", "Transfer target TTR distance", "GEOGRAPHIC", "GENETIC", "SYNTACTIC", "FEATURAL", "INVENTORY", "PHONOLOGICAL"]
    for feature in single_feature_name_list:
        NDCG_output_dict[feature] = {"task_lang": [], "NDCG_list": [], "avg": -1, "std": -1}

    for task_lang_idx in range(task_lang_set.shape[0]):
        test_lang_set = [task_lang_set[task_lang_idx]]
        train_lang_set = [lang for lang in aux_lang_set if lang not in test_lang_set]
        assert(len(train_lang_set) == 29)
        assert(test_lang_set[0] not in list(train_lang_set))

        # Count number of queries in a query group of a task language
        # {language name: number of queries}
        train_qg_size = defaultdict(int)
        test_qg_size = defaultdict(int)

        # Record the order of query language (task language) we see
        train_query_seq = []
        test_query_seq = []

        # Generate training/test data for LightGBM
        rank_train_dir = "./train_parsing"
        if not os.path.exists(rank_train_dir):
            os.makedirs(rank_train_dir)
        rank_train_file = open(os.path.join(rank_train_dir, "rank.train.txt"), "w")
        rank_test_file = open(os.path.join(rank_train_dir, "rank.test.txt"), "w")
        rank_train_lang_pair_file = open(os.path.join(rank_train_dir, "rank.train.langpair.txt"), "w")
        rank_test_lang_pair_file = open(os.path.join(rank_train_dir, "rank.test.langpair.txt"), "w")

        # Transform the acc level [cutoff, cutoff + 1, ...] into relevance exponent [1, 2, ...],
        # and assign the acc levels below cutoff to relevance exponent 0
        REL_EXP_CUTOFF = len(train_lang_set) - 9

        for data_row_idx in range(1, data.shape[0]):
            row = data[data_row_idx]
            task_lang = str(row[0])
            aux_lang = str(row[1])
            rank = int(row[3])
            acc_level = int(row[4])
            rel_exp = lgbm_rel_exp(acc_level, REL_EXP_CUTOFF)

            # Features are:
            # ["Word overlap", "Transfer lang dataset size", "Target lang dataset size", "Transfer over target size ratio", "Transfer lang TTR", "Target lang TTR", "Transfer target TTR distance", "GEOGRAPHIC", "GENETIC", "SYNTACTIC", "FEATURAL", "INVENTORY", "PHONOLOGICAL"]
            features = row[5:]

            # Here we use acc_level as our relevance exponent
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
                ### BEGIN FOR FIG 2 ###
                # print(",".join([task_lang, aux_lang, str(rank)] + list(map(str, features))), file=rank_test_lang_pair_file)
                print(",".join([task_lang, aux_lang, str(rank)] + list(map(str, features)) + [str(row[2])]), file=rank_test_lang_pair_file)
                ### END FOR FIG 2 ###
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
        model.booster_.save_model(os.path.join(output_dir, "lgbm_model_parsing_" + task_lang_set[task_lang_idx] + ".txt"))


        ### BEGIN FOR FIG 2 ###

        # single_feature_name_list = ["Word overlap", "Transfer lang dataset size", "Target lang dataset size", "Transfer over target size ratio", "Transfer lang TTR", "Target lang TTR", "Transfer target TTR distance", "GEOGRAPHIC", "GENETIC", "SYNTACTIC", "FEATURAL", "INVENTORY", "PHONOLOGICAL"]

        feature_names = ['o_w','s_tf','s_tk','s_tf/s_tk','ttr_tf','ttr_tk','d_ttr','d_geo','d_gen','d_syn','d_fea','d_inv','d_pho']

        # model.fit(X_train, y_train, group=qgsize_train,
        #          eval_set=[(X_test, y_test)], eval_group=[qgsize_test], eval_at=3,
        #          early_stopping_rounds=40, eval_metric="ndcg", verbose=False)
        
        model.fit(X_train, y_train, group=qgsize_train, eval_set=[(X_test, y_test)], eval_group=[qgsize_test], eval_at=3, early_stopping_rounds=40, eval_metric="ndcg", verbose=False, feature_name = feature_names)

        # model.booster_.save_model(os.path.join(output_dir, "lgbm_model_mt_" + lang_set[task_lang_idx] + ".txt"))

        ### END FOR FIG 2 ###



        print("================================")
        print("Features:", data[0, 5:])
        print("Feature importance:", model.feature_importances_)
        print("Feature importance num:", model.feature_importances_.shape)
        for i in range(model.feature_importances_.shape[0]):
            total_importance[i]+=model.feature_importances_[i]
        #print("Best test NDCG@1 during training =", model.best_score_['valid_0']['ndcg@1'])
        #print("Best test NDCG@2 during training =", model.best_score_['valid_0']['ndcg@2'])
        print("Best test NDCG@3 during training =", model.best_score_['valid_0']['ndcg@3'])
        #print("Best test NDCG@10 during training =", model.best_score_['valid_0']['ndcg@10'])
        print("Best iteration =", model.best_iteration_)
        print("Total number of training iterations =", len(model.evals_result_["valid_0"]["ndcg@3"]))

        test_lang_pair = np.loadtxt(os.path.join(rank_train_dir, "rank.test.langpair.txt"), dtype=str, delimiter=",")
        PRINT_TOP_K = 3

        predict_scores = model.predict(X_test)


        ### BEGIN FOR FIG 2 ###

        predict_list_rank = {}
        # Format: {test language: (predicted ranking score, true BLEU)}

        for a in range(X_test.shape[0]):
            #predict_list_rank[test_lang_pair[a][1]] = (model.predict(X_test[a])[0],float(test_lang_pair[a][-1]))
            #print("test_lang_pair[a]=",test_lang_pair[a])
            predict_list_rank[test_lang_pair[a][1]] = (model.predict(X_test[a])[0],float(test_lang_pair[a][-1]))   ### use "Overlap subword-level at [3+1]"
            #print("a=",a,", X_test[a]=",X_test[a],", model.predict(X_test[a])=",model.predict(X_test[a]),", test_lang_pair[a]=",test_lang_pair[a])

        # Sort w.r.t. true BLEU; so the first one is the highest true BLEU
        predict_list_bleu = sorted(predict_list_rank.items(),key=lambda x:-x[1][1])
        #print("predict_list_rank = ", predict_list_rank)

        # Sort w.r.t. predicted ranking score, so the first one is the best transfer language recommended by LangRank
        predict_list_rank = sorted(predict_list_rank.items(),key=lambda x:-x[1][0])
        #print (predict_list_bleu)

        #num_to_plot = len(predict_list_rank)
        num_to_plot = min(10,len(predict_list_rank))
        t = np.arange(1, num_to_plot+1)
        temp = []
        cur_max = -sys.maxsize
        for a in range(num_to_plot):
            if cur_max < predict_list_rank[a][1][1]:
                cur_max = predict_list_rank[a][1][1]    # [1][1] refers to the true BLEU of the a-th best language recommended by LangRank
            temp.append(cur_max/predict_list_bleu[0][1][1])
        #s = temp
        plot_list.append(temp)

        ### END FOR FIG 2 ###


        qg_start_idx = 0
        # Here we actually assert there are only one task language
        assert(len(qgsize_test) == 1)
        for qg_size in qgsize_test:
            qg_scores = predict_scores[qg_start_idx:qg_start_idx + int(qg_size)]
            best_aux_idx = np.argsort(-qg_scores)   # argsort: ascending
            task_lang = test_lang_pair[qg_start_idx, 0]

            true_ranking = test_lang_pair[qg_start_idx:qg_start_idx + int(qg_size), 2].astype(int)
            true_best_aux_idx = np.argsort(true_ranking)

            # topK_output_dict should be:
            # {
            #  "task_lang": task lang,
            #  "LambdaRank": [[top 1 aux lang, top 2 aux lang, top 3 aux lang], [true rank 1, true rank 2, true rank 3]],
            #  "single feature 1": [[top 1 aux lang, top 2 aux lang, top 3 aux lang], [true rank 1, true rank 2, true rank 3]],
            #  "single feature 2": [[top 1 aux lang, top 2 aux lang, top 3 aux lang], [true rank 1, true rank 2, true rank 3]],
            #  ...
            # }
            # We will save it as an json file
            topK_output_dict = {"task_lang": task_lang_set[task_lang_idx]}

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
            # single_feature_name_list = ["Word overlap", "Transfer lang dataset size", "Target lang dataset size", "Transfer over target size ratio", "Transfer lang TTR", "Target lang TTR", "Transfer target TTR distance", "GEOGRAPHIC", "GENETIC", "SYNTACTIC", "FEATURAL", "INVENTORY", "PHONOLOGICAL"]

            best_aux_idx_by_single_feature_lists = [[] for _ in range(len(single_feature_name_list))]
            # Smaller value is better (e.g. distance) => sign = +1
            # Larger value is better (e.g. dataset size) => sign = -1
            # 0 means we ignore this feature (don't compute single-feature result of it)
            sort_sign_list = [-1, -1, 0, -1, 0, 0, 1, 1, 1, 1, 1, 1, 1]
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

            with open(os.path.join(output_dir, "topK_" + task_lang_set[task_lang_idx] + ".json"), "w") as f:
                json.dump(topK_output_dict, f)

            # Compute NDCG
            print("[DEBUG] y_train =", y_train)
            print("[DEBUG] y_test =", y_test)

            true_rel_exp = y_test[qg_start_idx:qg_start_idx + int(qg_size)]
            relevance_sorted_true = -np.sort(-true_rel_exp)

            relevance_sorted_lgbm = y_test[qg_start_idx + best_aux_idx]
            NDCG = evaluation.ndcg(relevance_sorted_lgbm, PRINT_TOP_K, relevance_sorted_true)
            NDCG_output_dict["LambdaRank"]["task_lang"].append(task_lang_set[task_lang_idx])
            NDCG_output_dict["LambdaRank"]["NDCG_list"].append(NDCG)

            for single_feature_idx in range(len(single_feature_name_list)):
                if sort_sign_list[single_feature_idx] != 0:
                    relevance_sorted_single_feature = y_test[qg_start_idx + best_aux_idx_by_single_feature_lists[single_feature_idx]]
                    NDCG_single_feature = evaluation.ndcg(relevance_sorted_single_feature, PRINT_TOP_K, relevance_sorted_true)
                    NDCG_output_dict[single_feature_name_list[single_feature_idx]]["task_lang"].append(task_lang_set[task_lang_idx])
                    NDCG_output_dict[single_feature_name_list[single_feature_idx]]["NDCG_list"].append(NDCG_single_feature)
                else:
                    # Remove the un-used feature item placeholder
                    if single_feature_name_list[single_feature_idx] in NDCG_output_dict:
                        del NDCG_output_dict[single_feature_name_list[single_feature_idx]]

            qg_start_idx += int(qg_size)


    ### BEGIN FOR FIG 2 ###

    # Generate the values to plot for LangRank recommendation
    t = range(1,10+1)
    s = []
    #error = []
    #print("=============\n", plot_list)
    for i in zip(*plot_list):
        #print(i)
        #print (sum(i))
        #print (np.std(np.asarray(i)))
        #error.append(np.std(np.asarray(i)))
        s.append(sum(i)/len(plot_list))


    #plt.xticks(np.arange(1, 10+1, step=1))
    # plt.errorbar(t, s, yerr=error, fmt='-o')
    np.set_printoptions(precision=4)
    print ('result_max_eval',np.array([s]))



    # Generate the values to plot for single-feature (baseline) recommendation

    print("============\nLook at a row of data:\n")
    print(data[0])
    print(data[1])

    # plt.plot(t, s,'-s')
    # plt.xlabel('rank of selected systems')
    # plt.ylabel('max evaluation score ratio')
    # plt.title('Max evaluation score ratio vs. Rank of selected systems')
    # plt.savefig("./figMT.png")
    # plt.show()
    dict_task_transfer_genetic = {} # key = task, val = (genetic,acc)
    for data_row_idx in range(1, data.shape[0]):
        row = data[data_row_idx]
        if row[0] not in dict_task_transfer_genetic:
            dict_task_transfer_genetic[row[0]] = [(-float(row[5+0]),float(row[2]))]
        else:
            dict_task_transfer_genetic[row[0]].append((-float(row[5+0]),float(row[2])))
    for key in dict_task_transfer_genetic:
        dict_task_transfer_genetic[key] = sorted(dict_task_transfer_genetic[key])

    #print (dict_task_transfer_genetic)
    baseline_list = []
    for key in dict_task_transfer_genetic:
        cur_list = []
        max_eval = 0
        max_num = 0
        for i in range(len(dict_task_transfer_genetic[key])):
            max_num =  max(max_num,dict_task_transfer_genetic[key][i][1])
        for i in range(10):
            max_eval = max(max_eval,dict_task_transfer_genetic[key][i][1])
            cur_list.append(max_eval/max_num)
        baseline_list.append(cur_list)

    t = range(1,53+1)
    s = []
    #print (baseline_list,len(baseline_list))
    for i in zip(*baseline_list):
        #print(i)
        #print (sum(i))
        #print (np.std(np.asarray(i)))
        s.append(sum(i)/len(baseline_list))

    #plt.xticks(np.arange(1, 53+1, step=1))
    # plt.errorbar(t, s, yerr=error, fmt='-o')
    np.set_printoptions(precision=3)
    print ('result_max_eval',np.array([s]))

    ### END FOR FIG 2 ###
    print ('total_importance',np.array(total_importance)/(task_lang_set.shape[0]))

    # Compute avg/std of NDCG
    NDCG_output_dict["LambdaRank"]["avg"] = np.average(np.array(NDCG_output_dict["LambdaRank"]["NDCG_list"]))
    NDCG_output_dict["LambdaRank"]["std"] = np.std(np.array(NDCG_output_dict["LambdaRank"]["NDCG_list"]))
    for single_feature_idx in range(len(single_feature_name_list)):
        if sort_sign_list[single_feature_idx] != 0:
            NDCG_output_dict[single_feature_name_list[single_feature_idx]]["avg"] = np.average(np.array(NDCG_output_dict[single_feature_name_list[single_feature_idx]]["NDCG_list"]))
            NDCG_output_dict[single_feature_name_list[single_feature_idx]]["std"] = np.std(np.array(NDCG_output_dict[single_feature_name_list[single_feature_idx]]["NDCG_list"]))

    with open(os.path.join(output_dir, "NDCG.json"), "w") as f:
        json.dump(NDCG_output_dict, f)
