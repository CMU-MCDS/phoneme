import numpy as np

"""
relevance_sorted =
    [relevance level of the rank-1 doc predicted by the model,
     relevance level of the rank-2 doc predicted by the model,
     ...]

n is the n of "DCG@n"

Not considering averaging over tie at this moment.
"""
def dcg(relevance_sorted, n):
    dcg = 0.0
    for idx in range(n):
        dcg += (np.power(2, relevance_sorted[idx]) - 1) / np.log2(idx + 2)

    return dcg

"""
Currently I put return value as 1.0 if true DCG got is 0.
"""
def ndcg(relevance_sorted, n, true_relevance_sorted):
    true_dcg = dcg(true_relevance_sorted, n)
    pred_dcg = dcg(relevance_sorted, n)
    return pred_dcg / true_dcg if true_dcg > 0 else 1.0
