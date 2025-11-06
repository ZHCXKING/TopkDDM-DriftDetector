# %%
import numpy as np
from sklearn.metrics import roc_auc_score
# %%
def hit(top_k: np.ndarray, y_true: np.ndarray):  # One-dimensional X and scalar Y
    return int(bool(set(top_k) & set(y_true)))
# %%
def compute_hit_rate(item_list, topk_list):
    hr_list = [
        int(bool(set(top_k) & set(true_items)))
        for true_items, top_k in zip(item_list, topk_list)
    ]
    return np.mean(hr_list)
# %%
def compute_ndcg(item_list, topk_list, k):
    ndcg_list = []
    for true_items, top_k in zip(item_list, topk_list):
        dcg = 0.0
        for rank, item in enumerate(top_k):
            if item in true_items:
                dcg += 1 / np.log2(rank + 2)

        ideal_hits = min(len(true_items), k)
        idcg = sum(1 / np.log2(i + 2) for i in range(ideal_hits))
        ndcg = dcg / idcg if idcg > 0 else 0.0
        ndcg_list.append(ndcg)
    return np.mean(ndcg_list)
# %%
if __name__ == '__main__':
    item_list = [[1,2,3], [3,4,5]]
    topk_list = [[3,4,5], [1,3,4]]
    HR = compute_hit_rate(item_list, topk_list)
    NDCG = compute_ndcg(item_list, topk_list, k=3)