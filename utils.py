from sklearn.metrics import roc_auc_score, roc_curve
import numpy as np

def compute_auroc(id_pps, ood_pps, normalize=False, return_curve=False):
    y = np.concatenate((np.ones_like(ood_pps), np.zeros_like(id_pps)))
    scores = np.concatenate((ood_pps, id_pps))
    if normalize:
        scores = (scores - scores.min()) / (scores.max() - scores.min())
    if return_curve:
        return roc_curve(y, scores)
    else:
        return 100*roc_auc_score(y, scores)

def compute_far(id_pps, ood_pps, rate=5):
    incorrect = len(id_pps[id_pps > np.percentile(ood_pps, rate)])
    return 100*incorrect / len(id_pps)
