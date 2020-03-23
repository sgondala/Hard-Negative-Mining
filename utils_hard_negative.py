import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_fscore_support

def print_f1_scores(predicted_scores, actual_scores):
    pred_scores_numpy = np.array(predicted_scores)
    all_prec_recall_scores = []
    for threshold_value in np.arange(0.1, 1, 0.1):
        prec_recall_scores = precision_recall_fscore_support(pred_scores_numpy < threshold_value, actual_scores, average='weighted')
        print("Pres_recall_scores ", threshold_value, prec_recall_scores)
        all_prec_recall_scores.append(prec_recall_scores)
    return all_prec_recall_scores