import numpy as np
from typing import Dict, Any

class AnalyticalEvaluationSuite:

    @staticmethod
    def compute_all(y_true: np.ndarray, y_score: np.ndarray) -> Dict[str, float]:
        y_t = np.asarray(y_true, dtype = np.int32)
        y_s = np.asarray(y_score, dtype = np.float64)


        desc_indices = np.argsort(y_s)[::-1]
        y_t_sorted = y_t[desc_indices]
        y_s_sorted = y_s[desc_indices]

        n_pos =  int(np.sum(y_t))
        n_neg = len(y_t) - n_pos

        if n_pos == 0 or n_neg == 0:
            auc_roc = 0.5
        else:
            tp_counts = np.cumsum(y_t_sorted)
            fp_counts = np.cumsum(1 - y_t_sorted)
            tps = tp_counts[y_t_sorted == 0]
            auc_roc = float(np.sum(tps) / (n_pos * n_neg))

        tp_trajectory = np.cumsum(y_t_sorted)
        fp_trajectory = np.cumsum(1 - y_t_sorted)

        precisions = tp_trajectory / (tp_trajectory + fp_trajectory)
        recalls = tp_trajectory / n_pos if n_pos > 0 else np.zeros_like(tp_trajectory)

        pr_auc = 0.0
        if n_pos > 0:
            prev_recall = 0.0
            for p, r in zip(precisions, recalls):
                pr_auc += p*(r - prev_recall)
                prev_recall = r

        cutoff_idx = max(0, int(len(y_t) * 0.05) - 1)
        thresh_val = y_s_sorted[cutoff_idx] if len(y_s_sorted) > 0 else 0.0
        y_pred = (y_s >= thresh_val).astype(np.int32)

        tp = int(np.sum((y_pred == 1) & (y_t == 1)))
        fp = int(np.sum((y_pred == 1) & (y_t == 0)))
        fn = int(np.sum((y_pred == 0) & (y_t == 1)))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            "roc_auc": float(auc_roc),
            "pr_auc": float(pr_auc),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1)
        }