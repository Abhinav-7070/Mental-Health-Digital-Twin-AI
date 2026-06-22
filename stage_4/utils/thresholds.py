import numpy as np
from typing import Dict

class StaticPercentileEngine:

    def __init__(self, target_percentile: float = 95.0) -> None:
        self.target_percentile: float = target_percentile
        self.thresholds: Dict[str, float] = {}
        self.is_fitted: bool = False
    
    def fit(self, scored_dictionary: Dict[str, np.ndarray]) -> "StaticPercentileEngine":
        for key, scores in scored_dictionary.items():
            self.thresholds[key] = float(np.percentile(scores, self.target_percentile))
        self.is_fitted = True
        return self
    
    def eval_status(self, key: str, score: float) -> bool:
        if not self.is_fitted:
            return False
        return bool(score >= self.thresholds.get(key, float("inf")))