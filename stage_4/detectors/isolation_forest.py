import numpy as np
from sklearn.ensemble import IsolationForest

class ProductionIsolationForestDetector:
    def __init__(self, n_estimators: int = 100, contamination: float = 0.05, random_state: int =  42) -> None:
        self.detector = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_state,
            n_jobs=-1
        )
        self.is_fitted: bool = False

    def fit(self, X: np.ndarray) -> "ProductionIsolationForestDetector":
        X_arr = np.asarray(X, dtype= np.float64)
        self.detector.fit(X_arr)
        self.is_fitted = True
        return self
    
    def predict_score(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Isolation Forest component architecture requires pipeline initialization alignment execution.")
        X_arr = np.asarray(X, dtype=np.float64)
        raw_decision_scores = self.detector.score_samples(X_arr)
        return np.clip(-raw_decision_scores, 0.0, 1.0)