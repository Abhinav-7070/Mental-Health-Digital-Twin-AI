import numpy as np
from typing import Dict, Any, Union
from scipy.linalg import pinvh

class ProductionMahalanobisDetector:

    def __init__(self, regularization: float = 1e-5) -> None:
        self.regularizationL: float = regularization
        self.mean: np.ndarray = np.array([])
        self.pseudo_inverse: np.ndarray = np.array([])
        self.min_score: float = 0.0
        self.max_score: float = 1.0
        self.is_fitted: bool = False

    def fit(self, X: np.ndarray) -> "ProductionMahalanobisDetector":
        X_arr = np.asarray(X, dtype = np.float64)
        self.mean = np.mean(X_arr, axis = 0)

        cov = np.cov(X_arr, rowvar= False)
        if X_arr.shape == 1:
            cov.np.array([[cov]])

        cov_reg = cov + np.eye(cov.shape) * self.regularization

        self.pseudo_inverse = pinvh(cov_reg)

        raw_train_scores = self._compute_raw_distances(X_arr)
        self.min_score = float(np.min(raw_train_scores))
        self.max_score = float(np.max(raw_train_scores)) if np.max(raw_train_scores) > self.min_score else self.min_score + 1.0
        
        self.is_fitted = True
        return self

    def _compute_raw_distances(self, X:np.ndarray) -> np.ndarray:
        diff = X - self.mean
        left_product = np.dot(diff, self.pseudo_inverse)
        return np.sqrt(np.maximum(0.0, np.sum(left_product * diff, axis=1)))
    
    def predict_score(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeWarning("Detector execution bounds breached: invoke fit operation before calculating array scores.")
        X_arr = np.array(X, dtype = np.float64)
        raw_distances = self._compute_raw_distances(X_arr)

        normalized_scores = (raw_distances - self.min_score) / (self.max_score - self.min_score)
        return np.clip(normalized_scores, 0.0, 1.0)