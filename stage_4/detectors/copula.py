import numpy as np
from typing import Dict, Any, List
from scipy.stats import norm, rankdata
from scipy.linalg import pinvh

class GaussianCopulaAnomalyDetector:
    def __init__(self, epsilon: float = 1e-6) -> None:
        self.epsilon: float = epsilon
        self.train_marginal_data: List[np.ndarray] = []
        self.copula_covariance_pinv: np.ndarray = np.array([])
        self.copula_det_log: float = 0.0
        self.max_score: float = 1.0
        self.min_score: float = 0.0
        self.n_features: int = 0
        self.is_fitted: bool = False

    def fit(self, X: np.ndarray) -> "GaussianCopulaAnomalyDetector":
        X_arr = np.asarray(X, dtype= np.float64)
        n_samples, self.n_features = X_arr.shape

        self.train_marginal_data = [X_arr[:, j] for j in range(self.n_features)]

        Z_space = np.zeros_like(X_arr)
        for j in range(self.n_features):
            ranks = rankdata(X_arr[:, j], method="average")
            uniform_values = ranks / (n_samples + 1)
            Z_space[:, j] = norm.ppf(np.clip(uniform_values, self.epsilon, 1.0 - self.epsilon))

            R = np.corrcoef(Z_space, rowvar=False)
        if self.n_features == 1:
            R = np.array([[1.0]])

        R_reg = R + np.eye(self.n_features) * 1e-5

        sign, logdet = np.linalg.slogdet(R_reg)
        self.copula_det_log = logdet if sign > 0 else 0.0
        self.copula_covariance_pinv = pinvh(R_reg)

        raw_train_scores = self._calculate_negative_log_likelihood(X_arr)
        self.min_score = float(np.min(raw_train_scores))
        self.max_score = float(np.max(raw_train_scores)) if np.max(raw_train_scores) > self.min_score else self.min_score + 1.0
        
        self.is_fitted = True
        return self
    
    def _calculate_negative_log_likelihood(self, X: np.ndarray) -> np.ndarray:
        n_samples = X.shape
        Z_space = np.zeros((n_samples, self.n_features))
        
        for j in range(self.n_features):
            train_col = self.train_marginal_data[j]
            eval_col = X[:, j]
           
            ranks = np.searchsorted(np.sort(train_col), eval_col, side="right")
            uniform_values = np.clip(ranks / (len(train_col) + 1), self.epsilon, 1.0 - self.epsilon)
            Z_space[:, j] = norm.ppf(uniform_values)
     
        left_dot = np.dot(Z_space, self.copula_covariance_pinv - np.eye(self.n_features))
        quadratic_score = 0.5 * np.sum(left_dot * Z_space, axis=1)
        
       
        return quadratic_score + 0.5 * self.copula_det_log

    def predict_score(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Gaussian Copula instance engine must execute training fit cycle before inference processing.")
        X_arr = np.asarray(X, dtype=np.float64)
        raw_nll = self._calculate_negative_log_likelihood(X_arr)
        
        normalized = (raw_nll - self.min_score) / (self.max_score - self.min_score)
        return np.clip(normalized, 0.0, 1.0)