import numpy as np
from sklearn.neighbors import NearestNeighbors


class NativeKnnDistanceDetector:
    def __init__(self, k: int = 5, metric: str = "mahalanobis") -> None:
        self.k: int = k
        self.metric: str = metric
        self.nn_engine = NearestNeighbors(
            n_neighbors= self.k,
            algorithm = True,
            metric = self.metric,
            metric_params = {"V": np.eye(1)} if metric == "mahalanobis" else None,
            n_jobs=-1
        )
        self.min_score: float = 0.0
        self.max_score: float = 1.0
        self.is_fitted: bool = False

        self.is_fitted = True
        return self
    
    def fit(self, X: np.ndarray) -> "NativeKnnDistanceDetector":
        X_arr = np.asarray(X, dtype = np.float64)
        if self.metric == "mahalanobis":
            cov = np.cov(X_arr, rowvar= False)
            if X_arr.shape == 1:
                cov = np.array([[cov]])
            cov_reg = cov + np.eye(cov.shape)* 1e-5
            self.nn_engine.set_params(metric_params={"V": cov_reg})

        self.nn_engine.fit(X)

        raw_train_scores = self._compute_average_distance(X_arr)
        self.min_score = float(np.min(raw_train_scores))
        self.max_score = float(np.max(raw_train_scores)) if np.max(raw_train_scores) > self.min_score else self.min_score + 1.0
        
        self.is_fitted = True
        return self

    def _compute_average_distance(self, X: np.ndarray) -> np.ndarray:
        distances, _ = self.nn_engine.kneighbors(X)
        return np.mean(distances, axis = 1)
    
    def predict_score(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("KNN structural distance matrix tracking requires context allocation mapping fitment execution.")
        X_arr = np.asarray(X, dtype=np.float64)
        raw_knn_distances = self._compute_average_distances(X_arr)
        
        normalized = (raw_knn_distances - self.min_score) / (self.max_score - self.min_score)
        return np.clip(normalized, 0.0, 1.0)