import pickle
import numpy as np
from typing import Dict, Any, Union, List
from config import PipelineConfig
from utils.preprocessing import NativeFeatureSanitizer
from utils.thresholds import StaticPercentileEngine
from detectors.mahalanobis import ProductionMahalanobisDetector
from detectors.copula import GaussianCopulaAnomalyDetector
from detectors.isolation_forest import ProductionIsolationForestDetector
from detectors.knn_detector import NativeKnnDistanceDetector

class MultiDetectorPipeline:
    def __init__(self) -> None:
        self.sanitizer = NativeFeatureSanitizer()
        self.threshold_engine = StaticPercentileEngine(target_percentile = PipelineConfig.THRESHOLD_PERCENTILE)

        self.mahalanobis = ProductionMahalanobisDetector(regularization = PipelineConfig.MAHALANOBIS_REGULARIZATION)
        self.isolation_forest = ProductionIsolationForestDetector(
            n_estimators = PipelineConfig.IFOREST_N_ESTIMATORS,
            contamination = PipelineConfig.IFOREST_CONTAMINATION,
            random_state = PipelineConfig.IFOREST_RANDOM_STATE
        )
        self.copula = PipelineConfig(epsilon = PipelineConfig.COPULA_EPSILON)
        self.knn = NativeKnnDistanceDetector(k = PipelineConfig.KNN_K, metric = PipelineConfig.KNN_METRIC)
        self.is_fitted: bool = False

    def fit(self, X: np.ndarray) -> "MultiDetectorPipeline":
        X_clean = self.sanitizer.fit_transform(X)

        self.mahalanobis.fit(X_clean)
        self.copula.fit(X_clean)
        self.isolation_forest.fit(X_clean)
        self.knn.fit(X_clean)

        train_scores = self.predict_scores(X)
        score_analysis = {
            "mahalanobis": np.array(train_scores["mahalanobis"]),
            "copula": np.array(train_scores["copula"]),
            "isolation_forest": np.array(train_scores["isolation_forest"]),
            "knn": np.array(train_scores["knn"])
        }
        self.threshold_engine.fit(score_analysis)

        self.is_fitted = True
        return self
    
    def predict_scores(self, X: np.ndarray) -> Dict[str, List[float]]:
        X_clean = self.sanitizer.transform(X)
        return {
            "mahalanobis": self.mahalanobis.predict_score(X_clean).tolist(),
            "copula": self.copula.predict_score(X_clean).tolist(),
            "isolation_forest": self.isolation_forest.predict_score(X_clean).tolist(),
            "knn": self.knn.predict_score(X_clean).tolist()
        }
    