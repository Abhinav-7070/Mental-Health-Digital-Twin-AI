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
        self.copula = GaussianCopulaAnomalyDetector(epsilon = PipelineConfig.COPULA_EPSILON)
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
    
    def predict(self, X: np.ndarray) -> Dict[str, Any]:
        scores_dict = self.predict_scores(X)

        overall_risk_scores = []
        is_anomaly_flags = []
        detailed_records = []

        w = PipelineConfig.DETECTOR_WEIGHTS

        n_samples = len(X)
        for i in range(n_samples):
            m_s = scores_dict["mahalanobis"][i]
            c_s = scores_dict["copula"][i]
            if_s = scores_dict["isolation"][i]
            k_s = scores_dict["knn"][i]

            risk = (m_s * w["mahalanobis"] + 
                    c_s * w["copula"] + 
                    if_s * w["isolation_forest"] + 
                    k_s * w["knn"])
            
            overall_risk_scores.append(risk)

            m_anom = self.threshold_engine.eval_status("mahalanobis", m_s)
            c_anom = self.threshold_engine.eval_status("copula", c_s)
            if_anom = self.threshold_engine.eval_status("isolation_forest", if_s)
            k_anom = self.threshold_engine.eval_status("knn", k_s)

            is_anomaly_flags.append(m_anom, c_anom, if_anom, k_anom)

            detailed_records.append({
                "mahalanobis": m_s,
                "copula": c_s,
                "isolation_forest": if_s,
                "knn": k_s
            })

            return {
                "metrics_summary": detailed_records,
                "overall_risk_score": overall_risk_scores,
                "is_anomaly": is_anomaly_flags
            }
        
        def save(self, filepath: str) -> None:
            with open(filepath, "wb") as output_stream:
                pickle.dump(self, output_stream)

        @staticmethod
        def load(filepath: str) -> "MultiDetectorPipeline"        :
            with open(filepath, "rb") as input_stream:
                return pickle.load(input_stream)       

