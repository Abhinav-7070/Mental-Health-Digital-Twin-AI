import os
from typing import Dict, List, Any

class PipelineConfig:
    MAHALANOBIS_REGULARIZATION: float = 1e-5

    COPULA_EPSILON: float = 1e-6
    
    IFOREST_N_ESTIMATORS: int = 100
    IFOREST_CONTAMINATION: float = 0.05
    IFOREST_RANDOM_STATE: int = 42

    KNN_K: int = 5
    KNN_ALGORITHM: str = "auto"
    KNN_METRIC: str = "mahalanobis"

    DETECTOR_WEIGHTS: Dict[str, float] = {
        "mahalanobis": 0.35,
        "copula": 0.35,
        "isolation_forest": 0.15,
        "knn": 0.15
    }

    THRESHOLD_PERCENTILE: float = 95.0

    OUTPUT_DIR: str = "outputs"
    MODEL_PATH: str = os.path.join(OUTPUT_DIR, "anomaly_pipeline.pkl")
    PLOT_PATH_PREFIX: str = os.path.join(OUTPUT_DIR, "anomaly_plot_")

    @classmethod
    def validate(cls) -> None:

        if not abs(sum(cls.DETECTOR_WEIGHTS.values()) - 1.0) < 1e-6:
            raise ValueError("Ensemble configuration weights must sum up to exactly 1.0.")
        if not (0.0 < cls.IFOREST_CONTAMINATION < 0.5):
            raise ValueError("Isolation Forest contamination rate must sit inside the interval (0.0, 0.5).")
        if cls.KNN_K <= 0:
            raise ValueError("KNN parameter k must evaluate to a positive non-zero integer matrix pointer.")

PipelineConfig.validate()