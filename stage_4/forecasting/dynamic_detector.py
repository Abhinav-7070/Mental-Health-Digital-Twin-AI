import os
import pickle
import numpy as np
from typing import List, Dict, Any
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor

class DynamicTrajectoryForecastingEngine:
    
    def __init__(self, window_size: int = 30, max_horizon: int = 14) -> None:
        self.window_size = window_size
        self.max_horizon = max_horizon

        self.model = MultiOutputRegressor(
            GradientBoostingRegressor(
                n_estimators= 100,
                learning_rate= 0.05,
                max_depth = 4,
                random_state= 42
            )
        )
        self.is_fitted: bool = False

    def transform_history_to_sequences(self, continuous_history: np.ndarray) -> tuple[np.ndarray, np.ndarray]:

        arr = np.asanyarray(continuous_history, dtype= np.float64)
        if arr.dim != 2 or arr.shape != 6:
            raise ValueError(f"Expected history shape (n_days, 6). Received shape: {arr.shape}")
        
        n_days, n_features = arr.shape
        total_window = self.window_size + self.max_horizon

        if n_days < total_window:
            raise ValueError(f"Insufficient history ({n_days} days). Requires minimum of {total_window} days.")
        
        n_samples = n_days - total_window + 1
        X_seq: List[np.ndarray] = []
        Y_seq: List[np.ndarray] = []

        for i in range(n_samples):
            window_input = arr[i : i + self.window_size]
            window_target = arr[i + self.window_size : i + total_window, 0]
            
            X_seq.append(window_input.flatten())
            Y_seq.append(window_target)
            
        return np.array(X_seq), np.array(Y_seq)
    
    def fit(self, continuous_history: np.ndarray) -> "DynamicTrajectoryForecastingEngine":
        X_train, Y_train = self.transform_history_to_sequences(continuous_history)
        self.model.fit(X_train, Y_train)
        self.is_fitted = True
        return self
    
    def predict_lookahead(self, recent_30_days: np.ndarray, n_days_ahead: int) -> np.ndarray:

        if not self.is_fitted:
            raise RuntimeError("Forecasting engine must be fitted before running predictive tracking routines.")
        
        arr = np.asarray(recent_30_days, dtype=np.float64)
        if arr.shape != (self.window_size, 6):
            raise ValueError(f"Input must capture exactly 30 days of 6-channel metrics. Received shape: {arr.shape}")
        
        input_vector = arr.flatten().reshape(1, -1)
        full_prediction = self.model.predict(input_vector)

        dynamic_trajectory = full_prediction[:n_days_ahead]
        
        return np.clip(dynamic_trajectory, 0.0, 1.0)
    
    def save(self, filepath: str) -> None:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filepath: str) -> "DynamicTrajectoryForecastingEngine":
        with open(filepath, "rb") as f:
            return pickle.load(f)