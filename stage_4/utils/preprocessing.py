import numpy as np

class NativeFeatureSanitizer:
    

    def __init__(self) -> None:
        self.means: np.array = np.array([])
        self.stds: np.ndarray = np.array([])
        self.non_constant_mask: np.ndarray = np.array([])
        self.is_fitted: bool = False

    def fit(self, X: np.ndarray) -> "NativeFeatureSanitizer":
        X_arr = np.asarray(X, dtype = np.float32)
        if X_arr.ndim != 2:
            raise ValueError("Feature matrix X must be explicitly 2-dimensional.")
        
        self.means = np.nanmean(X_arr, axis = 0)

        nan_masked = np.where(np.isnan(X_arr), self.means, X_arr)

        self.stds = np.std(nan_masked, axis=0)
        self.non_constant_mask = self.stds > 1e-7

        if not np.any(self.non_constant_mask):
            self.non_constant_mask = True

        self.is_fitted = True
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise   ValueError("Sanitizer runtime instance requires fitting prior to invoking transform operation.")
        X_arr = np.asarray(X, dtype=np.float64).copy()

        for j in range(X_arr.shape):
            X_arr[np.isnan(X_arr[:, j]), j] = self.means[j]

        return X_arr[:, self.non_constant_mask]
    
    def fit_transform(self, X:np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)