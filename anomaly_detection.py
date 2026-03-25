"""
Module 3: Anomaly Detection
Three complementary methods:
  1. Z-score on forecast residuals (statistical)
  2. Isolation Forest (ML-based, unsupervised)
  3. IQR-based rolling window detector
Combined ensemble vote for robust detection.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, f1_score)


class ZScoreDetector:
    """
    Detect anomalies using rolling Z-score on forecast residuals.
    An anomaly is flagged when |residual| > threshold * rolling_std.
    """

    def __init__(self, window=24, threshold=3.0):
        self.window    = window
        self.threshold = threshold
        self.fitted    = False

    def fit(self, residuals: pd.Series):
        self.rolling_mean = residuals.rolling(self.window, min_periods=6).mean()
        self.rolling_std  = residuals.rolling(self.window, min_periods=6).std()
        self.fitted = True

    def predict(self, residuals: pd.Series) -> np.ndarray:
        """Return 1=anomaly, 0=normal."""
        if not self.fitted:
            self.fit(residuals)
        z = (residuals - self.rolling_mean) / (self.rolling_std + 1e-6)
        return (np.abs(z) > self.threshold).astype(int).values

    def score(self, residuals: pd.Series) -> np.ndarray:
        """Return absolute Z-scores."""
        z = (residuals - self.rolling_mean) / (self.rolling_std + 1e-6)
        return np.abs(z).values


class IsolationForestDetector:
    """
    Isolation Forest anomaly detector.
    Features: demand, residual, hour, day_of_week, rolling stats.
    """

    def __init__(self, contamination=0.01, n_estimators=100, random_state=42):
        self.model = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=random_state,
        )
        self.scaler  = StandardScaler()
        self.feature_cols = None
        self.fitted  = False

    def _build_features(self, df):
        feats = pd.DataFrame({
            'demand_mw':    df['demand_mw'],
            'residual':     df['residual'],
            'abs_residual': df['residual'].abs(),
            'hour':         df['hour'],
            'day_of_week':  df['day_of_week'],
            'temperature_c':df['temperature_c'],
            'rolling_24h':  df['rolling_24h'],
        })
        return feats.fillna(0)

    def fit(self, df):
        X = self._build_features(df)
        self.feature_cols = X.columns.tolist()
        X_s = self.scaler.fit_transform(X)
        self.model.fit(X_s)
        self.fitted = True

    def predict(self, df) -> np.ndarray:
        """Return 1=anomaly, 0=normal."""
        X = self._build_features(df)[self.feature_cols]
        X_s = self.scaler.transform(X)
        preds = self.model.predict(X_s)   # sklearn: -1=anomaly, 1=normal
        return (preds == -1).astype(int)

    def score(self, df) -> np.ndarray:
        """Return anomaly score (higher = more anomalous)."""
        X = self._build_features(df)[self.feature_cols]
        X_s = self.scaler.transform(X)
        return -self.model.score_samples(X_s)   # negate so higher = anomaly


class IQRDetector:
    """
    Rolling IQR-based outlier detector on residuals.
    Flags points outside [Q1 - k*IQR, Q3 + k*IQR] in a rolling window.
    """

    def __init__(self, window=168, k=2.5):
        self.window = window
        self.k = k

    def predict(self, residuals: pd.Series) -> np.ndarray:
        roll = residuals.rolling(self.window, min_periods=24)
        q1  = roll.quantile(0.25)
        q3  = roll.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - self.k * iqr
        upper = q3 + self.k * iqr
        flags = ((residuals < lower) | (residuals > upper)).astype(int)
        return flags.values


class EnsembleAnomalyDetector:
    """
    Ensemble of ZScore + IsolationForest + IQR detectors.
    Anomaly is flagged when >= min_votes detectors agree.
    """

    def __init__(self, contamination=0.01, z_threshold=3.0,
                 iqr_k=2.5, min_votes=2):
        self.zscore_det  = ZScoreDetector(threshold=z_threshold)
        self.iforest_det = IsolationForestDetector(contamination=contamination)
        self.iqr_det     = IQRDetector(k=iqr_k)
        self.min_votes   = min_votes
        self.fitted      = False

    def fit(self, df, residuals: pd.Series):
        """Fit all detectors on training data."""
        self.zscore_det.fit(residuals)
        self.iforest_det.fit(df)
        self.fitted = True
        print("  Ensemble anomaly detector fitted.")

    def predict(self, df, residuals: pd.Series) -> pd.DataFrame:
        """
        Predict anomalies. Returns DataFrame with per-detector flags
        and ensemble decision.
        """
        # Re-fit zscore rolling stats on current residuals
        self.zscore_det.fit(residuals)
        z_flags   = self.zscore_det.predict(residuals)
        if_flags  = self.iforest_det.predict(df)
        iqr_flags = self.iqr_det.predict(residuals)

        votes = z_flags + if_flags + iqr_flags
        ensemble = (votes >= self.min_votes).astype(int)

        result = pd.DataFrame({
            'zscore_flag':    z_flags,
            'iforest_flag':   if_flags,
            'iqr_flag':       iqr_flags,
            'votes':          votes,
            'anomaly_pred':   ensemble,
            'zscore_score':   self.zscore_det.score(residuals),
            'iforest_score':  self.iforest_det.score(df),
        })
        return result

    def evaluate(self, df, residuals, y_true):
        """Print classification metrics against ground truth."""
        results = self.predict(df, residuals)
        y_pred  = results['anomaly_pred'].values

        print("  Confusion Matrix:")
        cm = confusion_matrix(y_true, y_pred)
        print(f"    TN={cm[0,0]}  FP={cm[0,1]}")
        print(f"    FN={cm[1,0]}  TP={cm[1,1]}")
        print("\n  Classification Report:")
        print(classification_report(y_true, y_pred,
                                    target_names=['Normal', 'Anomaly']))
        f1  = f1_score(y_true, y_pred, zero_division=0)
        try:
            auc = roc_auc_score(y_true, results['iforest_score'].values)
        except Exception:
            auc = float('nan')
        print(f"  F1-Score: {f1:.4f}   ROC-AUC: {auc:.4f}")
        return results


if __name__ == '__main__':
    from data_generator import generate_dataset
    from forecasting_model import EnergyForecaster, FEATURE_COLS

    df = generate_dataset()
    df = df.dropna().reset_index(drop=True)

    split = -30 * 24
    df_train = df.iloc[:split].copy()
    df_test  = df.iloc[split:].copy()

    # Fit forecaster and compute residuals
    forecaster = EnergyForecaster(n_estimators=100)
    forecaster.fit(df_train)
    preds_train = forecaster.predict(df_train)
    preds_test  = forecaster.predict(df_test)

    df_train['residual'] = df_train['demand_mw'].values - preds_train['forecast_mw'].values
    df_test['residual']  = df_test['demand_mw'].values  - preds_test['forecast_mw'].values

    print("=== Training Ensemble Anomaly Detector ===")
    detector = EnsembleAnomalyDetector()
    detector.fit(df_train, df_train['residual'])

    print("\n=== Evaluating on Test Set ===")
    detector.evaluate(df_test, df_test['residual'], df_test['is_anomaly'].values)
