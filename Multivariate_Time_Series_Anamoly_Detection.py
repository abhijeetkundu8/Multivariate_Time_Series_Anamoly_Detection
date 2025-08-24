#!/usr/bin/env python
# coding: utf-8

# In[5]:


"""
- Train on a normal period
- Score anomalies on full analysis window
- Add Abnormality_score (0-100) + top_feature_1..7
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from typing import List, Tuple, Dict, Optional


# ---------------- Utilities ---------------- #

def _safe_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Convert to numeric where possible; non-convertible -> NaN."""
    out = df.copy()
    for c in out.columns:
        if not pd.api.types.is_numeric_dtype(out[c]):
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def _forward_fill_and_interp(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing values with forward fill + interpolation."""
    df2 = df.copy().ffill()
    if isinstance(df2.index, pd.DatetimeIndex):
        try:
            df2 = df2.interpolate(method="time", limit_direction="both")
        except Exception:
            df2 = df2.interpolate(limit_direction="both")
    else:
        df2 = df2.interpolate(limit_direction="both")
    for c in df2.columns:
        if df2[c].isna().any():
            df2[c] = df2[c].fillna(df2[c].median())
    return df2


def _ecdf_percentile(train_scores: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """Map scores to percentiles wrt training distribution."""
    train_sorted = np.sort(train_scores)
    n = len(train_sorted)
    ranks = np.searchsorted(train_sorted, scores, side="right")
    pct = 100.0 * (ranks / max(1, n))
    return np.clip(pct + 1e-6, 0.0, 100.0)


# ---------------- Data Processor ---------------- #

class DataProcessor:
    def __init__(self, time_col: Optional[str] = None):
        self.time_col = time_col

    def detect_time_column(self, df: pd.DataFrame) -> str:
        if self.time_col and self.time_col in df.columns:
            col = self.time_col
        else:
            candidates = ["timestamp", "time", "date", "datetime", df.columns[0]]
            col = None
            for c in candidates:
                if c in df.columns:
                    col = c
                    break
            if col is None:
                raise ValueError("No timestamp column found")
        df[col] = pd.to_datetime(df[col], errors="raise")
        return col

    def prepare(self, df: pd.DataFrame,
                train_start: str, train_end: str,
                analysis_start: str, analysis_end: str
               ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        time_col = self.detect_time_column(df)
        df = df.sort_values(time_col).set_index(time_col)
        numeric = _safe_numeric(df.select_dtypes(include=[np.number]))
        numeric = _forward_fill_and_interp(numeric)

        df_train = numeric.loc[train_start:train_end]
        df_analysis = numeric.loc[analysis_start:analysis_end]
        if df_train.empty or df_analysis.empty:
            raise ValueError("Training or analysis window is empty")
        return numeric, df_train, df_analysis


# ---------------- PCA-based Detector ---------------- #

class PCABasedDetector:
    def __init__(self, var_threshold: float = 0.95, weight_pca: float = 0.5, weight_z: float = 0.5):
        self.var_threshold = var_threshold
        self.weight_pca = weight_pca
        self.weight_z = weight_z

    def fit(self, train: pd.DataFrame):
        self.columns_ = list(train.columns)
        self.mu_ = train.mean(axis=0).values
        self.sigma_ = train.std(axis=0, ddof=0).replace(0, 1).values
        Z = (train.values - self.mu_) / self.sigma_

        pca_full = PCA(svd_solver="full").fit(Z)
        cum = np.cumsum(pca_full.explained_variance_ratio_)
        n_comp = max(1, np.searchsorted(cum, self.var_threshold) + 1)
        self.pca_ = PCA(n_components=n_comp, svd_solver="full").fit(Z)

        z2, resid2, z_sum, resid_sum = self._raw_score_components(Z)
        self.train_raw_ = self._combine_raw(z2, resid2, z_sum, resid_sum)

    def _raw_score_components(self, Z):
        Z_hat = self.pca_.inverse_transform(self.pca_.transform(Z))
        resid = Z - Z_hat
        z2 = Z**2
        resid2 = resid**2
        z_sum = z2.sum(axis=1)
        resid_sum = resid2.sum(axis=1)
        return z2, resid2, z_sum, resid_sum

    def _combine_raw(self, z2, resid2, z_sum, resid_sum):
        return self.weight_z*np.sqrt(z_sum) + self.weight_pca*np.sqrt(resid_sum) + 1e-9

    def score_and_attribution(self, df: pd.DataFrame):
        X = df[self.columns_].values
        Z = (X - self.mu_) / self.sigma_
        z2, resid2, z_sum, resid_sum = self._raw_score_components(Z)
        total_raw = self._combine_raw(z2, resid2, z_sum, resid_sum)

        scores = _ecdf_percentile(self.train_raw_, total_raw)

        # Compress so training mean <10
        scores = np.where(scores <= 80, scores*0.2,
                          80*0.2 + ((scores-80)**2)*(1-80*0.2)/(20**2))
        scores = np.clip(scores, 0, 100)

        # Feature contributions
        z_share = np.where(z_sum[:,None]>0, z2/z_sum[:,None], 0)
        r_share = np.where(resid_sum[:,None]>0, resid2/resid_sum[:,None], 0)
        share = self.weight_z*z_share + self.weight_pca*r_share
        names = np.array(self.columns_)
        top_feats = []
        for row_share in share:
            mask = row_share >= 0.01
            cols = names[mask]
            contribs = row_share[mask]
            order = np.lexsort((cols, -contribs))
            ranked = [cols[j] for j in order[:7]]
            while len(ranked) < 7: ranked.append("")
            top_feats.append(ranked)
        return scores, top_feats


# ---------------- Pipeline ---------------- #

def run_pipeline(input_csv: str, output_csv: str,
                 time_col: Optional[str] = None,
                 train_start: str = "2004-01-01 00:00",
                 train_end: str = "2004-01-05 23:59",
                 analysis_start: str = "2004-01-01 00:00",
                 analysis_end: str = "2004-01-19 07:59") -> Dict[str, float]:
    
    df = pd.read_csv(input_csv)
    processor = DataProcessor(time_col)
    full, df_train, df_analysis = processor.prepare(df, train_start, train_end, analysis_start, analysis_end)

    detector = PCABasedDetector()
    detector.fit(df_train)

    scores, top_feats = detector.score_and_attribution(df_analysis)

    out = df.copy()
    time_col_used = processor.detect_time_column(df)
    out = out.sort_values(time_col_used).set_index(time_col_used)

    out["Abnormality_score"] = 0.0
    for k in range(1,8):
        out[f"top_feature_{k}"] = ""

    idx = df_analysis.index
    out.loc[idx, "Abnormality_score"] = scores
    for i, ts in enumerate(idx):
        feats = top_feats[i]
        for k in range(7):
            out.loc[ts, f"top_feature_{k+1}"] = feats[k]

    out = out.reset_index()
    out.to_csv(output_csv, index=False)

    tr_scores, _ = detector.score_and_attribution(df_train)
    return {"training_mean": float(np.mean(tr_scores)), "training_max": float(np.max(tr_scores))}


# In[7]:


metrics = run_pipeline(
    input_csv="TEP_Train_Test.csv",
    output_csv="TEP_with_anomaly_output.csv"
)

print(metrics)

import pandas as pd
df = pd.read_csv("TEP_with_anomaly_output.csv")
df.head()


# In[ ]:




