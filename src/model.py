"""model.py -- 打分器加载与判决。

判决只有一层：单个窗口的分数与一个固定阈值比较

    risk = phi(x) ;  fault = 1 if risk > tau else 0

`tau` 从模型文件读取，预测期间保持固定。
"""
from __future__ import annotations

import os
import pickle

import numpy as np
import pandas as pd


class Detector:
    """窗口打分 + 单阈值判断"""

    def __init__(self, model, features, med, iqr, tau):
        self.model = model
        self.features = list(features)
        self.med = np.asarray(med, float)
        self.iqr = np.asarray(iqr, float)
        self.tau = float(tau)

    # ---------------------------------------------------------------- scaling
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """按训练集的中位数/四分位距标准化，截断到 ±10，缺失补 0。"""
        M = X.reindex(columns=self.features).to_numpy(float)
        Z = (M - self.med) / self.iqr
        Z = np.clip(Z, -10.0, 10.0)
        return pd.DataFrame(np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0),
                            columns=self.features)

    # ---------------------------------------------------------------- predict
    def score(self, X: pd.DataFrame) -> np.ndarray:
        Z = self.transform(X)
        s = self.model.predict(Z)
        return np.asarray(s, float).ravel()

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (self.score(X) > self.tau).astype(int)


def load(model_dir: str) -> Detector:
    with open(os.path.join(model_dir, "model.pkl"), "rb") as f:
        pack = pickle.load(f)
    with open(os.path.join(model_dir, "scaler.pkl"), "rb") as f:
        sc = pickle.load(f)
    return Detector(pack["model"], pack["features"], sc["med"], sc["iqr"],
                    pack["tau"])
