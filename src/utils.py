"""utils.py -- 数据读取与清洗。

一个"输入"= 一段 96 h、5 min 采样的多传感器序列（1152 行 × 21 列），
不含时间戳、不含机组编号。所有处理都只依赖这段序列本身。
"""
from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

# 21 路测点
SENSORS = [
    "RQDQLL", "FEP", "ETT", "BOPP", "FWF", "DLDRKWD", "DLDCKWD", "FSDRKYL",
    "FSDCKYL", "FSDCKGW", "BFP", "FLL", "DJFDQKD", "GRQRKYL", "GRQCKYL",
    "OPP", "POT", "SOD", "TZKZQGRD", "IAT", "SET",
]

TEMP = ["ETT", "POT", "IAT", "SET", "DLDRKWD", "DLDCKWD", "FSDCKGW"]
PRESS = ["FEP", "BOPP", "FSDRKYL", "FSDCKYL", "GRQRKYL", "GRQCKYL", "OPP", "BFP"]

# 这些测点在"运行"状态下出现 0 是坏值；流量类不适用
ZERO_INVALID = {c: True for c in TEMP + PRESS}
ZERO_INVALID.update({"SOD": True, "FLL": True, "RQDQLL": False, "FWF": False,
                     "DJFDQKD": False, "TZKZQGRD": False})

WIN_ROWS = 1152          # 96 h / 5 min


def load_window_csv(path: str) -> pd.DataFrame:
    """读入一段序列，统一列名与类型。"""
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = [c.strip().lstrip("\ufeff") for c in df.columns]
    df = df.reindex(columns=SENSORS).apply(pd.to_numeric, errors="coerce")
    df = df.reset_index(drop=True)
    df.index = pd.date_range("2000-01-01", periods=len(df), freq="5min")
    return df


def state_machine(raw: pd.DataFrame, gas_ref: float) -> pd.DataFrame:
    """依据燃气流量把每个时刻分成 停机/启动/停止/调整/稳定运行。"""
    gas, fw = raw["RQDQLL"], raw["FWF"]
    g = gas.fillna(0.0)
    f = fw.fillna(0.0)
    ok = bool(gas_ref and np.isfinite(gas_ref))
    on = (g > 0.20 * gas_ref) if ok else pd.Series(False, index=raw.index)
    if ok:
        fw_ref = fw[fw > 0].median() if (fw > 0).any() else np.nan
        if np.isfinite(fw_ref) and fw_ref > 0:
            on = on | (f > 0.20 * fw_ref)
    on = np.array(on.fillna(False).to_numpy(bool), dtype=bool, copy=True)
    for i in np.where((~on[1:-1]) & on[:-2] & on[2:])[0] + 1:   # 填单点抖动
        on[i] = True
    d = np.diff(np.r_[0, on.astype(int), 0])
    b_start, b_end = np.where(d == 1)[0], np.where(d == -1)[0]
    start = np.zeros(len(on), bool)
    stop = np.zeros(len(on), bool)
    for s, e in zip(b_start, b_end):
        start[s:min(s + 12, e)] = True
        stop[max(s, e - 12):e] = True
    on_s = pd.Series(on, index=raw.index)
    med6 = pd.Series(g).rolling(72, center=True, min_periods=12).median()
    dev = (pd.Series(g) - med6).abs() / max(gas_ref, 1e-9)
    band = pd.Series(g).between(0.75 * gas_ref, 1.25 * gas_ref) if ok \
        else pd.Series(False, index=raw.index)
    ramp = on_s & ((dev > 0.20) | ~band)
    run = on_s & ~pd.Series(start, index=raw.index) & ~pd.Series(stop, index=raw.index) & ~ramp
    return pd.DataFrame(dict(on=on_s, off=pd.Series(~on, index=raw.index),
                             start=pd.Series(start, index=raw.index),
                             stop=pd.Series(stop, index=raw.index),
                             ramp=ramp, run=run))


def clean(df: pd.DataFrame, run: pd.Series) -> pd.DataFrame:
    """按开机状态处理哨兵值、指定测点零值及满足连续零值条件的流量。"""
    out = df.mask(df <= -900).copy()
    for c in SENSORS:
        if ZERO_INVALID.get(c, False):
            out.loc[(out[c] == 0) & run, c] = np.nan
    for c in ("RQDQLL", "FWF"):
        z = out[c] == 0
        short = z & z.shift(1, fill_value=False) & z.shift(-1, fill_value=False)
        out.loc[short & run, c] = np.nan
    return out


def frozen_mask(df: pd.DataFrame, run: pd.Series) -> pd.DataFrame:
    """稳定运行且可识别正常波动时，标记连续相同值计数达到 24 的点。"""
    out = pd.DataFrame(False, index=df.index, columns=df.columns)
    rv = run.to_numpy(bool)
    for c in df.columns:
        v = df[c].to_numpy(float)
        runlen = np.ones(len(v))
        for i in range(1, len(v)):
            if np.isfinite(v[i]) and np.isfinite(v[i - 1]) and v[i] == v[i - 1]:
                runlen[i] = runlen[i - 1] + 1
        sd = pd.Series(v).rolling(24, min_periods=8).std()
        typ = np.nanmedian(sd[rv]) if rv.any() else np.nan
        if not np.isfinite(typ) or typ <= 1e-9:
            continue
        out[c] = (runlen >= 24) & rv
    return out


def prepare_window(raw: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """清洗一段序列，返回 (清洗后的 5 min 帧, 燃气参考值)。"""
    gas = raw["RQDQLL"]
    gas_ref = float(gas[gas > 0].median()) if (gas > 0).any() else np.nan
    st = state_machine(raw, gas_ref)
    df = clean(raw, st["on"])
    fz = frozen_mask(df, st["run"])
    return df.mask(fz), gas_ref
