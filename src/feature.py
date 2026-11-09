"""feature.py -- 单个 96 h 窗口 -> 1498 维特征向量。

特征分 11 族，全部只由当前这一段序列算出（不引用其它窗口、不引用批次统计）：
  1  偏差水平 l1/l3/l6/l12/l24      末 k 段的平均 z
  2  极值 am6/am24                 末 k 段的最大 |z|
  3  持续性 pers2 / dwin           |z|>2 的占比、首尾 24 h 差
  4  斜率 s6/s24/s48/s72            末 k 段线性斜率
  5  绝对/变异 absmed/abssd/cv6/cv96
  6  形状 zc/skew/kurt/sprd/ac/mdd
  7  跨尺度差 d16/d624/d124/acc
  8  自参照比值 rh_cv/rh_am/rh_abs/rh_zc   末 24 h 除以窗内前 72 h
  9  窗内秩 rank24/rank6           末 24 h/6 h 在窗内滚动均值中的百分位
 10  CUSUM 变点 cusp/cusn          窗内单边累积和最大值 / sqrt(n)
 11  长尺度 l48/l72                末 48/72 h 平均 z
另有 12 h 块形状 c6_*、参数协同 co_*、工况上下文 ctx_*、5 min 频谱 hf_*。
"""
from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import utils as C

S = C.SENSORS
WIN = 384
BASE = 96
# 计算水/蒸汽饱和温度所需的固定公式系数
_N = [0.11670521452767e4, -0.72421316703206e6, -0.17073846940092e2,
      0.12020824702470e5, -0.32325550322333e7, 0.14915108613530e2,
      -0.48232657361591e4, 0.40511340542057e6, -0.23855557567849,
      0.65017534844798e3]


def tsat_c(p_mpa) -> np.ndarray:
    p = np.clip(np.asarray(p_mpa, float), 0.000611, 22.06)
    b = p ** 0.25
    e = b * b + _N[2] * b + _N[5]
    f = _N[0] * b * b + _N[3] * b + _N[6]
    g = _N[1] * b * b + _N[4] * b + _N[7]
    d = 2 * g / (-f - np.sqrt(np.maximum(f * f - 4 * e * g, 0)))
    t = (_N[9] + d - np.sqrt((_N[9] + d) ** 2 - 4 * (_N[8] + _N[9] * d))) / 2
    return t - 273.15


def hfg_kjkg(p_mpa, tsat):
    tr = (np.asarray(tsat) + 273.15) / 647.096
    return 2257.0 * ((1 - tr) / (1 - 373.15 / 647.096)) ** 0.38


INVARIANTS = ["SFC", "SH_obs", "SH_gap", "x_energy", "dP_rad", "dP_rad_rel",
              "head", "head_rel", "wall_minus_POT", "dT_conv", "SET_minus_IAT",
              "dP_super", "level_valve", "Tsat", "OPP_rel", "BFP_rel"]
PARAMS = S + INVARIANTS

# (name, window length in 15-min samples)
LEVELS = [("l1", 4), ("l3", 12), ("l6", 24), ("l12", 48), ("l24", 96)]
EXTREMES = [("am6", 24), ("am24", 96)]
SLOPES = [("s6", 24), ("s24", 96)]

C6_SENSORS = ["RQDQLL", "FWF", "POT", "SET", "FSDCKGW", "FSDCKYL", "BFP",
              "DLDRKWD", "DLDCKWD", "FSDRKYL", "OPP"]


def to15(dfm: pd.DataFrame) -> pd.DataFrame:
    return dfm.resample("15min").mean()


def invariants(df15: pd.DataFrame, gas_ref: float) -> pd.DataFrame:
    out = pd.DataFrame(index=df15.index)
    gas = df15["RQDQLL"].where(df15["RQDQLL"] > 0)
    fw = df15["FWF"].where(df15["FWF"] > 0)
    opp = df15["OPP"].where(df15["OPP"] > 0)
    pot = df15["POT"].where(df15["POT"] > 0)
    out["SFC"] = gas / fw
    ts = tsat_c(opp.to_numpy(float))
    out["Tsat"] = ts
    out["SH_obs"] = pot.to_numpy(float) - ts
    out["SH_gap"] = df15["TZKZQGRD"] - out["SH_obs"]
    q = 35.6e3 * gas / (1000.0 * fw)
    p = np.where(np.isfinite(opp.to_numpy(float)), opp.to_numpy(float), 5.0)
    tsv = np.where(np.isfinite(ts), ts, 250.0)
    dh = (4.186 * tsv + hfg_kjkg(p, tsv)) - 4.186 * df15["ETT"].to_numpy(float)
    out["x_energy"] = 0.90 * q / dh
    out["dP_rad"] = df15["FSDRKYL"] - df15["FSDCKYL"]
    out["dP_rad_rel"] = out["dP_rad"] / opp
    out["head"] = df15["BOPP"] - df15["FEP"]
    out["head_rel"] = out["head"] / opp
    out["wall_minus_POT"] = df15["FSDCKGW"] - pot
    out["dT_conv"] = df15["DLDCKWD"] - df15["DLDRKWD"]
    out["SET_minus_IAT"] = df15["SET"] - df15["IAT"]
    out["dP_super"] = df15["GRQRKYL"] - df15["GRQCKYL"]
    out["level_valve"] = df15["FLL"] / df15["DJFDQKD"].where(df15["DJFDQKD"] > 1)
    gr = gas_ref if gas_ref and np.isfinite(gas_ref) else 1.0
    out["OPP_rel"] = opp / gr
    out["BFP_rel"] = df15["BFP"] / gr
    return out


def _wm(x: np.ndarray, ends: np.ndarray) -> np.ndarray:
    idx = ends[:, None] + np.arange(-WIN + 1, 1)[None, :]
    return x[idx]


def _slope(z: np.ndarray, n: int) -> np.ndarray:
    seg = z[:, -n:]
    k = np.arange(n) - (n - 1) / 2.0
    sxx = (k * k).sum()
    return np.nansum(seg * k, axis=1) / sxx * 4.0


def _block_stats(Xw: np.ndarray, blocks: int = 8) -> dict:
    n = Xw.shape[0]
    L = Xw.shape[1] // blocks
    B = Xw.reshape(n, blocks, L)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mean = np.nanmean(B, axis=2)
        sd = np.nanstd(B, axis=2)
        h = L // 2
        m_last = np.nanmean(B[:, :, -h:], axis=2)
        m_first = np.nanmean(B[:, :, :h], axis=2)
    scale = np.where(sd > 1e-9, sd, np.where(np.abs(mean) > 1e-9, 0.01 * np.abs(mean), 1.0))
    cvd = sd / np.where(np.abs(mean) > 1e-9, np.abs(mean), 1.0)
    drift = (m_last - m_first) / scale
    lastrel = (m_last - mean) / scale
    d = np.diff(B, axis=2)
    pos = np.nansum(d > 0, axis=2).astype(float)
    neg = np.nansum(d < 0, axis=2).astype(float)
    sb = (pos - neg) / np.maximum(pos + neg, 1.0)
    out = {}
    for k, v in [("cv", cvd), ("drift", drift), ("lastrel", lastrel), ("sb", sb)]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out[k] = np.nanmean(v, axis=1)
    return out


def _baseline(Xw: np.ndarray, base: int = BASE, min_valid: float = 0.3):
    """Reference for each window: its first 24 h, or the first usable 24 h block.

    Check consecutive 24 h blocks inside the current window and use the
    first block with at least 30 % finite samples. If no block qualifies,
    use the current window as the reference.
    """
    nb = Xw.shape[1] // base
    B = Xw[:, :nb * base].reshape(Xw.shape[0], nb, base)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        med_b = np.nanmedian(B, axis=2)
        sd_b = np.nanstd(B, axis=2)
        ok_b = np.isfinite(B).mean(axis=2)
    good = ok_b >= min_valid
    any_good = good.any(axis=1)
    k = np.argmax(good, axis=1)
    rows = np.arange(len(Xw))
    med = med_b[rows, k].copy()
    sd = sd_b[rows, k].copy()
    if (~any_good).any():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            med[~any_good] = np.nanmedian(Xw[~any_good], axis=1)
            sd[~any_good] = np.nanstd(Xw[~any_good], axis=1)
    return med, sd


def features_at(series15: pd.DataFrame, ends: np.ndarray) -> pd.DataFrame:
    """Feature matrix for the windows ending at the given 15-min indices."""
    ends = np.asarray(ends, dtype=int)
    n = len(ends)
    cols: dict[str, np.ndarray] = {}
    prof = {}
    for c in PARAMS:
        x = series15[c].to_numpy(float)
        if not np.isfinite(x).any():
            continue
        Xw = _wm(x, ends)
        med, sd = _baseline(Xw)
        span = np.nanpercentile(np.abs(Xw), 90, axis=1)
        sd = np.where(np.isfinite(sd) & (sd > np.maximum(0.01 * span, 1e-9)), sd,
                      np.maximum(0.01 * span, 1e-9))
        z = (Xw - med[:, None]) / sd[:, None]
        z = np.where(np.abs(z) < 1e6, z, np.nan)
        ok = np.isfinite(z[:, -BASE:]).mean(axis=1) > 0.2
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for name, k in LEVELS:
                v = np.nanmean(z[:, -k:], axis=1)
                cols[f"{c}__{name}"] = np.where(ok, v, np.nan)
            for name, k in EXTREMES:
                v = np.nanmax(np.abs(z[:, -k:]), axis=1)
                cols[f"{c}__{name}"] = np.where(ok, v, np.nan)
            cols[f"{c}__pers2"] = np.where(ok, np.nanmean(np.abs(z[:, -96:]) > 2, axis=1), np.nan)
            cols[f"{c}__dwin"] = np.where(ok, np.nanmean(z[:, -96:], axis=1)
                                          - np.nanmean(z[:, :BASE], axis=1), np.nan)
        for name, k in SLOPES:
            cols[f"{c}__{name}"] = np.where(ok, _slope(z, k), np.nan)
        # absolute statistics: the coefficient of
        # variation of a load-normalised signal is comparable across units and
        # captures "how unstable is this window" irrespective of its baseline
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cols[f"{c}__absmed"] = np.where(ok, np.nanmedian(Xw, axis=1), np.nan)
            cols[f"{c}__abssd"] = np.where(ok, np.nanstd(Xw, axis=1), np.nan)
            m24 = np.nanmean(Xw[:, -24:], axis=1)
            s24 = np.nanstd(Xw[:, -24:], axis=1)
            cols[f"{c}__cv6"] = np.where(ok, s24 / np.maximum(np.abs(m24), 1e-9), np.nan)
            m96 = np.nanmean(Xw, axis=1)
            s96 = np.nanstd(Xw, axis=1)
            cols[f"{c}__cv96"] = np.where(ok, s96 / np.maximum(np.abs(m96), 1e-9), np.nan)

        # ---- Within-window ratios: last 24 h versus first 72 h.
        # Numerator and denominator come from the SAME window and the SAME sensor
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            n24, n72 = 96, BASE * 3
            a24, a72 = np.abs(z[:, -n24:]), np.abs(z[:, :n72])
            cv_r = (np.nanstd(a24, axis=1) + 1e-6) / (np.nanstd(a72, axis=1) + 1e-6)
            am_r = (np.nanmax(a24, axis=1) + 1e-6) / (np.nanmax(a72, axis=1) + 1e-6)
            ab_r = (np.nanmean(a24, axis=1) + 1e-6) / (np.nanmean(a72, axis=1) + 1e-6)
            d24 = np.diff(z[:, -n24:], axis=1)
            d72 = np.diff(z[:, :n72], axis=1)
            s24 = np.sign(np.nan_to_num(d24)); s72 = np.sign(np.nan_to_num(d72))
            zc24 = np.nanmean(s24[:, 1:] * s24[:, :-1] < 0, axis=1)
            zc72 = np.nanmean(s72[:, 1:] * s72[:, :-1] < 0, axis=1)
            zc_r = (zc24 + 1e-3) / (zc72 + 1e-3)
        for kk, v in [("rh_cv", cv_r), ("rh_am", am_r), ("rh_abs", ab_r), ("rh_zc", zc_r)]:
            cols[f"{c}__{kk}"] = np.where(ok, np.clip(v, 0.0, 50.0), np.nan)
        # ---- Within-window ranks and CUSUM.
        # Everything below aggregates ONLY the current 96 h window
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cs = np.nancumsum(np.nan_to_num(z, nan=0.0), axis=1)
            ncol = z.shape[1]

            def _roll(w, step):
                pos = np.arange(w, ncol + 1, step)
                hi = cs[:, pos - 1]
                lo = np.where(pos - w - 1 >= 0,
                              cs[:, np.maximum(pos - w - 1, 0)], 0.0)
                return (hi - lo) / w

            r24 = _roll(96, 4)
            r6 = _roll(24, 4)
            rank24 = (r24 < r24[:, -1:]).mean(axis=1) if r24.size else np.full(len(z), np.nan)
            rank6 = (r6 < r6[:, -1:]).mean(axis=1) if r6.size else np.full(len(z), np.nan)
            kk0 = 0.5
            sp = np.zeros(len(z)); sn = np.zeros(len(z))
            mxp = np.zeros(len(z)); mxn = np.zeros(len(z))
            for t in range(ncol):
                zt = np.nan_to_num(z[:, t], nan=0.0)
                sp = np.maximum(0.0, sp + zt - kk0)
                sn = np.maximum(0.0, sn - zt - kk0)
                mxp = np.maximum(mxp, sp); mxn = np.maximum(mxn, sn)
            cusp = mxp / np.sqrt(ncol); cusn = mxn / np.sqrt(ncol)
            l48 = np.nanmean(z[:, -192:], axis=1)
            l72 = np.nanmean(z[:, -288:], axis=1)
            s48 = _slope(z, 192)
            s72 = _slope(z, 288)
        for kk, v in [("rank24", rank24), ("rank6", rank6), ("cusp", cusp),
                      ("cusn", cusn), ("l48", l48), ("l72", l72),
                      ("s48", s48), ("s72", s72)]:
            cols[f"{c}__{kk}"] = np.where(ok, v, np.nan)

        # ---- Shape statistics of the current deviation series.
        z24 = z[:, -96:]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dd = np.diff(z24, axis=1)
            sg = np.sign(np.nan_to_num(dd))
            zc = np.nanmean(sg[:, 1:] * sg[:, :-1] < 0, axis=1)
            mu = np.nanmean(z24, axis=1)
            sd24 = np.nanstd(z24, axis=1) + 1e-9
            skew = np.nanmean(((z24 - mu[:, None]) / sd24[:, None]) ** 3, axis=1)
            kurt = np.nanmean(((z24 - mu[:, None]) / sd24[:, None]) ** 4, axis=1)
            sprd = np.nanpercentile(z24, 90, axis=1) - np.nanpercentile(z24, 10, axis=1)
            a = z24[:, 1:] - mu[:, None]
            b = z24[:, :-1] - mu[:, None]
            ac = np.nanmean(a * b, axis=1) / (sd24 ** 2)
            cs = np.nancumsum(z24, axis=1)
            runmax = np.maximum.accumulate(np.where(np.isfinite(cs), cs, -np.inf), axis=1)
            mdd = np.nanmax(np.where(np.isfinite(cs), runmax - cs, np.nan), axis=1)
        for kk, v in [("zc", zc), ("skew", skew), ("kurt", kurt), ("sprd", sprd),
                      ("ac", ac), ("mdd", mdd)]:
            cols[f"{c}__{kk}"] = np.where(ok, v, np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            prof[c] = np.nanmean(z[:, -96:], axis=1)
        # ---- C6 shape family on the same window
        if c in C6_SENSORS:
            st = _block_stats(Xw)
            for kk, v in st.items():
                cols[f"c6_{c}_{kk}"] = np.where(ok, v, np.nan)

    F = pd.DataFrame(cols, index=range(n))
    # ---- Within-window cross-scale contrasts.
    # "is the last hour worse than the last 6 h / the last 24 h?" isolates the
    # acceleration of a deviation from its level, and is scale-free.
    for c in PARAMS:
        l1, l6, l24 = cols.get(f"{c}__l1"), cols.get(f"{c}__l6"), cols.get(f"{c}__l24")
        s6, s24 = cols.get(f"{c}__s6"), cols.get(f"{c}__s24")
        if l1 is not None and l6 is not None:
            F[f"{c}__d16"] = l1 - l6
            F[f"{c}__d624"] = l6 - l24
            F[f"{c}__d124"] = l1 - l24
        if s6 is not None and s24 is not None:
            F[f"{c}__acc"] = s6 - s24

    # cross relations
    def ratio(a, b, op="div"):
        if a not in series15.columns or b not in series15.columns:
            return None
        xa = series15[a].to_numpy(float)
        xb = series15[b].to_numpy(float)
        with np.errstate(invalid="ignore", divide="ignore"):
            r = xa / np.where(np.abs(xb) > 1e-6, np.abs(xb), np.nan) if op == "div" else xa - xb
        Xw = _wm(r, ends)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return np.nanmean(Xw, axis=1)

    for name, a, b, op in [("c6_eff_fw_pot", "FWF", "POT", "div"),
                           ("c6_set_over_wall", "SET", "FSDCKGW", "div"),
                           ("c6_approach", "SET", "DLDCKWD", "sub")]:
        v = ratio(a, b, op)
        if v is not None:
            F[name] = v
    if "DLDCKWD" in series15.columns and "DLDRKWD" in series15.columns:
        d = series15["DLDCKWD"].to_numpy(float) - series15["DLDRKWD"].to_numpy(float)
        st = _block_stats(_wm(d, ends))
        F["c6_dTconv_drift"] = st["drift"]
        F["c6_dTconv_cv"] = st["cv"]

    # ---- coordination over the 37-parameter last-24 h profile
    P = pd.DataFrame(prof, index=range(n))
    Z = P.to_numpy(float)
    F["co_up2"] = np.nansum(Z > 2, axis=1)
    F["co_dn2"] = np.nansum(Z < -2, axis=1)
    F["co_any2"] = np.nansum(np.abs(Z) > 2, axis=1)
    F["co_abs3"] = np.nansum(np.abs(Z) > 3, axis=1)
    F["co_maxabs"] = np.nanmax(np.abs(Z), axis=1)
    F["co_std"] = np.nanstd(Z, axis=1)

    # ---- per-window context 
    def win_stat(name, fn, k=None):
        if name not in series15.columns:
            return None
        x = series15[name].to_numpy(float)
        Xw = _wm(x, ends)
        if k is not None:
            Xw = Xw[:, -k:]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return fn(Xw)

    gas = series15["RQDQLL"].to_numpy(float)
    Xg = _wm(gas, ends)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        F["ctx_on_share"] = np.nanmean(Xg > 0, axis=1)
        gm = np.nanmedian(Xg[:, -96:], axis=1)
        gs = np.nanstd(Xg[:, -96:], axis=1)
        F["ctx_gas_cv"] = gs / np.maximum(np.abs(gm), 1e-6)
        F["ctx_gas_lvl"] = gm
    v = win_stat("IAT", lambda A: np.nanmedian(A, axis=1), 96)
    if v is not None:
        F["ctx_iat"] = v
    valid_all = []
    for c in S:
        if c in series15.columns:
            valid_all.append(np.isfinite(_wm(series15[c].to_numpy(float), ends)))
    if valid_all:
        V = np.stack(valid_all, axis=2)          # (n, 384, n_sensors)
        F["ctx_avail"] = V.mean(axis=(1, 2))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            F["ctx_n_sensor"] = (V.mean(axis=1) > 0.3).sum(axis=1).astype(float)
            # ---- Missingness statistics of the current window.
            F["ctx_miss24"] = 1.0 - V[:, -96:, :].mean(axis=(1, 2))
            F["ctx_miss_sensors"] = (V[:, -96:, :].mean(axis=1) < 0.5).sum(axis=1).astype(float)
            F["ctx_first24_avail"] = V[:, :BASE, :].mean(axis=(1, 2))
    F["ctx_superheat"] = float("TZKZQGRD" in series15.columns
                               and series15["TZKZQGRD"].notna().any())
    # Instrument-coverage descriptors; the saved feature list determines
    # which extracted columns enter the numerical scorer.
    if valid_all:
        mask = (V.mean(axis=1) > 0.3).astype(int)
        F["cat_cov"] = [",".join(map(str, row)) for row in mask]
    else:
        F["cat_cov"] = "na"
    F["cat_family"] = "过热" if F["ctx_superheat"].iloc[0] > 0 else "饱和"
    return F


def series_for(dfm: pd.DataFrame, gas_ref: float) -> pd.DataFrame:
    s15 = to15(dfm)
    inv = invariants(s15, gas_ref)
    return pd.concat([s15, inv], axis=1)[PARAMS]


def context(dfm: pd.DataFrame) -> dict:
    s15 = to15(dfm)
    gas = s15["RQDQLL"]
    av = s15.notna().mean()
    return {
        "ctx_gas_med": float(gas[gas > 0].median()) if (gas > 0).any() else np.nan,
        "ctx_gas_cv": float(gas[gas > 0].std() / gas[gas > 0].mean()) if (gas > 0).any() else np.nan,
        "ctx_on_share": float((gas > 0).mean()),
        "ctx_iat": float(s15["IAT"].median()) if s15["IAT"].notna().any() else np.nan,
        "ctx_n_sensor": float((av > 0.3).sum()),
        "ctx_superheat": float(s15["TZKZQGRD"].notna().mean() > 0.3),
        "ctx_avail": float(av.mean()),
    }



# ============================================================== 5-min block
HF_SENSORS = ["BFP", "FLL", "DJFDQKD", "OPP", "POT", "FSDCKGW", "SET", "FWF"]
HF_N = 72          # 6 h at 5 min


def hf5_at(dfm5: pd.DataFrame, ends5: np.ndarray) -> pd.DataFrame:
    """High-frequency features computed on the raw 5-min series (last 6 h).

    The 15-min grid averages three samples and may suppress short-period
    variation. Here we take the cleaned 5-min series directly, difference it
    (which removes the trend), and describe the spectrum and the jumpiness.
    All statistics are computed from the current input window.
    """
    ends5 = np.asarray(ends5, dtype=int)
    n = len(ends5)
    out: dict[str, np.ndarray] = {}
    off = np.arange(-HF_N + 1, 1)
    for c in HF_SENSORS:
        if c not in dfm5.columns:
            continue
        x = dfm5[c].to_numpy(float)
        idx = ends5[:, None] + off[None, :]
        idx = np.clip(idx, 0, len(x) - 1)
        Xw = x[idx]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dx = np.diff(Xw, axis=1)
            ok = np.isfinite(dx).mean(axis=1) > 0.7
            dx = np.where(np.isfinite(dx), dx, 0.0)
            P = np.abs(np.fft.rfft(dx, axis=1)) ** 2
            P[:, 0] = 0.0
            tot = P.sum(axis=1) + 1e-12
            lo = P[:, 1:6].sum(axis=1) / tot
            mid = P[:, 6:18].sum(axis=1) / tot
            hi = P[:, 18:].sum(axis=1) / tot
            q = P / tot[:, None]
            ent = -np.nansum(q * np.log(q + 1e-12), axis=1) / np.log(P.shape[1])
            dom = np.argmax(P, axis=1) / P.shape[1]
            sd = np.nanstd(dx, axis=1) + 1e-12
            mj = np.nanmax(np.abs(dx), axis=1) / sd
            s = np.sign(dx)
            zc = np.nanmean(s[:, 1:] * s[:, :-1] < 0, axis=1)
            a = dx[:, 1:] - dx[:, :-1].mean(axis=1, keepdims=True)
            ac = np.nanmean((dx[:, 1:] - np.nanmean(dx, axis=1, keepdims=True))
                            * (dx[:, :-1] - np.nanmean(dx, axis=1, keepdims=True)), axis=1) / sd ** 2
        for k, v in [("lo", lo), ("mid", mid), ("hi", hi), ("ent", ent), ("dom", dom),
                     ("maxjump", mj), ("zc", zc), ("ac", ac)]:
            out[f"hf_{c}_{k}"] = np.where(ok, v, np.nan)
    return pd.DataFrame(out, index=range(n))





def window_features(dfm: pd.DataFrame, gas_ref: float) -> pd.DataFrame:
    """一段 96 h 序列 -> 一行特征（1498 列）。"""
    ser = series_for(dfm, gas_ref)
    if len(ser) < WIN:
        raise ValueError("序列长度不足一个完整窗口")
    e15 = np.array([len(ser) - 1])
    X = features_at(ser, e15)
    e5 = np.clip(3 * e15 + 2, 0, len(dfm) - 1)
    return pd.concat([X.reset_index(drop=True),
                      hf5_at(dfm, e5).reset_index(drop=True)], axis=1)
