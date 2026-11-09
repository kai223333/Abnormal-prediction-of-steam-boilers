"""predict.py -- 主预测脚本。

    评测数据目录 --data_dir
        -> 逐段读取（每段一个 96 h / 5 min 的多传感器序列）
        -> 清洗 + 1498 维特征
        -> CatBoost 排序模型打分 phi(x)
        -> 阈值 tau 判决
        -> result.csv（切片 ID, 故障判断）

用法：
    python predict.py --data_dir ../data --output ../result.csv
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "src"))

import utils as U          # noqa: E402
import feature as F        # noqa: E402
import model as M          # noqa: E402


def list_inputs(data_dir: str) -> list[str]:
    """定位输入文件：优先 <data_dir>/test/*.csv，否则 <data_dir>/*.csv。"""
    sub = os.path.join(data_dir, "test")
    d = sub if os.path.isdir(sub) else data_dir
    files = sorted(glob.glob(os.path.join(d, "*.csv")))
    if not files:
        raise FileNotFoundError(f"在 {d} 下没有找到 .csv 输入文件")
    return files


def main() -> None:
    ap = argparse.ArgumentParser(description="锅炉故障风险预测")
    ap.add_argument("--data_dir", required=True, help="评测数据目录")
    ap.add_argument("--output", required=True, help="输出结果 csv 路径")
    ap.add_argument("--model_dir", default=os.path.join(HERE, "models"),
                    help="模型目录（默认 ./models）")
    ap.add_argument("--quiet", action="store_true", help="不打印进度")
    args = ap.parse_args()

    det = M.load(args.model_dir)
    files = list_inputs(args.data_dir)
    if not args.quiet:
        print(f"输入 {len(files)} 段，特征 {len(det.features)} 维，"
              f"阈值 tau = {det.tau:.6f}")

    rows = []
    t0 = time.time()
    for i, fp in enumerate(files, 1):
        sid = os.path.splitext(os.path.basename(fp))[0]
        try:
            raw = U.load_window_csv(fp)
            dfm, gas_ref = U.prepare_window(raw)
            X = F.window_features(dfm, gas_ref)
            risk = float(det.score(X)[0])
            pred = int(risk > det.tau)
        except Exception as exc:                       # 单段失败不中断整批
            print(f"[warn] {sid}: {exc}", file=sys.stderr)
            risk, pred = 0.0, 0
        rows.append(dict(test_id=sid, prediction=pred, risk=risk))
        if not args.quiet and (i % 50 == 0 or i == len(files)):
            print(f"  {i}/{len(files)}  {time.time() - t0:.0f}s")

    out = pd.DataFrame(rows)
    # 提交文件仅保留两列；不写入 DataFrame 索引或风险分。
    submission = out[["test_id", "prediction"]].rename(
        columns={"test_id": "切片ID", "prediction": "故障判断"})
    submission.to_csv(args.output, index=False, encoding="utf-8-sig")
    detail = os.path.splitext(args.output)[0] + "_risk.csv"
    out.to_csv(detail, index=False, encoding="utf-8-sig")
    if not args.quiet:
        print(f"完成：{int(out.prediction.sum())}/{len(out)} 判为有故障风险")
        print(f"结果 -> {args.output}；风险分明细 -> {detail}")


if __name__ == "__main__":
    main()
