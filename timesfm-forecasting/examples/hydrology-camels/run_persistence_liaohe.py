#!/usr/bin/env python3
"""
Persistence baseline —— 辽河松花江流域

与 run_forecast_liaohe.py 共享数据流水线（dayfirst 解析、汛期 ffill / 非汛期插值
切段、Segment 结构、滑动窗口与三档 NSE）。区别只在预测函数：
    pred[t+1..t+H] = obs[t]
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from run_forecast_liaohe import (
    CONTEXT_LEN,
    DATA_FILE,
    HORIZON,
    Segment,
    build_station_segments,
    compute_nse_metrics,
    forecast_single_station,
    load_all_stations,
    safe_filename,
)


def persistence_predict(contexts: list[np.ndarray], horizon: int) -> np.ndarray:
    """每个窗口的预测 = 窗口最后一个值，复制 horizon 次。"""
    last_values = np.asarray([c[-1] for c in contexts], dtype=np.float32)
    return np.repeat(last_values[:, None], horizon, axis=1)


def main():
    output_dir = Path(__file__).parent / "output_liaohe_persistence"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"加载数据: {DATA_FILE}")
    stations = load_all_stations(DATA_FILE)
    print(f"共检测到 {len(stations)} 个站点")

    print("构建连续段（与 forecast 脚本完全一致的流水线）...")
    station_segments = build_station_segments(stations)
    print(f"CONTEXT_LEN={CONTEXT_LEN}, HORIZON={HORIZON}")

    results = []
    skipped = []
    for station, segments in station_segments.items():
        max_len = max((len(s.values) for s in segments), default=0)
        if max_len < CONTEXT_LEN + HORIZON:
            msg = f"最长段 {max_len} < {CONTEXT_LEN + HORIZON}"
            print(f"\n  [跳过] {station}: {msg}")
            skipped.append({"station": station, "reason": msg})
            continue
        try:
            r = forecast_single_station(
                model=None,
                station=station,
                segments=segments,
                output_dir=output_dir,
                forecast_fn=persistence_predict,
                file_suffix="persistence",
            )
            if r is not None:
                results.append(r)
            else:
                skipped.append({"station": station, "reason": "no usable windows"})
        except Exception as e:
            print(f"  [跳过] {station}: {e}")
            skipped.append({"station": station, "reason": str(e)})

    if results:
        summary_df = pd.DataFrame(results)
        summary_path = output_dir / "nse_summary_persistence.csv"
        summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

        print(f"\n{'='*55}")
        print("  Persistence Baseline 汇总")
        print(f"{'='*55}")
        print(summary_df.to_string(index=False))

        if len(results) > 1:
            print("\n  各指标中位数:")
            for col in ["nse_flood", "nse_dry", "nse_overall"]:
                vals = summary_df[col].dropna()
                if len(vals):
                    print(f"    {col}: {vals.median():.4f}  (有效站数 {len(vals)})")
        print(f"\n  汇总已保存: {summary_path}")

    if skipped:
        skip_df = pd.DataFrame(skipped)
        skip_path = output_dir / "skipped_stations.csv"
        skip_df.to_csv(skip_path, index=False, encoding="utf-8-sig")
        print(f"  已跳过 {len(skipped)} 个站点，明细: {skip_path}")


if __name__ == "__main__":
    main()
