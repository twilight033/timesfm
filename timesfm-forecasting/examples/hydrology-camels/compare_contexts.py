#!/usr/bin/env python3
"""
对比 CONTEXT_LEN ∈ {96, 180, 365} 下 TimesFM 与 Persistence 的表现。
共享数据流水线，模型只加载一次（每个 context 重新 compile）。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import timesfm

from run_forecast_liaohe import (
    BATCH_SIZE,
    DATA_FILE,
    HORIZON,
    build_station_segments,
    build_windows_from_segments,
    compute_nse_metrics,
    load_all_stations,
)
from run_persistence_liaohe import persistence_predict

OUT = Path(__file__).parent / "output_context_compare"
OUT.mkdir(exist_ok=True)

CONTEXTS = [96, 180, 365]


def main():
    print("加载数据 + 切段（仅一次）...")
    stations = load_all_stations(DATA_FILE)
    station_segments = build_station_segments(stations)
    print(f"  {len(station_segments)} 个站点")

    print("加载 TimesFM 2.5 (200M) 模型...")
    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")

    all_summary = []
    for ctx in CONTEXTS:
        print(f"\n{'#'*60}")
        print(f"# CONTEXT_LEN = {ctx}")
        print(f"{'#'*60}")

        model.compile(
            timesfm.ForecastConfig(
                max_context=ctx,
                max_horizon=HORIZON,
                per_core_batch_size=BATCH_SIZE,
                normalize_inputs=True,
                infer_is_positive=True,
            )
        )

        rows = []
        for station, segments in station_segments.items():
            max_len = max((len(s.values) for s in segments), default=0)
            if max_len < ctx + HORIZON:
                continue
            bundle = build_windows_from_segments(segments, ctx, HORIZON)
            if bundle is None:
                continue
            n_win = len(bundle["contexts"])

            # 先做 persistence（避免 TimesFM 就地填充 contexts list 影响长度）
            per_preds = persistence_predict(bundle["contexts"], HORIZON)
            per_preds = per_preds[:n_win]
            per_m = compute_nse_metrics(
                bundle["targets"], per_preds,
                bundle["target_observed"], bundle["target_in_flood"],
            )

            tfm_preds, _ = model.forecast(horizon=HORIZON, inputs=bundle["contexts"])
            tfm_preds = tfm_preds[:n_win]
            tfm_m = compute_nse_metrics(
                bundle["targets"], tfm_preds,
                bundle["target_observed"], bundle["target_in_flood"],
            )

            rows.append({
                "station": station,
                "n_windows": n_win,
                "tfm_flood": tfm_m["nse_flood"],
                "tfm_dry": tfm_m["nse_dry"],
                "tfm_overall": tfm_m["nse_overall"],
                "per_flood": per_m["nse_flood"],
                "per_dry": per_m["nse_dry"],
                "per_overall": per_m["nse_overall"],
            })

        df = pd.DataFrame(rows)
        df.to_csv(OUT / f"per_station_ctx{ctx}.csv", index=False, encoding="utf-8-sig")

        summary = {
            "context_len": ctx,
            "n_stations": len(df),
            "total_windows": int(df["n_windows"].sum()),
            "tfm_flood_med": round(df["tfm_flood"].median(), 4),
            "tfm_dry_med": round(df["tfm_dry"].median(), 4),
            "tfm_overall_med": round(df["tfm_overall"].median(), 4),
            "per_flood_med": round(df["per_flood"].median(), 4),
            "per_dry_med": round(df["per_dry"].median(), 4),
            "per_overall_med": round(df["per_overall"].median(), 4),
            "delta_flood": round(df["tfm_flood"].median() - df["per_flood"].median(), 4),
            "delta_overall": round(df["tfm_overall"].median() - df["per_overall"].median(), 4),
        }
        all_summary.append(summary)
        print(f"\n  完成 {len(df)} 站、{summary['total_windows']} 窗口")
        print(f"  TFM flood={summary['tfm_flood_med']:.4f}  overall={summary['tfm_overall_med']:.4f}")
        print(f"  PER flood={summary['per_flood_med']:.4f}  overall={summary['per_overall_med']:.4f}")
        print(f"  Δflood={summary['delta_flood']:+.4f}  Δoverall={summary['delta_overall']:+.4f}")

    summary_df = pd.DataFrame(all_summary)
    summary_df.to_csv(OUT / "summary.csv", index=False, encoding="utf-8-sig")
    print("\n" + "=" * 60)
    print("  总览")
    print("=" * 60)
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
