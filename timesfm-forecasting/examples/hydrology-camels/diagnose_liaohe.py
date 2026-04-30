#!/usr/bin/env python3
"""
数据体检脚本 —— 辽河松花江流域水文数据

输出：
  data_diagnosis/overall.txt          总体摘要（行/列/站/时间跨度/列缺失）
  data_diagnosis/per_station.csv      每站统计（采样密度、间隔、流量分布）
  data_diagnosis/gap_distribution.csv 全局日期间隔分位数（诊断日级 vs 事件采样）
  data_diagnosis/segments.csv         按"> 2×median_gap"切段后的连续段统计
  data_diagnosis/feasibility.csv      不同 CONTEXT_LEN 下可跑通的站点数
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

DATA_FILE = Path(__file__).parent / "data" / "辽河松花江流域水文数据.xlsx"
OUT_DIR = Path(__file__).parent / "data_diagnosis"

STATION_COL = "站名"
DATE_COL = "日期"
TARGET_COL = "流量"

# 评估不同 context 长度下的可行站点数
CONTEXT_GRID = [48, 96, 180, 365]
HORIZON = 1


# ---------------------------------------------------------------------------
# 加载（不清洗，保留原始状态用于诊断）
# ---------------------------------------------------------------------------

def load_raw(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path)
    # 保留原始列名，额外解析日期/流量为数值类型；dayfirst=True 适配 DD/MM/YYYY
    df["_date_parsed"] = pd.to_datetime(df[DATE_COL], errors="coerce", dayfirst=True)
    df["_flow_num"] = pd.to_numeric(df[TARGET_COL], errors="coerce")
    return df


# ---------------------------------------------------------------------------
# 统计模块
# ---------------------------------------------------------------------------

def overall_summary(df: pd.DataFrame) -> list[str]:
    lines = []
    lines.append(f"文件: {DATA_FILE}")
    lines.append(f"总行数: {len(df)}")
    lines.append(f"原始列: {df.columns.tolist()[:-2]}")  # 去掉内部列
    lines.append("")

    lines.append("== 各列缺失（原始） ==")
    for col in df.columns:
        if col.startswith("_"):
            continue
        n_missing = df[col].isna().sum()
        pct = n_missing / len(df) * 100
        lines.append(f"  {col:10s}  missing={n_missing:>7d}  ({pct:5.2f}%)")
    lines.append("")

    lines.append("== 日期解析 ==")
    n_date_bad = df["_date_parsed"].isna().sum()
    lines.append(f"  解析失败行: {n_date_bad}")
    if df["_date_parsed"].notna().any():
        lines.append(
            f"  日期范围: {df['_date_parsed'].min()} ~ {df['_date_parsed'].max()}"
        )
    lines.append("")

    lines.append("== 流量数值（_flow_num） ==")
    flow = df["_flow_num"]
    lines.append(f"  非空: {flow.notna().sum()} / {len(flow)}")
    lines.append(f"  负值: {(flow < 0).sum()}")
    lines.append(f"  零值: {(flow == 0).sum()}")
    if flow.notna().any():
        q = flow.dropna().quantile([0, 0.01, 0.5, 0.99, 1.0])
        lines.append(f"  分位(0/1%/50%/99%/100%): "
                     f"{q.iloc[0]:.3g} / {q.iloc[1]:.3g} / {q.iloc[2]:.3g} / {q.iloc[3]:.3g} / {q.iloc[4]:.3g}")
    lines.append("")

    lines.append("== 站点 ==")
    lines.append(f"  唯一站名数: {df[STATION_COL].nunique()}")
    station_na = df[STATION_COL].isna().sum()
    lines.append(f"  站名缺失行: {station_na}")
    lines.append("")

    lines.append("== 重复记录 ==")
    dup_key = df.dropna(subset=[STATION_COL, "_date_parsed"])
    n_dup = dup_key.duplicated(subset=[STATION_COL, "_date_parsed"]).sum()
    lines.append(f"  (站名+日期) 完全重复行: {n_dup}")

    return lines


def per_station_stats(df: pd.DataFrame) -> pd.DataFrame:
    """每站点的采样、时间跨度、流量分布、间隔统计。"""
    d = df.dropna(subset=[STATION_COL, "_date_parsed"]).copy()
    d = d.sort_values([STATION_COL, "_date_parsed"])

    rows = []
    for name, g in d.groupby(STATION_COL, sort=False):
        flow = g["_flow_num"]
        dates = g["_date_parsed"]

        n_rows = len(g)
        n_valid_flow = flow.notna().sum()
        valid_ratio = n_valid_flow / n_rows if n_rows else float("nan")

        date_min = dates.min()
        date_max = dates.max()
        span_days = (date_max - date_min).days + 1 if pd.notna(date_min) else 0

        # 相邻间隔（只在 flow 非空记录之间算，反映"有效观测"的间隔）
        valid = g.dropna(subset=["_flow_num"])
        if len(valid) >= 2:
            gaps = valid["_date_parsed"].diff().dt.total_seconds() / 86400
            gaps = gaps.dropna()
            median_gap = float(gaps.median())
            p90_gap = float(gaps.quantile(0.9))
            max_gap = float(gaps.max())
        else:
            median_gap = p90_gap = max_gap = float("nan")

        # 流量分布
        if n_valid_flow:
            flow_min = float(flow.min())
            flow_max = float(flow.max())
            flow_median = float(flow.median())
            flow_std = float(flow.std())
            n_negative = int((flow < 0).sum())
            n_zero = int((flow == 0).sum())
        else:
            flow_min = flow_max = flow_median = flow_std = float("nan")
            n_negative = n_zero = 0

        rows.append({
            "station": name,
            "n_rows": n_rows,
            "n_valid_flow": n_valid_flow,
            "valid_ratio": round(valid_ratio, 4),
            "date_min": date_min.date() if pd.notna(date_min) else None,
            "date_max": date_max.date() if pd.notna(date_max) else None,
            "span_days": span_days,
            "density_per_day": round(n_valid_flow / span_days, 4) if span_days else float("nan"),
            "median_gap_days": round(median_gap, 3),
            "p90_gap_days": round(p90_gap, 3),
            "max_gap_days": round(max_gap, 3),
            "flow_min": flow_min,
            "flow_max": flow_max,
            "flow_median": flow_median,
            "flow_std": round(flow_std, 4) if not np.isnan(flow_std) else float("nan"),
            "n_negative": n_negative,
            "n_zero": n_zero,
        })

    return pd.DataFrame(rows)


def global_gap_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """所有站点有效观测间的日期间隔分位数。"""
    d = df.dropna(subset=[STATION_COL, "_date_parsed", "_flow_num"]).sort_values(
        [STATION_COL, "_date_parsed"]
    )
    gaps = d.groupby(STATION_COL)["_date_parsed"].diff().dt.total_seconds() / 86400
    gaps = gaps.dropna()

    qs = [0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0]
    rows = [{"quantile": f"{q*100:.0f}%", "gap_days": round(float(gaps.quantile(q)), 3)} for q in qs]
    rows.append({"quantile": "mean", "gap_days": round(float(gaps.mean()), 3)})
    rows.append({"quantile": "count", "gap_days": int(len(gaps))})
    return pd.DataFrame(rows)


def segments_stats(
    df: pd.DataFrame, gap_threshold_factor: float = 2.0
) -> pd.DataFrame:
    """按"间隔 > gap_threshold_factor × 该站 median_gap"切段，统计连续段。
    通用性：对日级站点阈值≈2天，对事件采样站点阈值会自动放大。
    """
    d = df.dropna(subset=[STATION_COL, "_date_parsed", "_flow_num"]).sort_values(
        [STATION_COL, "_date_parsed"]
    )

    rows = []
    for name, g in d.groupby(STATION_COL, sort=False):
        n = len(g)
        if n < 2:
            rows.append({
                "station": name, "n_points": n, "n_segments": 1 if n else 0,
                "max_segment_len": n, "median_segment_len": n,
                "threshold_days": float("nan"),
            })
            continue

        gaps = g["_date_parsed"].diff().dt.total_seconds() / 86400
        median_gap = gaps.dropna().median()
        threshold = max(gap_threshold_factor * median_gap, 1.5)  # 至少 1.5 天

        break_points = (gaps > threshold).fillna(False).values
        seg_id = np.cumsum(break_points)
        seg_sizes = pd.Series(seg_id).value_counts().sort_index()

        rows.append({
            "station": name,
            "n_points": n,
            "n_segments": int(seg_sizes.size),
            "max_segment_len": int(seg_sizes.max()),
            "median_segment_len": int(seg_sizes.median()),
            "threshold_days": round(float(threshold), 2),
        })

    return pd.DataFrame(rows)


def feasibility_with_pipeline(stations: dict) -> pd.DataFrame:
    """按 run_forecast_liaohe 的真实流水线（汛期 ffill / 非汛期插值）切段后，
    报告不同 CONTEXT_LEN 下可用站数与窗口数。"""
    from run_forecast_liaohe import build_segments, FLOOD_MONTHS, MAX_FLOOD_FFILL, MAX_DRY_INTERP

    seg_lens_per_station = []
    for name, s in stations.items():
        segs = build_segments(s, FLOOD_MONTHS, MAX_FLOOD_FFILL, MAX_DRY_INTERP)
        if not segs:
            seg_lens_per_station.append((name, []))
            continue
        seg_lens_per_station.append((name, [len(seg.values) for seg in segs]))

    rows = []
    for ctx in CONTEXT_GRID:
        need = ctx + HORIZON
        usable = 0
        total_win = 0
        for _, lens in seg_lens_per_station:
            if not lens:
                continue
            station_win = sum(max(L - need + 1, 0) for L in lens)
            if station_win > 0:
                usable += 1
                total_win += station_win
        rows.append({
            "context_len": ctx,
            "horizon": HORIZON,
            "stations_usable": usable,
            "total_windows": total_win,
        })
    return pd.DataFrame(rows)


def feasibility_under_context(seg_df: pd.DataFrame) -> pd.DataFrame:
    """不同 CONTEXT_LEN 下：
       (a) 按原脚本——整站长度 >= CONTEXT+HORIZON 的站数
       (b) 按连续段——最长段 >= CONTEXT+HORIZON 的站数
       (c) 可构建的总窗口数（按连续段累加）
    """
    rows = []
    for ctx in CONTEXT_GRID:
        need = ctx + HORIZON
        usable_whole = int((seg_df["n_points"] >= need).sum())
        usable_seg = int((seg_df["max_segment_len"] >= need).sum())
        # 窗口总数粗估：对每站仅按 max_segment_len 算
        total_wins = int(np.clip(seg_df["max_segment_len"] - need + 1, a_min=0, a_max=None).sum())
        rows.append({
            "context_len": ctx,
            "horizon": HORIZON,
            "stations_usable_whole_series": usable_whole,
            "stations_usable_longest_segment": usable_seg,
            "total_windows_from_longest_segment": total_wins,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"加载 {DATA_FILE} ...")
    df = load_raw(DATA_FILE)
    print(f"  行数: {len(df)}  列数: {df.shape[1] - 2}（不含内部解析列）")

    # 1) 总体摘要
    overall_lines = overall_summary(df)
    overall_txt = "\n".join(overall_lines)
    (OUT_DIR / "overall.txt").write_text(overall_txt, encoding="utf-8")
    print("\n" + overall_txt)

    # 2) 每站统计
    station_df = per_station_stats(df)
    station_df.to_csv(OUT_DIR / "per_station.csv", index=False, encoding="utf-8-sig")
    print(f"\n== per_station.csv 关键分位数（{len(station_df)} 站） ==")
    cols = ["n_rows", "n_valid_flow", "valid_ratio", "span_days",
            "density_per_day", "median_gap_days", "max_gap_days",
            "flow_median", "flow_std"]
    print(station_df[cols].describe(percentiles=[0.1, 0.5, 0.9]).round(3).to_string())

    # 3) 全局间隔分布
    gap_df = global_gap_distribution(df)
    gap_df.to_csv(OUT_DIR / "gap_distribution.csv", index=False, encoding="utf-8-sig")
    print("\n== 全局相邻有效观测日期间隔（单位：天） ==")
    print(gap_df.to_string(index=False))

    # 4) 连续段统计
    seg_df = segments_stats(df)
    seg_df.to_csv(OUT_DIR / "segments.csv", index=False, encoding="utf-8-sig")
    print(f"\n== 连续段统计（阈值: 2×median_gap，至少1.5天） ==")
    print(seg_df[["n_points", "n_segments", "max_segment_len",
                 "median_segment_len", "threshold_days"]]
          .describe(percentiles=[0.1, 0.5, 0.9]).round(2).to_string())

    # 5) 不同 CONTEXT_LEN 的可行性（按原始连续段定义）
    feas_df = feasibility_under_context(seg_df)
    feas_df.to_csv(OUT_DIR / "feasibility.csv", index=False, encoding="utf-8-sig")
    print("\n== 不同 CONTEXT_LEN 的站点可用性（按原始连续段） ==")
    print(feas_df.to_string(index=False))

    # 6) 按真实流水线（汛期 ffill / 非汛期线性插值）切段后的可行性
    print("\n按 run_forecast_liaohe 真实流水线切段...")
    stations = {}
    for name, g in df.dropna(subset=[STATION_COL, "_date_parsed", "_flow_num"]).groupby(
        STATION_COL, sort=False
    ):
        s = g.set_index("_date_parsed")["_flow_num"].sort_index()
        s = s[s >= 0]
        if len(s):
            stations[str(name)] = s

    pipeline_df = feasibility_with_pipeline(stations)
    pipeline_df.to_csv(OUT_DIR / "feasibility_pipeline.csv", index=False, encoding="utf-8-sig")
    print("\n== 不同 CONTEXT_LEN 的站点可用性（汛期ffill+非汛期插值后） ==")
    print(pipeline_df.to_string(index=False))

    print(f"\n所有诊断文件已输出到: {OUT_DIR}")


if __name__ == "__main__":
    main()
