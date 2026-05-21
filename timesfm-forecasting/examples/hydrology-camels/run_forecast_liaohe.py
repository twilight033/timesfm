#!/usr/bin/env python3
"""
TimesFM 零样本水文预测 —— 辽河松花江流域

数据源：data/辽河松花江流域水文数据.xlsx

数据流水线（关键改动）：
  1. dayfirst=True 解析 DD/MM/YYYY 格式日期；
  2. 按站点重采样到日级；
  3. 缺测填充：
       汛期 (FLOOD_MONTHS) 内 → 前值填充 ffill，gap > MAX_FLOOD_FFILL 切段；
       非汛期内               → 线性插值，gap > MAX_DRY_INTERP 切段；
  4. 在每段内独立做滑动窗口；
  5. 评估三档 NSE：
       nse_flood   —— horizon 落在汛期且为实测
       nse_dry     —— horizon 落在非汛期且为实测
       nse_overall —— horizon 为实测的全部窗口
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import timesfm

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

DATA_FILE = Path(__file__).parent / "data" / "辽河松花江流域水文数据.xlsx"

STATION_COL = "站名"
DATE_COL = "日期"
TARGET_COL = "流量"

FLOOD_MONTHS = {6, 7, 8, 9}
MAX_FLOOD_FFILL = 7   # 汛期内允许 ffill 的最大跨度（天）
MAX_DRY_INTERP = 90   # 非汛期内允许线性插值的最大跨度（天）

CONTEXT_LEN = 384
HORIZON = 1
BATCH_SIZE = 32


# ---------------------------------------------------------------------------
# Segment：连续日级片段
# ---------------------------------------------------------------------------

@dataclass
class Segment:
    dates: pd.DatetimeIndex      # 日级、连续
    values: np.ndarray           # 日级流量，含填充
    is_observed: np.ndarray      # 该日是否实测（False = 填充）
    in_flood: np.ndarray         # 该日是否在汛期


def _finalize_segment(
    dates: list, values: list, observed: list, flood_months: Iterable[int]
) -> Segment:
    flood_set = set(flood_months)
    didx = pd.DatetimeIndex(dates)
    return Segment(
        dates=didx,
        values=np.asarray(values, dtype=np.float32),
        is_observed=np.asarray(observed, dtype=bool),
        in_flood=np.asarray([d.month in flood_set for d in didx], dtype=bool),
    )


def build_segments(
    series: pd.Series,
    flood_months: Iterable[int] = FLOOD_MONTHS,
    max_flood_ffill: int = MAX_FLOOD_FFILL,
    max_dry_interp: int = MAX_DRY_INTERP,
) -> list[Segment]:
    """对单站实测序列构建连续日级段。

    输入要求：series 的 index 是日期（可含时分秒），值是实测流量。
    会先按"自然日"聚合（同日多观取均值），再按 gap 性质决定填充或切段。
    """
    if len(series) == 0:
        return []

    # 日级聚合
    s = series.copy()
    s.index = pd.DatetimeIndex(s.index).normalize()
    s = s.groupby(s.index).mean().sort_index().astype(np.float32)

    obs_dates = s.index
    obs_values = s.values

    flood_set = set(flood_months)

    segments: list[Segment] = []
    cur_dates: list = [obs_dates[0]]
    cur_values: list = [float(obs_values[0])]
    cur_observed: list = [True]

    for i in range(1, len(obs_dates)):
        prev_d = obs_dates[i - 1]
        cur_d = obs_dates[i]
        prev_v = float(obs_values[i - 1])
        cur_v = float(obs_values[i])
        gap_days = (cur_d - prev_d).days  # 相邻实测之间相差天数

        if gap_days <= 0:
            # 重复日期（理论上 groupby 后不会出现）
            continue

        if gap_days == 1:
            cur_dates.append(cur_d)
            cur_values.append(cur_v)
            cur_observed.append(True)
            continue

        # 需要决策：检查 gap 内（prev_d+1 ~ cur_d-1）是否含汛期日
        gap_inner = pd.date_range(
            prev_d + pd.Timedelta(days=1), cur_d - pd.Timedelta(days=1), freq="D"
        )
        has_flood = any(d.month in flood_set for d in gap_inner)
        threshold = max_flood_ffill if has_flood else max_dry_interp

        if (gap_days - 1) > threshold:
            # 切段
            segments.append(_finalize_segment(cur_dates, cur_values, cur_observed, flood_set))
            cur_dates = [cur_d]
            cur_values = [cur_v]
            cur_observed = [True]
            continue

        # 段内填充
        for j, gd in enumerate(gap_inner):
            cur_dates.append(gd)
            if gd.month in flood_set:
                # 汛期：前值填充
                cur_values.append(prev_v)
            else:
                # 非汛期：线性插值
                frac = (j + 1) / gap_days
                cur_values.append(prev_v + frac * (cur_v - prev_v))
            cur_observed.append(False)
        cur_dates.append(cur_d)
        cur_values.append(cur_v)
        cur_observed.append(True)

    segments.append(_finalize_segment(cur_dates, cur_values, cur_observed, flood_set))
    return segments


# ---------------------------------------------------------------------------
# 数据加载
# ---------------------------------------------------------------------------

def load_all_stations(path: Path) -> dict[str, pd.Series]:
    """读 Excel，剔空剔负，按站名返回 {站名: 实测流量序列}。"""
    df = pd.read_excel(path)

    missing = [c for c in (STATION_COL, DATE_COL, TARGET_COL) if c not in df.columns]
    if missing:
        raise ValueError(f"Excel 缺少必需列 {missing}，实际列为 {df.columns.tolist()}")

    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce", dayfirst=True)
    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")

    df = df.dropna(subset=[STATION_COL, DATE_COL, TARGET_COL])
    df = df[df[TARGET_COL] >= 0]
    df = df.drop_duplicates(subset=[STATION_COL, DATE_COL])

    stations: dict[str, pd.Series] = {}
    for name, g in df.groupby(STATION_COL, sort=False):
        s = g.set_index(DATE_COL)[TARGET_COL].sort_index().astype(np.float32)
        if len(s) == 0:
            continue
        stations[str(name)] = s
    return stations


def build_station_segments(
    stations: dict[str, pd.Series]
) -> dict[str, list[Segment]]:
    return {name: build_segments(s) for name, s in stations.items()}


# ---------------------------------------------------------------------------
# 滑动窗口
# ---------------------------------------------------------------------------

def build_windows_from_segments(
    segments: list[Segment], context_len: int, horizon: int
) -> dict | None:
    """对一个站的所有段做滑动窗口，并合并成批次结构。
    返回 None 表示该站没有任何段够长。
    """
    contexts: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    tgt_dates: list[pd.DatetimeIndex] = []
    tgt_observed: list[np.ndarray] = []
    tgt_in_flood: list[np.ndarray] = []
    ctx_end_dates: list = []

    for seg in segments:
        n = len(seg.values)
        if n < context_len + horizon:
            continue
        n_win = n - context_len - horizon + 1
        for i in range(n_win):
            contexts.append(seg.values[i : i + context_len])
            t_slice = slice(i + context_len, i + context_len + horizon)
            targets.append(seg.values[t_slice])
            tgt_dates.append(seg.dates[t_slice])
            tgt_observed.append(seg.is_observed[t_slice])
            tgt_in_flood.append(seg.in_flood[t_slice])
            ctx_end_dates.append(seg.dates[i + context_len - 1])

    if not contexts:
        return None
    return {
        "contexts": contexts,
        "targets": np.stack(targets),
        "target_dates": tgt_dates,
        "target_observed": np.stack(tgt_observed),
        "target_in_flood": np.stack(tgt_in_flood),
        "ctx_end_dates": ctx_end_dates,
    }


# ---------------------------------------------------------------------------
# 评估
# ---------------------------------------------------------------------------

def nse(obs: np.ndarray, sim: np.ndarray) -> float:
    if obs.size == 0:
        return float("nan")
    denom = float(np.sum((obs - obs.mean()) ** 2))
    if denom == 0:
        return float("nan")
    return float(1.0 - np.sum((obs - sim) ** 2) / denom)


def compute_nse_metrics(
    targets: np.ndarray,
    preds: np.ndarray,
    target_observed: np.ndarray,
    target_in_flood: np.ndarray,
) -> dict:
    """三档 NSE 全部基于实测真值，避免插值点污染评估。"""
    flood_mask = target_in_flood & target_observed
    dry_mask = (~target_in_flood) & target_observed
    obs_mask = target_observed

    return {
        "nse_flood": nse(targets[flood_mask], preds[flood_mask]) if flood_mask.any() else float("nan"),
        "nse_dry": nse(targets[dry_mask], preds[dry_mask]) if dry_mask.any() else float("nan"),
        "nse_overall": nse(targets[obs_mask], preds[obs_mask]) if obs_mask.any() else float("nan"),
        "n_flood_eval": int(flood_mask.sum()),
        "n_dry_eval": int(dry_mask.sum()),
        "n_obs_eval": int(obs_mask.sum()),
    }


# ---------------------------------------------------------------------------
# 文件名安全化
# ---------------------------------------------------------------------------

_FORBIDDEN = '<>:"/\\|?*'


def safe_filename(name: str) -> str:
    return "".join("_" if ch in _FORBIDDEN else ch for ch in name).strip() or "unnamed"


# ---------------------------------------------------------------------------
# 单站点预测
# ---------------------------------------------------------------------------

def forecast_single_station(
    model,
    station: str,
    segments: list[Segment],
    output_dir: Path,
    forecast_fn=None,
    file_suffix: str = "forecast",
) -> dict | None:
    """forecast_fn 默认调用 TimesFM；persistence 脚本会传入自定义函数。"""
    print(f"\n{'='*55}")
    print(f"  站点: {station}")
    print(f"{'='*55}")

    seg_lens = [len(s.values) for s in segments]
    print(f"  段数: {len(segments)}  最长段: {max(seg_lens) if seg_lens else 0}")

    bundle = build_windows_from_segments(segments, CONTEXT_LEN, HORIZON)
    if bundle is None:
        print("  无可用窗口，跳过")
        return None

    n_win = len(bundle["contexts"])
    print(f"  滑动窗口数: {n_win}")

    if forecast_fn is None:
        print("  推理中...")
        preds, _ = model.forecast(horizon=HORIZON, inputs=bundle["contexts"])
    else:
        preds = forecast_fn(bundle["contexts"], HORIZON)
    preds = preds[:n_win]

    metrics = compute_nse_metrics(
        bundle["targets"], preds, bundle["target_observed"], bundle["target_in_flood"]
    )
    print(f"  NSE flood   : {metrics['nse_flood']:.4f}  (n={metrics['n_flood_eval']})")
    print(f"  NSE dry     : {metrics['nse_dry']:.4f}  (n={metrics['n_dry_eval']})")
    print(f"  NSE overall : {metrics['nse_overall']:.4f}  (n={metrics['n_obs_eval']})")

    rows = []
    for i in range(n_win):
        for k in range(HORIZON):
            rows.append({
                "station": station,
                "context_end_date": bundle["ctx_end_dates"][i],
                "lead_step": k + 1,
                "forecast_date": bundle["target_dates"][i][k],
                "obs_flow": float(bundle["targets"][i, k]),
                "pred_flow": float(preds[i, k]),
                "in_flood_season": bool(bundle["target_in_flood"][i, k]),
                "horizon_observed": bool(bundle["target_observed"][i, k]),
            })
    out_df = pd.DataFrame(rows)
    out_path = output_dir / f"{safe_filename(station)}_{file_suffix}.csv"
    out_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"  结果已保存: {out_path}")

    return {
        "station": station,
        "n_segments": len(segments),
        "max_seg_len": max(seg_lens) if seg_lens else 0,
        "n_windows": n_win,
        **metrics,
    }


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main():
    output_dir = Path(__file__).parent / "output_liaohe"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"加载数据: {DATA_FILE}")
    stations = load_all_stations(DATA_FILE)
    print(f"共检测到 {len(stations)} 个站点")

    print("构建连续段...")
    station_segments = build_station_segments(stations)
    total_segs = sum(len(v) for v in station_segments.values())
    print(f"总段数: {total_segs}")

    print("加载 TimesFM 2.5 (200M) PyTorch 模型...")
    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
    model.compile(
        timesfm.ForecastConfig(
            max_context=CONTEXT_LEN,
            max_horizon=HORIZON,
            per_core_batch_size=BATCH_SIZE,
            normalize_inputs=True,
            infer_is_positive=True,
        )
    )
    print("模型加载完成。")

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
            r = forecast_single_station(model, station, segments, output_dir)
            if r is not None:
                results.append(r)
            else:
                skipped.append({"station": station, "reason": "no usable windows"})
        except Exception as e:
            print(f"  [跳过] {station}: {e}")
            skipped.append({"station": station, "reason": str(e)})

    if results:
        summary_df = pd.DataFrame(results)
        summary_path = output_dir / "nse_summary.csv"
        summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

        print(f"\n{'='*55}")
        print("  汇总结果")
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
