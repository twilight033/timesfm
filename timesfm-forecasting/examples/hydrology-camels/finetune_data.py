#!/usr/bin/env python3
"""
统一加载 CAMELS-US + 辽河松花江数据 → Segment 列表 → 随机窗口数据集。

设计：
  1. CAMELS-US 与辽河共用 run_forecast_liaohe.build_segments 切段逻辑
     （汛期 ffill / 非汛期插值 / gap 超阈值切段）。
  2. 训练/评估按"段内时间"切分：
       - train 段：截到前 train_ratio，避免未来泄漏。
       - val 段：保留完整 segment + val_start 标记（EvalSegment），
         允许 context 跨切分点回看 train 期，仅约束 target 必须落在 val 期。
  3. RandomWindowDataset 只采样 target 全实测的窗口，防止 loss 拟合插值点；
     遇到带 val_start 的段时，采样下限自动收紧到使 target 落在 val 期。
  4. 整体 segments 字典支持 pickle 缓存，避免每次重新解析 671 个 CAMELS 站。
"""
from __future__ import annotations

import pickle
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from config import DATA_DIR as CAMELS_ROOT
from run_forecast_liaohe import (
    DATA_FILE as LIAOHE_FILE,
    Segment,
    build_segments,
    load_all_stations as _load_liaohe_stations,
)


@dataclass
class EvalSegment:
    """val 评估专用段：持有完整序列 + val_start 索引。

    val_start 表示 target 必须 >= 该索引（半开区间下限）；context 可以从
    val_start - context_len 这种位置开始，跨越切分点回看到 train 期。
    """
    dates: pd.DatetimeIndex
    values: np.ndarray
    is_observed: np.ndarray
    in_flood: np.ndarray
    val_start: int = 0

CAMELS_STREAMFLOW_ROOT = (
    CAMELS_ROOT
    / "camels_us"
    / "CAMELS_US"
    / "basin_timeseries_v1p2_metForcing_obsFlow"
    / "basin_dataset_public_v1p2"
    / "usgs_streamflow"
)

CACHE_DIR = Path(__file__).parent / "_finetune_cache"


# ---------------------------------------------------------------------------
# CAMELS-US 加载
# ---------------------------------------------------------------------------

def _load_camels_us_station(file_path: Path) -> pd.Series:
    df = pd.read_csv(
        file_path,
        sep=r"\s+",
        header=None,
        names=["gauge_id", "year", "month", "day", "streamflow", "qc"],
    )
    df["date"] = pd.to_datetime(df[["year", "month", "day"]])
    df["streamflow"] = pd.to_numeric(df["streamflow"], errors="coerce")
    df = df[df["streamflow"] >= 0]
    return df.set_index("date")["streamflow"].sort_index().astype(np.float32)


def discover_camels_files(
    max_basins: int | None = None, seed: int = 42
) -> list[Path]:
    files = sorted(CAMELS_STREAMFLOW_ROOT.glob("**/*_streamflow_qc.txt"))
    if max_basins is not None and len(files) > max_basins:
        rng = np.random.default_rng(seed)
        idx = sorted(rng.choice(len(files), size=max_basins, replace=False))
        files = [files[i] for i in idx]
    return files


def load_camels_segments(
    max_basins: int | None = None, seed: int = 42
) -> dict[str, list[Segment]]:
    """每个 CAMELS-US 站点 → segments；剔 -999 后复用辽河切段逻辑。"""
    files = discover_camels_files(max_basins, seed=seed)
    out: dict[str, list[Segment]] = {}
    for i, f in enumerate(files):
        gauge_id = f.stem.split("_")[0]
        try:
            s = _load_camels_us_station(f)
            if len(s) == 0:
                continue
            segs = build_segments(s)
            if segs:
                out[f"camels_{gauge_id}"] = segs
        except Exception as e:
            print(f"  [skip CAMELS] {gauge_id}: {e}")
        if (i + 1) % 50 == 0:
            print(f"  CAMELS 加载进度 {i + 1}/{len(files)}")
    return out


def load_liaohe_segments() -> dict[str, list[Segment]]:
    stations = _load_liaohe_stations(LIAOHE_FILE)
    out: dict[str, list[Segment]] = {}
    for name, s in stations.items():
        segs = build_segments(s)
        if segs:
            out[f"liaohe_{name}"] = segs
    return out


# ---------------------------------------------------------------------------
# 缓存合并
# ---------------------------------------------------------------------------

def build_or_load_segments(
    max_camels_basins: int | None = 200,
    seed: int = 42,
    refresh: bool = False,
) -> dict[str, list[Segment]]:
    CACHE_DIR.mkdir(exist_ok=True)
    tag = "all" if max_camels_basins is None else str(max_camels_basins)
    cache = CACHE_DIR / f"segments_camels{tag}_seed{seed}.pkl"
    if cache.exists() and not refresh:
        print(f"  [cache] 加载 {cache}")
        with cache.open("rb") as f:
            return pickle.load(f)
    print(f"  构建 segments（CAMELS≤{max_camels_basins} + 辽河全部）...")
    camels = load_camels_segments(max_camels_basins, seed=seed)
    liaohe = load_liaohe_segments()
    merged = {**camels, **liaohe}
    print(f"  CAMELS 站 {len(camels)}, 辽河 站 {len(liaohe)}, 共 {len(merged)}")
    with cache.open("wb") as f:
        pickle.dump(merged, f)
    print(f"  [cache] 已写入 {cache}")
    return merged


# ---------------------------------------------------------------------------
# 时间切分
# ---------------------------------------------------------------------------

def _split_segment(
    seg: Segment, train_ratio: float
) -> tuple[Segment | None, EvalSegment | None]:
    """train 段截到前 cut；val 段保留完整序列并带 val_start=cut。

    val 段持有完整序列后，下游滑窗可以让 context 从 val_start - context_len
    开始（跨切分点回看 train 期），只要 target 落在 val 期即可。
    """
    n = len(seg.values)
    cut = int(n * train_ratio)
    if cut >= n:
        # 整段全部归 train，无 val
        return seg, None
    if cut <= 0:
        # 整段全部归 val（cut=0 表示无 train 部分，val 不受下限约束）
        val = EvalSegment(
            dates=seg.dates,
            values=seg.values,
            is_observed=seg.is_observed,
            in_flood=seg.in_flood,
            val_start=0,
        )
        return None, val
    train = Segment(
        dates=seg.dates[:cut],
        values=seg.values[:cut],
        is_observed=seg.is_observed[:cut],
        in_flood=seg.in_flood[:cut],
    )
    val = EvalSegment(
        dates=seg.dates,
        values=seg.values,
        is_observed=seg.is_observed,
        in_flood=seg.in_flood,
        val_start=cut,
    )
    return train, val


def split_by_time(
    station_segments: dict[str, list[Segment]],
    train_ratio: float = 0.8,
) -> tuple[dict[str, list[Segment]], dict[str, list[EvalSegment]]]:
    train_d: dict[str, list[Segment]] = {}
    val_d: dict[str, list[EvalSegment]] = {}
    for name, segs in station_segments.items():
        ts: list[Segment] = []
        vs: list[EvalSegment] = []
        for seg in segs:
            t, v = _split_segment(seg, train_ratio)
            if t is not None:
                ts.append(t)
            if v is not None:
                vs.append(v)
        if ts:
            train_d[name] = ts
        if vs:
            val_d[name] = vs
    return train_d, val_d


def filter_liaohe(station_segments: dict[str, list]) -> dict[str, list]:
    """按 key 前缀筛站点；元素是 Segment 或 EvalSegment 均可。"""
    return {k: v for k, v in station_segments.items() if k.startswith("liaohe_")}


def filter_camels(station_segments: dict[str, list]) -> dict[str, list]:
    """按 key 前缀筛站点；元素是 Segment 或 EvalSegment 均可。"""
    return {k: v for k, v in station_segments.items() if k.startswith("camels_")}


def flatten_segments(station_segments: dict[str, list]) -> list:
    """把 dict[站名 -> 段列表] 平铺成单个列表；Segment / EvalSegment 通用。"""
    out: list = []
    for segs in station_segments.values():
        out.extend(segs)
    return out


# ---------------------------------------------------------------------------
# 评估滑窗：只约束 target 落在 val 期，context 可跨切分点
# ---------------------------------------------------------------------------

def build_eval_windows(
    segments: list,
    context_len: int,
    horizon: int,
) -> dict | None:
    """对 EvalSegment 列表做滑窗，只保留 target 落在 val 期的窗口。

    与 run_forecast_liaohe.build_windows_from_segments 的差异：
      - context 起点可以早于 val_start（跨切分点回看 train 期）；
      - 只生成 target_start >= seg.val_start 的窗口。

    元素若无 val_start 属性，按 0 处理（行为退化为遍历全段所有窗口），
    便于在零样本场景或老对象上复用。
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
        val_start = int(getattr(seg, "val_start", 0))
        # i 是 context 起点；target 起点 = i + context_len，要求 >= val_start
        i_lo = max(0, val_start - context_len)
        i_hi = n - context_len - horizon + 1  # 半开
        if i_hi <= i_lo:
            continue
        for i in range(i_lo, i_hi):
            t_start = i + context_len
            t_slice = slice(t_start, t_start + horizon)
            contexts.append(seg.values[i : i + context_len])
            targets.append(seg.values[t_slice])
            tgt_dates.append(seg.dates[t_slice])
            tgt_observed.append(seg.is_observed[t_slice])
            tgt_in_flood.append(seg.in_flood[t_slice])
            ctx_end_dates.append(seg.dates[t_start - 1])

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
# 随机窗口数据集
# ---------------------------------------------------------------------------

class RandomWindowDataset(Dataset):
    """从 segments 中随机采样 (context, target) 窗口。

    - 只保留 target 全实测的窗口（is_observed 全 True），避免 loss 拟合插值点。
    - 不做外部归一化，直接喂原始流量值（TimesFM 内置 RevIN）。
    """

    def __init__(
        self,
        segments: list,
        context_len: int,
        horizon: int,
        num_samples: int,
        seed: int = 42,
        max_attempts_factor: int = 20,
    ):
        """segments 元素可以是 Segment 或 EvalSegment。

        - EvalSegment：val_start 起约束 target 起点 >= val_start（context 可跨段）。
        - Segment：等价于 val_start=0，行为同旧版。
        """
        self.context_len = context_len
        self.horizon = horizon
        min_len = context_len + horizon

        # valid: 满足最短长度；同时计算每段 context 起点的可采样区间 [lo, hi)
        valid: list = []
        seg_ranges: list[tuple[int, int]] = []
        for s in segments:
            n = len(s.values)
            if n < min_len:
                continue
            val_start = int(getattr(s, "val_start", 0))
            lo = max(0, val_start - context_len)
            hi = n - min_len + 1  # 半开
            if hi <= lo:
                continue
            valid.append(s)
            seg_ranges.append((lo, hi))
        if not valid:
            raise ValueError(
                f"无段满足 context_len={context_len} + horizon={horizon}"
                "（含 val_start 约束）"
            )
        # 加权（按可采样窗口数）抽样
        weights = np.array(
            [hi - lo for lo, hi in seg_ranges], dtype=np.float64
        )
        weights = weights / weights.sum()

        rng = np.random.default_rng(seed)
        self.windows: list[tuple[np.ndarray, np.ndarray]] = []
        max_attempts = num_samples * max_attempts_factor
        attempts = 0
        while len(self.windows) < num_samples and attempts < max_attempts:
            attempts += 1
            i = rng.choice(len(valid), p=weights)
            seg = valid[i]
            lo, hi = seg_ranges[i]
            start = rng.integers(lo, hi)
            tgt_obs = seg.is_observed[start + context_len : start + min_len]
            if not tgt_obs.all():
                continue
            ctx = seg.values[start : start + context_len].astype(np.float32)
            tgt = seg.values[start + context_len : start + min_len].astype(
                np.float32
            )
            self.windows.append((ctx, tgt))
        if len(self.windows) < num_samples:
            print(
                f"  [warn] RandomWindowDataset 只收集到 "
                f"{len(self.windows)}/{num_samples} 个全实测窗口"
            )

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, i: int):
        ctx, tgt = self.windows[i]
        return torch.from_numpy(ctx), torch.from_numpy(tgt)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
