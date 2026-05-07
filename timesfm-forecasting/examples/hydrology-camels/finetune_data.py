#!/usr/bin/env python3
"""
统一加载 CAMELS-US + 辽河松花江数据 → Segment 列表 → 随机窗口数据集。

设计：
  1. CAMELS-US 与辽河共用 run_forecast_liaohe.build_segments 切段逻辑
     （汛期 ffill / 非汛期插值 / gap 超阈值切段）。
  2. 训练/评估按"段内时间"切分（前 train_ratio 训练，后 1-train_ratio 评估），
     避免未来数据泄漏到训练集。
  3. RandomWindowDataset 只采样 target 全实测的窗口，防止 loss 拟合插值点。
  4. 整体 segments 字典支持 pickle 缓存，避免每次重新解析 671 个 CAMELS 站。
"""
from __future__ import annotations

import pickle
import random
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
) -> tuple[Segment | None, Segment | None]:
    n = len(seg.values)
    cut = int(n * train_ratio)
    if cut <= 0 or cut >= n:
        return (seg, None) if cut >= n else (None, seg)
    train = Segment(
        dates=seg.dates[:cut],
        values=seg.values[:cut],
        is_observed=seg.is_observed[:cut],
        in_flood=seg.in_flood[:cut],
    )
    val = Segment(
        dates=seg.dates[cut:],
        values=seg.values[cut:],
        is_observed=seg.is_observed[cut:],
        in_flood=seg.in_flood[cut:],
    )
    return train, val


def split_by_time(
    station_segments: dict[str, list[Segment]],
    train_ratio: float = 0.8,
) -> tuple[dict[str, list[Segment]], dict[str, list[Segment]]]:
    train_d: dict[str, list[Segment]] = {}
    val_d: dict[str, list[Segment]] = {}
    for name, segs in station_segments.items():
        ts, vs = [], []
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


def filter_liaohe(
    station_segments: dict[str, list[Segment]]
) -> dict[str, list[Segment]]:
    return {k: v for k, v in station_segments.items() if k.startswith("liaohe_")}


def filter_camels(
    station_segments: dict[str, list[Segment]]
) -> dict[str, list[Segment]]:
    return {k: v for k, v in station_segments.items() if k.startswith("camels_")}


def flatten_segments(
    station_segments: dict[str, list[Segment]]
) -> list[Segment]:
    out: list[Segment] = []
    for segs in station_segments.values():
        out.extend(segs)
    return out


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
        segments: list[Segment],
        context_len: int,
        horizon: int,
        num_samples: int,
        seed: int = 42,
        max_attempts_factor: int = 20,
    ):
        self.context_len = context_len
        self.horizon = horizon
        min_len = context_len + horizon

        valid = [s for s in segments if len(s.values) >= min_len]
        if not valid:
            raise ValueError(
                f"无段满足 context_len={context_len} + horizon={horizon}"
            )
        # 加权（按段长）抽样，长段被采到更多次
        weights = np.array([len(s.values) - min_len + 1 for s in valid],
                           dtype=np.float64)
        weights = weights / weights.sum()

        rng = np.random.default_rng(seed)
        self.windows: list[tuple[np.ndarray, np.ndarray]] = []
        max_attempts = num_samples * max_attempts_factor
        attempts = 0
        while len(self.windows) < num_samples and attempts < max_attempts:
            attempts += 1
            i = rng.choice(len(valid), p=weights)
            seg = valid[i]
            n = len(seg.values)
            start = rng.integers(0, n - min_len + 1)
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
