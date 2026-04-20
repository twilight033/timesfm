#!/usr/bin/env python3
"""
TimesFM 零样本水文预测 —— CAMELS 数据集（日尺度）

策略：滑动窗口，每次用前 365 天预测下 1 天，汇总所有预测值与真实值计算 NSE。

CAMELS-US 数据格式（usgs_streamflow）：
    gauge_id  year  month  day  streamflow(cfs)  qc_flag
    -999 表示缺测，会被自动剔除。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import timesfm

# ============================================================
# 配置区：按需修改
# ============================================================

# CAMELS-US 数据集根目录
DATA_DIR = Path(r"D:\download\camels")

# 要预测的站点列表（USGS gauge ID）
GAUGE_IDS = [
    "01013500",
    "01022500",
    "01030500",
]

# ============================================================

CONTEXT_LEN = 365   # 上下文长度：1 年
HORIZON = 1         # 预测步长：1 天
BATCH_SIZE = 32     # 每批推理的窗口数


# ---------------------------------------------------------------------------
# 数据加载
# ---------------------------------------------------------------------------

def load_camels_us(data_dir: Path, gauge_id: str) -> pd.Series:
    """加载 CAMELS-US 单站点日径流序列，返回按日期索引的 Series（单位 cfs）。"""
    files = list(data_dir.glob(f"**/{gauge_id}_streamflow_qc.txt"))
    if not files:
        raise FileNotFoundError(f"未找到站点 {gauge_id} 的径流文件，请检查 data_dir 路径。")

    df = pd.read_csv(
        files[0],
        sep=r"\s+",
        header=None,
        names=["gauge_id", "year", "month", "day", "streamflow", "qc"],
    )
    df["date"] = pd.to_datetime(df[["year", "month", "day"]])
    df["streamflow"] = pd.to_numeric(df["streamflow"], errors="coerce")
    # 剔除缺测（-999）和负值
    df = df[df["streamflow"] >= 0].set_index("date")["streamflow"]
    df = df.sort_index()
    return df.astype(np.float32)


# ---------------------------------------------------------------------------
# 预测
# ---------------------------------------------------------------------------

def build_windows(series: np.ndarray, context: int) -> tuple[list[np.ndarray], np.ndarray]:
    """从连续序列构建滑动窗口。返回 (windows, true_values)。"""
    n = len(series)
    if n <= context:
        raise ValueError(f"数据长度 {n} 不足以构建长度为 {context} 的上下文窗口。")
    windows = [series[i : i + context] for i in range(n - context)]
    true_values = series[context:]
    return windows, true_values


def run_forecast(model: timesfm.TimesFM_2p5_200M_torch, windows: list[np.ndarray]) -> np.ndarray:
    """批量推理，返回每个窗口下一步的点预测，shape=(n_windows,)。"""
    point_forecast, _ = model.forecast(horizon=HORIZON, inputs=windows)
    return point_forecast[:, 0]  # (n_windows, 1) -> (n_windows,)


# ---------------------------------------------------------------------------
# 评估
# ---------------------------------------------------------------------------

def nse(obs: np.ndarray, sim: np.ndarray) -> float:
    """Nash-Sutcliffe Efficiency。"""
    denom = np.sum((obs - obs.mean()) ** 2)
    if denom == 0:
        return float("nan")
    return float(1.0 - np.sum((obs - sim) ** 2) / denom)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def forecast_single_gauge(
    model: timesfm.TimesFM_2p5_200M_torch,
    data_dir: Path,
    gauge_id: str,
    output_dir: Path,
) -> dict:
    print(f"\n{'='*55}")
    print(f"  站点: {gauge_id}")
    print(f"{'='*55}")

    series = load_camels_us(data_dir, gauge_id)
    print(f"  数据范围: {series.index[0].date()} ~ {series.index[-1].date()}  ({len(series)} 天)")

    windows, true_values = build_windows(series.values, CONTEXT_LEN)
    print(f"  滑动窗口数: {len(windows)}")

    print("  推理中...")
    pred_values = run_forecast(model, windows)

    score = nse(true_values, pred_values)
    print(f"  NSE = {score:.4f}")

    # 保存预测结果
    result_dates = series.index[CONTEXT_LEN:]
    out_df = pd.DataFrame({
        "date": result_dates,
        "obs_cfs": true_values,
        "pred_cfs": pred_values,
    })
    out_path = output_dir / f"{gauge_id}_forecast.csv"
    out_df.to_csv(out_path, index=False)
    print(f"  结果已保存: {out_path}")

    return {"gauge_id": gauge_id, "n_days": len(true_values), "nse": score}


def main():
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("加载 TimesFM 2.5 (200M) PyTorch 模型...")
    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
    model.compile(
        timesfm.ForecastConfig(
            max_context=CONTEXT_LEN,
            max_horizon=HORIZON,
            per_core_batch_size=BATCH_SIZE,
            normalize_inputs=True,   # 径流量级差异大，归一化有助于稳定推理
            infer_is_positive=True,  # 径流为非负值
        )
    )
    print("模型加载完成。")

    results = []
    for gid in GAUGE_IDS:
        try:
            r = forecast_single_gauge(model, DATA_DIR, gid, output_dir)
            results.append(r)
        except Exception as e:
            print(f"  [跳过] {gid}: {e}")

    summary_df = pd.DataFrame(results)
    summary_path = output_dir / "nse_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    print(f"\n{'='*55}")
    print("  汇总结果")
    print(f"{'='*55}")
    print(summary_df.to_string(index=False))
    if len(results) > 1:
        valid = summary_df["nse"].dropna()
        print(f"\n  中位数 NSE: {valid.median():.4f}")
        print(f"  平均   NSE: {valid.mean():.4f}")
    print(f"\n  汇总已保存: {summary_path}")


if __name__ == "__main__":
    main()
