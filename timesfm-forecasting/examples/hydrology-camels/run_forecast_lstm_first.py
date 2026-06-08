"""
LSTM-First + TimesFM 残差 水文预测 —— CAMELS-US（反向 cascade）

流程（与前向 run_forecast_lstm_xreg.py 对照）：
  前向：微调 TimesFM 出基线 → LSTM 学残差 → final = tf_hor + lstm_resid
  反向（本脚本）：
    1. 训练 LSTM 预测径流（仅强迫 ⊕ 静态，不含历史径流真值）
    2. 滚动推理 → lstm_pred[t]；residual = streamflow - lstm_pred
    3. zero-shot TimesFM 对 residual 序列滚动预测 → tf_resid
    4. final = lstm_pred + tf_resid

关键判据：nse_lstm_first vs nse_lstm_only —— TimesFM 残差步是否真带来提升。
若残差近白噪声（lag-1 自相关≈0），该步无效，会如实反映在结果里。

用法：
  # 首次运行（训练 LSTM + zero-shot TimesFM 残差 + 评估）：
  python run_forecast_lstm_first.py

  # 单站冒烟测试：
  python run_forecast_lstm_first.py --gauges 01013500 --epochs 3

  # 加载已有 LSTM 直接推理：
  python run_forecast_lstm_first.py --skip_train

  # 缩短 TimesFM 残差上下文（减少前段有效区损失）：
  python run_forecast_lstm_first.py --tf_ctx 128
"""
from __future__ import annotations

import argparse
import hashlib
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from config import DATA_DIR, GAUGE_IDS
from lstm_first import (
    DYNAMIC_COLS,
    STATIC_ATTR_COLS,
    LSTMStreamflowPredictor,
    align_forcing_streamflow,
    load_camels_forcing,
    load_camels_static_attrs,
    rolling_predict_full,
    train_lstm_streamflow,
)
from run_forecast_lstm_xreg import (
    CONTEXT_LEN,
    DEFAULT_MODEL_ID,
    HORIZON,
    STREAMFLOW_ROOT,
    _load_finetuned_model,
    load_camels_us,
    nse,
    precompute_tf_forecasts,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

BATCH_SIZE = 64   # TimesFM 残差推理批大小

CKPT_DIR = Path(__file__).parent / "_lstm_first_ckpt"
OUTPUT_DIR = Path(__file__).parent / "output_lstm_first"
CACHE_DIR = Path(__file__).parent / "_lstm_first_cache"


# ---------------------------------------------------------------------------
# 数据准备（不预计算 TimesFM —— TimesFM 用在残差上，见主流程）
# ---------------------------------------------------------------------------

def load_gauge_data(
    gauge_id: str,
    static_df: pd.DataFrame,
    train_frac: float = 0.8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int] | None:
    """加载单站数据，返回 (dyn [T,N_DYN], sf [T], static [N_STATIC], split_idx) 或 None。"""
    try:
        streamflow = load_camels_us(STREAMFLOW_ROOT, gauge_id)
        forcing = load_camels_forcing(DATA_DIR, gauge_id)
        aligned = align_forcing_streamflow(forcing, streamflow)
    except FileNotFoundError as e:
        logger.warning("[%s] 文件缺失: %s", gauge_id, e)
        return None

    if len(aligned) < CONTEXT_LEN + HORIZON:
        logger.warning("[%s] 数据仅 %d 天，不足，跳过", gauge_id, len(aligned))
        return None

    if gauge_id not in static_df.index:
        logger.warning("[%s] 静态属性缺失，跳过", gauge_id)
        return None

    dyn = aligned[DYNAMIC_COLS].values.astype(np.float32)
    sf = aligned["streamflow"].values.astype(np.float32)
    static = static_df.loc[gauge_id, STATIC_ATTR_COLS].values.astype(np.float32)

    # 静态属性 NaN 用全体均值填充
    if np.isnan(static).any():
        col_means = static_df[STATIC_ATTR_COLS].mean().values.astype(np.float32)
        static = np.where(np.isnan(static), col_means, static)

    split_idx = int(len(sf) * train_frac)
    return dyn, sf, static, split_idx


def discover_all_gauges() -> list[str]:
    """递归扫描 usgs_streamflow 目录，发现 CAMELS-US 全部站点 gauge_id（排序、去重）。"""
    files = STREAMFLOW_ROOT.glob("**/*_streamflow_qc.txt")
    gauges = {f.stem.split("_")[0] for f in files}
    return sorted(gauges)


# ---------------------------------------------------------------------------
# TimesFM 对残差序列滚动预测（处理 NaN 前缀 + 有效区对齐）
# ---------------------------------------------------------------------------

def _arr_hash(a: np.ndarray) -> str:
    """对数组取短哈希（NaN 归零后），用作缓存 key —— 残差变化则缓存自动失效。"""
    buf = np.ascontiguousarray(np.nan_to_num(a, nan=0.0).astype(np.float32)).tobytes()
    return hashlib.md5(buf).hexdigest()[:12]


def tf_predict_on_residual(
    residual: np.ndarray,   # [T]，前 CONTEXT_LEN 天为 NaN
    model_hf,
    tf_ctx: int,
    device: str,
    gauge_id: str,
    use_cache: bool = True,
) -> np.ndarray:
    """zero-shot TimesFM 对残差序列滚动预测，返回长度 T 的 tf_resid。

    残差有效区为 [CONTEXT_LEN, T)。只把有效段喂给 TimesFM，避免 NaN 上下文，
    再按偏移对齐回全序列。tf_resid 的有效区为 [CONTEXT_LEN + tf_ctx, T)。
    """
    T = len(residual)
    tf_resid = np.full(T, np.nan, dtype=np.float32)

    valid = residual[CONTEXT_LEN:]            # 残差有效段
    valid = valid[~np.isnan(valid)] if np.isnan(valid).any() else valid
    # 正常情况下残差有效段无内部 NaN；若有（数据缺口），上面的过滤会错位，
    # 故这里断言连续性，异常则跳过该站。
    if len(valid) != T - CONTEXT_LEN:
        logger.warning("[%s] 残差有效段含内部 NaN，跳过 TimesFM 残差预测", gauge_id)
        return tf_resid
    if len(valid) < tf_ctx + HORIZON:
        logger.warning("[%s] 残差有效段过短（%d < %d），跳过", gauge_id, len(valid), tf_ctx + HORIZON)
        return tf_resid

    cache_path = CACHE_DIR / f"{gauge_id}_ctx{tf_ctx}_{_arr_hash(residual)}.npy"
    if use_cache and cache_path.exists():
        cached = np.load(cache_path)
        if len(cached) == T:
            logger.info("[%s] 从缓存加载 tf_resid", gauge_id)
            return cached

    logger.info("[%s] zero-shot TimesFM 对残差滚动预测（有效段 %d 天，tf_ctx=%d）...",
                gauge_id, len(valid), tf_ctx)
    # precompute_tf_forecasts: tf_on_valid[s+tf_ctx] = TimesFM(valid[s:s+tf_ctx])
    tf_on_valid = precompute_tf_forecasts(
        valid, model_hf, tf_ctx, HORIZON, BATCH_SIZE, device
    )  # 长度 = len(valid)，前 tf_ctx 个为 NaN
    # 对齐回全序列：valid 索引 k 对应全序列 CONTEXT_LEN + k
    tf_resid[CONTEXT_LEN:] = tf_on_valid

    if use_cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, tf_resid)
        logger.info("[%s] tf_resid 已缓存至 %s", gauge_id, cache_path)

    return tf_resid


# ---------------------------------------------------------------------------
# 残差诊断
# ---------------------------------------------------------------------------

def lag1_autocorr(x: np.ndarray) -> float:
    """残差 lag-1 自相关：接近 0 → 近白噪声 → TimesFM 残差步预期无效。"""
    x = x[~np.isnan(x)]
    if len(x) < 3:
        return float("nan")
    x0, x1 = x[:-1], x[1:]
    s0, s1 = x0.std(), x1.std()
    if s0 < 1e-9 or s1 < 1e-9:
        return float("nan")
    return float(np.mean((x0 - x0.mean()) * (x1 - x1.mean())) / (s0 * s1))


# ---------------------------------------------------------------------------
# 单站评估
# ---------------------------------------------------------------------------

def evaluate_gauge(
    gauge_id: str,
    sf: np.ndarray,
    lstm_pred: np.ndarray,
    tf_resid: np.ndarray,
    residual: np.ndarray,
    eval_start: int,
    diagnose_only: bool = False,
) -> dict | None:
    """在评估期（target >= eval_start）算 NSE。

    nse_lstm_first：final = lstm_pred + tf_resid
    nse_lstm_only ：lstm_pred（同一批评估点，对比公平）
    diagnose_only=True：不依赖 TimesFM，只在 lstm_pred 有效点上报 lstm_only + 残差诊断
                        （nse_lstm_first 置 NaN），用于无 transformers 环境快速判断方向。
    """
    idx = np.arange(len(sf))
    if diagnose_only:
        mask = (idx >= eval_start) & ~np.isnan(lstm_pred)
    else:
        final = lstm_pred + tf_resid  # NaN 自然传播
        mask = (idx >= eval_start) & ~np.isnan(final) & ~np.isnan(lstm_pred)
    if mask.sum() == 0:
        logger.warning("[%s] 无有效评估点，跳过", gauge_id)
        return None

    obs = sf[mask]
    # 残差诊断在评估期残差上算
    res_eval = residual[(idx >= eval_start) & ~np.isnan(residual)]

    return {
        "gauge_id":          gauge_id,
        "n_points":          int(mask.sum()),
        "nse_lstm_first":    float("nan") if diagnose_only else nse(obs, (lstm_pred + tf_resid)[mask]),
        "nse_lstm_only":     nse(obs, lstm_pred[mask]),
        "res_lag1_autocorr": lag1_autocorr(res_eval),
        "tf_resid_std":      float("nan") if diagnose_only else float(np.nanstd(tf_resid[mask])),
        "obs_std":           float(obs.std()),
    }


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.all_gauges:
        eval_gauges = discover_all_gauges()
        logger.info("发现 CAMELS-US 全部站点 %d 个", len(eval_gauges))
        if args.max_gauges is not None:
            eval_gauges = eval_gauges[: args.max_gauges]
            logger.info("按 --max_gauges 限制为前 %d 个站点", len(eval_gauges))
    else:
        eval_gauges = args.gauges or GAUGE_IDS
    # 默认：训练与评估用同一批站点（多站联合训练 + 各站时间切分评估，CAMELS LSTM 标准做法）
    train_gauges = args.train_gauges or eval_gauges

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("使用设备: %s", device)

    # ---- 1. 静态属性 ----
    logger.info("加载 CAMELS 静态流域属性（27 列）...")
    static_df = load_camels_static_attrs(DATA_DIR)
    logger.info("静态属性已加载：%d 个站点 × %d 列", len(static_df), len(static_df.columns))

    # ---- 2. 预加载所有站点（训练 + 评估，去重） ----
    all_gauges = sorted(set(train_gauges) | set(eval_gauges))
    gauge_cache: dict[str, tuple] = {}
    for gid in all_gauges:
        result = load_gauge_data(gid, static_df, train_frac=args.train_frac)
        if result is not None:
            gauge_cache[gid] = result

    if not gauge_cache:
        raise RuntimeError("没有可用站点，请检查数据路径。")

    # 训练数组：每站只取训练期 [:split_idx]，传 3 元组 (dyn, sf, static)
    train_arrays = []
    for gid in sorted(set(train_gauges)):
        if gid not in gauge_cache:
            continue
        dyn, sf, static, split_idx = gauge_cache[gid]
        train_arrays.append((dyn[:split_idx], sf[:split_idx], static))
    if not train_arrays:
        raise RuntimeError("没有可用的训练站点，请检查数据路径。")

    # ---- 3. 训练 or 加载 LSTM（预测径流） ----
    if not args.skip_train:
        logger.info("开始训练 LSTM 预测径流（%d 个站点）...", len(train_arrays))
        model_lstm, dyn_mean, dyn_std, static_mean, static_std = train_lstm_streamflow(
            gauge_arrays=train_arrays,
            context_len=CONTEXT_LEN,
            horizon=HORIZON,
            hidden=args.hidden,
            n_layers=args.n_layers,
            epochs=args.epochs,
            batch_size=args.train_batch,
            lr=args.lr,
            stride=args.stride,
            device=device,
            save_dir=CKPT_DIR,
        )
        predictor = LSTMStreamflowPredictor(
            model_lstm, dyn_mean, dyn_std, static_mean, static_std, device=device
        )
    else:
        logger.info("从 %s 加载已有 LSTM...", CKPT_DIR)
        predictor = LSTMStreamflowPredictor.from_checkpoint(CKPT_DIR, device=device)

    if args.train_only:
        logger.info("--train_only 模式，训练完成，退出。")
        return

    # ---- 4. zero-shot TimesFM（用于残差预测；diagnose_only 跳过） ----
    if args.diagnose_only:
        logger.info("--diagnose_only 模式：跳过 TimesFM，仅做 LSTM + 残差诊断。")
        model_hf = None
    else:
        logger.info("加载 zero-shot TimesFM 从 %s...", args.model_id)
        model_hf = _load_finetuned_model("zero_shot", args.model_id, Path("."), device)
        logger.info("zero-shot TimesFM 加载完成。")

    # ---- 5. 逐站推理评估 ----
    results = []
    for gid in eval_gauges:
        if gid not in gauge_cache:
            logger.warning("[%s] 数据不可用，跳过", gid)
            continue
        logger.info("推理站点: %s", gid)
        dyn, sf, static, split_idx = gauge_cache[gid]

        # 5a. LSTM 滚动预测径流（全序列）
        lstm_pred = rolling_predict_full(predictor, dyn, sf, static, CONTEXT_LEN, HORIZON)
        residual = sf - lstm_pred   # 前 CONTEXT_LEN 天为 NaN

        # 5b. zero-shot TimesFM 预测残差（diagnose_only 时置 NaN）
        if args.diagnose_only:
            tf_resid = np.full(len(sf), np.nan, dtype=np.float32)
        else:
            tf_resid = tf_predict_on_residual(
                residual, model_hf, args.tf_ctx, device, gid, use_cache=not args.refresh
            )

        # 5c. 评估（target 落在 [split_idx, T) 且 final 有效）
        try:
            r = evaluate_gauge(gid, sf, lstm_pred, tf_resid, residual,
                               eval_start=split_idx, diagnose_only=args.diagnose_only)
            if r is not None:
                results.append(r)
                logger.info(
                    "  NSE  lstm_first=%.4f  lstm_only=%.4f  res_lag1=%.3f  (n=%d)",
                    r["nse_lstm_first"], r["nse_lstm_only"],
                    r["res_lag1_autocorr"], r["n_points"],
                )
        except Exception as e:
            logger.warning("[%s] 评估失败: %s", gid, e)

    # ---- 6. 汇总 ----
    if not results:
        logger.warning("没有有效结果。")
        return

    summary_df = pd.DataFrame(results)
    out_path = OUTPUT_DIR / "nse_summary.csv"
    summary_df.to_csv(out_path, index=False, encoding="utf-8-sig")

    print(f"\n{'='*60}")
    print("  汇总结果（反向 cascade：LSTM 先 + TimesFM 残差）")
    print(f"{'='*60}")
    print(summary_df.to_string(index=False))

    nse_cols = [c for c in summary_df.columns if c.startswith("nse_")]
    if len(results) > 1:
        print("\n  各方法中位数 NSE:")
        for col in nse_cols:
            vals = summary_df[col].dropna()
            if len(vals):
                print(f"    {col}: {vals.median():.4f}  (n={len(vals)})")
        lag1 = summary_df["res_lag1_autocorr"].dropna()
        if len(lag1):
            print(f"\n  残差 lag-1 自相关中位数: {lag1.median():.3f}"
                  "  （接近 0 表示残差近白噪声，TimesFM 残差步难有提升）")

    print(f"\n  汇总已保存: {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="LSTM-First + TimesFM 残差 水文预测（CAMELS-US，反向 cascade）"
    )
    p.add_argument("--gauges", nargs="+", default=None,
                   help="评估站点列表（默认 config.py 的 GAUGE_IDS）")
    p.add_argument("--train_gauges", nargs="+", default=None,
                   help="训练 LSTM 的站点列表（默认与评估站点相同）")
    p.add_argument("--all_gauges", action="store_true",
                   help="在 CAMELS-US 全部站点上训练+评估（自动发现，覆盖 --gauges）")
    p.add_argument("--max_gauges", type=int, default=None,
                   help="限制站点数量（配合 --all_gauges，便于分批/测试）")
    p.add_argument("--skip_train", action="store_true", help="跳过训练，从 _lstm_first_ckpt 加载")
    p.add_argument("--train_only", action="store_true", help="只训练 LSTM，不做推理")
    p.add_argument("--train_frac", type=float, default=0.8,
                   help="训练/评估时间切分比例（默认 0.8）")
    p.add_argument("--refresh", action="store_true", help="忽略 tf_resid 缓存，强制重算")
    p.add_argument("--diagnose_only", action="store_true",
                   help="跳过 TimesFM，仅训练 LSTM + 残差诊断（无 transformers 环境也可跑）")

    # TimesFM 残差上下文
    p.add_argument("--tf_ctx", type=int, default=CONTEXT_LEN,
                   help="TimesFM 对残差的上下文长度（默认与 LSTM context 一致 365；调小可减少前段有效区损失）")
    p.add_argument("--model_id", default=DEFAULT_MODEL_ID,
                   help="HuggingFace 基础模型 ID（zero-shot）")

    # LSTM 超参数
    p.add_argument("--hidden",      type=int,   default=128,  help="LSTM 隐层维度（默认 128）")
    p.add_argument("--n_layers",    type=int,   default=2,    help="LSTM 层数（默认 2）")
    p.add_argument("--epochs",      type=int,   default=20,   help="训练轮数（默认 20）")
    p.add_argument("--train_batch", type=int,   default=512,  help="训练批大小（默认 512）")
    p.add_argument("--lr",          type=float, default=1e-3, help="学习率（默认 1e-3）")
    p.add_argument("--stride",      type=int,   default=15,   help="训练窗口步长（默认 15）")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
