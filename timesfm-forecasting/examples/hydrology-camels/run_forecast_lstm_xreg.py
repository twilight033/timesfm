"""
TimesFM + LSTM-XReg 水文预测 —— CAMELS-US（timesfm + xreg 模式）

流程：
  1. 离线预计算：微调 TimesFM 对训练/评估序列滚动生成 horizon 步超前预测 → tf_preds[T]
  2. LSTM 训练目标 = 径流 - tf_preds（残差）；(动态强迫 ⊕ 静态属性) → 残差
  3. 推理：
       tf_hor     = tf_preds[i+ctx : i+ctx+hor]   （预计算，直接查表）
       lstm_resid = LSTM(ctx_dyn, hor_dyn, static)
       final      = tf_hor + lstm_resid

注意：若微调 TimesFM 与 LSTM 使用相同训练期，存在一定数据泄露风险；
     建议通过 --test_start 参数将评估窗口限定在时序后段。

用法：
  # 首次运行（训练 LSTM + 推理）：
  python run_forecast_lstm_xreg.py --finetune_method lora_r4

  # 只训练 LSTM：
  python run_forecast_lstm_xreg.py --train_only

  # 加载已有 LSTM 直接推理：
  python run_forecast_lstm_xreg.py --skip_train

  # 指定评估站点：
  python run_forecast_lstm_xreg.py --gauges 01013500 01022500
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from config import DATA_DIR, GAUGE_IDS
from lstm_xreg import (
    DYNAMIC_COLS,
    STATIC_ATTR_COLS,
    LSTMXRegPredictor,
    align_forcing_streamflow,
    load_camels_forcing,
    load_camels_static_attrs,
    train_lstm,
)


def load_camels_us(data_dir: Path, gauge_id: str) -> pd.Series:
    """加载 CAMELS-US 单站点日径流序列，返回按日期索引的 Series（单位 cfs）。"""
    files = list(data_dir.glob(f"**/{gauge_id}_streamflow_qc.txt"))
    if not files:
        raise FileNotFoundError(f"未找到站点 {gauge_id} 的径流文件，请检查 data_dir 路径。")
    df = pd.read_csv(
        files[0], sep=r"\s+", header=None,
        names=["gauge_id", "year", "month", "day", "streamflow", "qc"],
    )
    df["date"] = pd.to_datetime(df[["year", "month", "day"]])
    df["streamflow"] = pd.to_numeric(df["streamflow"], errors="coerce")
    df = df[df["streamflow"] >= 0].set_index("date")["streamflow"]
    return df.sort_index().astype(np.float32)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

CONTEXT_LEN = 365
HORIZON = 1
BATCH_SIZE = 32

STREAMFLOW_ROOT = (
    DATA_DIR / "camels_us" / "CAMELS_US"
    / "basin_timeseries_v1p2_metForcing_obsFlow"
    / "basin_dataset_public_v1p2" / "usgs_streamflow"
)

CKPT_DIR = Path(__file__).parent / "_lstm_ckpt"
OUTPUT_DIR = Path(__file__).parent / "output_lstm_xreg"
FINETUNE_OUT_ROOT = Path(__file__).parent / "output_finetune"

DEFAULT_MODEL_ID = "google/timesfm-2.5-200m-transformers"


# ---------------------------------------------------------------------------
# 微调 TimesFM 加载（transformers 格式）
# ---------------------------------------------------------------------------

def _load_finetuned_model(method: str, model_id: str, finetune_dir: Path, device: str):
    """加载微调后的 TimesFM（transformers 格式），返回处于 eval 状态的模型。"""
    from transformers import TimesFm2_5ModelForPrediction

    base = TimesFm2_5ModelForPrediction.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    base.eval()

    if method == "zero_shot":
        return base

    if method == "full":
        sd_path = finetune_dir / "model_state.pt"
        if sd_path.exists():
            sd = torch.load(sd_path, map_location=device)
            base.load_state_dict(sd, strict=False)
            logger.info("已加载 full 微调权重: %s", sd_path)
        else:
            logger.warning("未找到 %s，使用 base 权重", sd_path)
        return base

    if method.startswith("lora_r"):
        from peft import PeftModel
        if finetune_dir.exists() and any(finetune_dir.iterdir()):
            model = PeftModel.from_pretrained(base, str(finetune_dir))
            model.eval()
            return model
        logger.warning("未找到 LoRA adapter: %s，使用 base 权重", finetune_dir)
        return base

    raise ValueError(f"未知微调方式: {method}，可选: zero_shot, full, lora_r4, lora_r16")


# ---------------------------------------------------------------------------
# TimesFM 滚动预测预计算
# ---------------------------------------------------------------------------

@torch.no_grad()
def precompute_tf_forecasts(
    sf: np.ndarray,
    model_hf,
    context_len: int,
    horizon: int,
    batch_size: int,
    device: str,
) -> np.ndarray:
    """滚动生成微调 TimesFM 的超前预测，返回长度为 T 的 tf_preds 数组。

    tf_preds[t] = 以 sf[t-context_len:t] 为上下文预测 sf[t] 的值。
    t < context_len 的位置填 NaN（无有效上下文）。

    对于 horizon=1：每个起始位置 s 生成 tf_preds[s+context_len]，无重叠覆盖。
    对于 horizon>1 ：相邻窗口存在重叠，后序窗口（使用更近上下文）覆盖前序，属预期行为。
    """
    T = len(sf)
    tf_preds = np.full(T, np.nan, dtype=np.float32)

    starts = list(range(0, T - context_len - horizon + 1))
    if not starts:
        return tf_preds

    for i in range(0, len(starts), batch_size):
        batch_starts = starts[i : i + batch_size]
        ctx = np.stack(
            [sf[s : s + context_len] for s in batch_starts]
        ).astype(np.float32)
        ctx_t = torch.from_numpy(ctx).to(device)
        out = model_hf(past_values=ctx_t)
        # mean_predictions shape: [B, max_horizon] — 取前 horizon 步
        preds = out.mean_predictions[:, :horizon].float().cpu().numpy()
        for j, s in enumerate(batch_starts):
            tf_preds[s + context_len : s + context_len + horizon] = preds[j]

    return tf_preds


# ---------------------------------------------------------------------------
# 数据准备
# ---------------------------------------------------------------------------

def load_gauge(
    gauge_id: str,
    static_df: pd.DataFrame,
    model_hf,
    device: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """加载单站数据并预计算 TimesFM 滚动预测。

    Returns:
        (dyn [T, N_DYN], sf [T], static [N_STATIC], tf_preds [T]) 或 None
    """
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

    # 预计算 TimesFM 滚动预测
    n_windows = len(sf) - CONTEXT_LEN - HORIZON + 1
    logger.info("[%s] 预计算 TimesFM 滚动预测（%d 个窗口）...", gauge_id, n_windows)
    tf_preds = precompute_tf_forecasts(sf, model_hf, CONTEXT_LEN, HORIZON, BATCH_SIZE, device)

    return dyn, sf, static, tf_preds


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


# ---------------------------------------------------------------------------
# 单站推理（批量窗口）
# ---------------------------------------------------------------------------

def forecast_single_gauge(
    gauge_id: str,
    dyn: np.ndarray,       # [T, N_DYN]
    sf: np.ndarray,        # [T]
    static: np.ndarray,    # [N_STATIC]
    tf_preds: np.ndarray,  # [T] 预计算的 TimesFM 预测
    predictor: LSTMXRegPredictor,
) -> dict | None:
    """滑动窗口推理：微调 TimesFM + LSTM 残差，并与 TimesFM-only 基线对比。

    最终预测 = tf_hor + lstm_resid
      tf_hor    ：预计算的 TimesFM 预测（直接查表，无需重复调用模型）
      lstm_resid：LSTM 对残差的预测
    """
    n = len(sf)
    n_windows = n - CONTEXT_LEN - HORIZON + 1
    if n_windows <= 0:
        return None

    # 只保留 tf_preds 完整（无 NaN）的窗口
    valid_idx = [
        i for i in range(n_windows)
        if not np.isnan(tf_preds[i + CONTEXT_LEN : i + CONTEXT_LEN + HORIZON]).any()
    ]
    if not valid_idx:
        logger.warning("[%s] 无有效 tf_preds 窗口，跳过", gauge_id)
        return None

    n_valid = len(valid_idx)
    ctx_dyn = np.stack(
        [dyn[i : i + CONTEXT_LEN] for i in valid_idx]
    )  # [N, 365, N_DYN]
    hor_dyn = np.stack(
        [dyn[i + CONTEXT_LEN : i + CONTEXT_LEN + HORIZON] for i in valid_idx]
    )  # [N, HORIZON, N_DYN]
    ctx_sf = np.stack(
        [sf[i : i + CONTEXT_LEN] for i in valid_idx]
    )  # [N, 365]
    obs_hor = np.stack(
        [sf[i + CONTEXT_LEN : i + CONTEXT_LEN + HORIZON] for i in valid_idx]
    )  # [N, HORIZON]
    tf_hor = np.stack(
        [tf_preds[i + CONTEXT_LEN : i + CONTEXT_LEN + HORIZON] for i in valid_idx]
    )  # [N, HORIZON]

    static_batch = np.broadcast_to(static[None], (n_valid, len(static))).copy()

    # ---- LSTM 残差预测 ----
    lstm_resid = predictor.predict_residual_batch(
        ctx_dyn, hor_dyn, ctx_sf, static_batch
    )  # [N, HORIZON]

    # ---- 最终预测 = 微调 TimesFM + LSTM 残差 ----
    final = tf_hor + lstm_resid  # [N, HORIZON]

    obs_flat = obs_hor.ravel()
    return {
        "gauge_id":           gauge_id,
        "n_windows":          n_valid,
        "nse_tf_lstm_xreg":   nse(obs_flat, final.ravel()),
        "nse_tf_only":        nse(obs_flat, tf_hor.ravel()),
    }


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    eval_gauges  = args.gauges       or GAUGE_IDS
    train_gauges = args.train_gauges or eval_gauges

    # ---- 1. 静态属性 ----
    logger.info("加载 CAMELS 静态流域属性（27 列）...")
    static_df = load_camels_static_attrs(DATA_DIR)
    logger.info("静态属性已加载：%d 个站点 × %d 列", len(static_df), len(static_df.columns))

    # ---- 2. 微调 TimesFM（HF 格式，用于预计算） ----
    device = "cuda" if torch.cuda.is_available() else "cpu"
    finetune_dir = Path(args.finetune_dir) if args.finetune_dir else (
        FINETUNE_OUT_ROOT / args.finetune_method
    )
    logger.info("加载微调 TimesFM（%s）从 %s...", args.finetune_method, finetune_dir)
    model_hf = _load_finetuned_model(
        args.finetune_method, args.model_id, finetune_dir, device
    )
    logger.info("微调 TimesFM 加载完成。")

    # ---- 3. 预加载所有站点（训练 + 评估，去重） ----
    all_gauges = sorted(set(train_gauges) | set(eval_gauges))
    gauge_cache: dict[str, tuple] = {}
    for gid in all_gauges:
        result = load_gauge(gid, static_df, model_hf, device)
        if result is not None:
            gauge_cache[gid] = result

    # 训练数组（4 元组列表）
    train_arrays = [gauge_cache[gid] for gid in sorted(set(train_gauges)) if gid in gauge_cache]
    if not train_arrays:
        raise RuntimeError("没有可用的训练站点，请检查数据路径。")

    # ---- 4. 训练 or 加载 LSTM ----
    if not args.skip_train:
        logger.info("开始训练 LSTM（%d 个站点）...", len(train_arrays))
        model_lstm, dyn_mean, dyn_std, static_mean, static_std = train_lstm(
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
        predictor = LSTMXRegPredictor(
            model_lstm, dyn_mean, dyn_std, static_mean, static_std, device=device
        )
    else:
        logger.info("从 %s 加载已有 LSTM...", CKPT_DIR)
        predictor = LSTMXRegPredictor.from_checkpoint(CKPT_DIR, device=device)

    if args.train_only:
        logger.info("--train_only 模式，训练完成，退出。")
        return

    # ---- 5. 推理评估 ----
    results = []
    for gid in eval_gauges:
        if gid not in gauge_cache:
            logger.warning("[%s] 数据不可用，跳过", gid)
            continue
        logger.info("推理站点: %s", gid)
        dyn, sf, static, tf_preds = gauge_cache[gid]
        try:
            r = forecast_single_gauge(gid, dyn, sf, static, tf_preds, predictor)
            if r is not None:
                results.append(r)
                logger.info(
                    "  NSE  tf+lstm_xreg=%.4f  tf_only=%.4f  (n_win=%d)",
                    r["nse_tf_lstm_xreg"],
                    r["nse_tf_only"],
                    r["n_windows"],
                )
        except Exception as e:
            logger.warning("[%s] 推理失败: %s", gid, e)

    # ---- 6. 汇总 ----
    if not results:
        logger.warning("没有有效结果。")
        return

    summary_df = pd.DataFrame(results)
    out_path = OUTPUT_DIR / "nse_summary.csv"
    summary_df.to_csv(out_path, index=False, encoding="utf-8-sig")

    print(f"\n{'='*60}")
    print("  汇总结果")
    print(f"{'='*60}")
    print(summary_df.to_string(index=False))

    nse_cols = [c for c in summary_df.columns if c.startswith("nse_")]
    if len(results) > 1:
        print("\n  各方法中位数 NSE:")
        for col in nse_cols:
            vals = summary_df[col].dropna()
            if len(vals):
                print(f"    {col}: {vals.median():.4f}  (n={len(vals)})")

    print(f"\n  汇总已保存: {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="TimesFM + LSTM-XReg 水文预测（CAMELS-US，timesfm+xreg 模式）"
    )
    p.add_argument(
        "--gauges", nargs="+", default=None,
        help="评估站点列表（默认使用 config.py 中的 GAUGE_IDS）",
    )
    p.add_argument(
        "--train_gauges", nargs="+", default=None,
        help="训练 LSTM 的站点列表（默认与评估站点相同）",
    )
    p.add_argument("--skip_train", action="store_true", help="跳过训练，从 _lstm_ckpt 加载")
    p.add_argument("--train_only", action="store_true", help="只训练 LSTM，不做推理")

    # 微调 TimesFM 参数
    p.add_argument(
        "--finetune_method",
        default="lora_r4",
        choices=["zero_shot", "full", "lora_r4", "lora_r16"],
        help="微调方式，对应 output_finetune/{method}/ 目录（默认 lora_r4）",
    )
    p.add_argument(
        "--finetune_dir", default=None,
        help="微调权重目录（默认 output_finetune/{finetune_method}/）",
    )
    p.add_argument(
        "--model_id", default=DEFAULT_MODEL_ID,
        help="HuggingFace 基础模型 ID（默认 google/timesfm-2.5-200m-transformers）",
    )

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
