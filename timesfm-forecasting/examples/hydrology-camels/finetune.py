#!/usr/bin/env python3
"""
TimesFM 2.5 在 CAMELS-US + 辽河松花江上的微调，支持 4 种方式：

  - zero_shot   零样本基线，不训练
  - lora_r4     LoRA r=4（轻量，~0.6% 可训参数）
  - lora_r16    LoRA r=16（中等，~2% 可训参数）
  - full        全参微调（开梯度检查点以适配 12GB 显存）

训练数据：CAMELS-US（默认 200 站）+ 辽河松花江全部站，按"段内时间"切 80/20。
评估口径：辽河 val 段上的滑动窗口 NSE（沿用 nse_flood/dry/overall 三档）。

依赖：
    pip install transformers accelerate peft pandas pyarrow scikit-learn

单方法用法：
    python finetune.py --method lora_r4 --num_samples 50000 --epochs 5
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from finetune_data import (
    RandomWindowDataset,
    build_or_load_segments,
    filter_camels,
    filter_liaohe,
    flatten_segments,
    set_seed,
    split_by_time,
)
from run_forecast_liaohe import (
    CONTEXT_LEN as DEFAULT_CTX,
    HORIZON as DEFAULT_HORIZON,
    Segment,
    build_windows_from_segments,
    compute_nse_metrics,
    safe_filename,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

OUT_ROOT = Path(__file__).parent / "output_finetune"
DEFAULT_MODEL_ID = "google/timesfm-2.5-200m-transformers"


# ---------------------------------------------------------------------------
# 模型构建（不同微调方式）
# ---------------------------------------------------------------------------

def build_model(method: str, model_id: str, device: str):
    """根据 method 返回（可训练）模型，未匹配则报错。"""
    from transformers import TimesFm2_5ModelForPrediction

    model = TimesFm2_5ModelForPrediction.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )

    if method == "zero_shot":
        for p in model.parameters():
            p.requires_grad = False
        return model

    if method == "full":
        for p in model.parameters():
            p.requires_grad = True
        if hasattr(model, "gradient_checkpointing_enable"):
            try:
                model.gradient_checkpointing_enable()
                logger.info("已启用 gradient_checkpointing")
            except Exception as e:
                logger.warning("gradient_checkpointing 启用失败: %s", e)
        return model

    if method.startswith("lora_r"):
        from peft import LoraConfig, get_peft_model
        r = int(method.split("lora_r", 1)[1])
        cfg = LoraConfig(
            r=r,
            lora_alpha=r * 2,
            target_modules="all-linear",
            lora_dropout=0.05,
            bias="none",
        )
        model = get_peft_model(model, cfg)
        model.print_trainable_parameters()
        return model

    raise ValueError(f"未知 method: {method}")


def reload_for_eval(method: str, model_id: str, ckpt_dir: Path, device: str):
    """评估前重新加载（保证用最佳 checkpoint）。"""
    from transformers import TimesFm2_5ModelForPrediction

    base = TimesFm2_5ModelForPrediction.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map=device
    )

    if method == "zero_shot":
        return base

    if method == "full":
        sd_path = ckpt_dir / "model_state.pt"
        if sd_path.exists():
            sd = torch.load(sd_path, map_location=device)
            base.load_state_dict(sd, strict=False)
            logger.info("已加载 full 微调权重: %s", sd_path)
        else:
            logger.warning("未找到 %s，使用 base 权重", sd_path)
        return base

    if method.startswith("lora_r"):
        from peft import PeftModel
        adapter = ckpt_dir
        if adapter.exists() and any(adapter.iterdir()):
            return PeftModel.from_pretrained(base, adapter)
        logger.warning("未找到 LoRA adapter: %s，使用 base", adapter)
        return base

    raise ValueError(method)


def save_ckpt(model, method: str, ckpt_dir: Path) -> None:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    if method.startswith("lora_r"):
        model.save_pretrained(ckpt_dir)
    elif method == "full":
        torch.save(model.state_dict(), ckpt_dir / "model_state.pt")


# ---------------------------------------------------------------------------
# 训练
# ---------------------------------------------------------------------------

def train_loop(args, model, train_ds, val_ds, ckpt_dir: Path) -> dict:
    device = next(model.parameters()).device
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True,
        num_workers=args.num_workers, pin_memory=True,
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, num_workers=0)

    trainable = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)
    total_steps = max(1, args.epochs * max(1, len(train_loader)))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=total_steps)

    best_val = float("inf")
    history: list[dict] = []
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        ep_loss = 0.0
        n_batch = 0
        for ctx, tgt in train_loader:
            ctx = ctx.to(device)
            tgt = tgt.to(device)
            outputs = model(
                past_values=ctx,
                future_values=tgt,
                forecast_context_len=args.context_len,
            )
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
            optim.step()
            optim.zero_grad()
            sched.step()
            ep_loss += loss.item()
            n_batch += 1
        avg_train = ep_loss / max(n_batch, 1)

        model.eval()
        v_loss = 0.0
        v_n = 0
        with torch.no_grad():
            for ctx, tgt in val_loader:
                ctx = ctx.to(device)
                tgt = tgt.to(device)
                outputs = model(
                    past_values=ctx,
                    future_values=tgt,
                    forecast_context_len=args.context_len,
                )
                v_loss += outputs.loss.item()
                v_n += 1
        avg_val = v_loss / max(v_n, 1)
        elapsed = time.time() - t0
        logger.info(
            "[%s] epoch %d/%d  steps=%d  train=%.4f  val=%.4f  elapsed=%.1fs",
            args.method, epoch, args.epochs, n_batch, avg_train, avg_val, elapsed,
        )
        history.append({"epoch": epoch, "train_loss": avg_train, "val_loss": avg_val})
        if avg_val < best_val:
            best_val = avg_val
            save_ckpt(model, args.method, ckpt_dir)
            logger.info("  ✓ 保存最佳 ckpt → %s", ckpt_dir)

    return {
        "best_val_loss": best_val,
        "train_minutes": (time.time() - t0) / 60,
        "history": history,
    }


# ---------------------------------------------------------------------------
# 在辽河 val 段上做窗口预测 → NSE
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_nse(
    model,
    liaohe_val_segments: dict[str, list[Segment]],
    device: str,
    context_len: int,
    horizon: int,
    batch_size: int,
    save_dir: Path,
    method_tag: str,
) -> pd.DataFrame:
    save_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    model.eval()
    for station, segs in liaohe_val_segments.items():
        bundle = build_windows_from_segments(segs, context_len, horizon)
        if bundle is None:
            continue
        n_win = len(bundle["contexts"])
        all_preds = []
        for i in range(0, n_win, batch_size):
            chunk = bundle["contexts"][i : i + batch_size]
            ctx = torch.from_numpy(np.stack(chunk).astype(np.float32)).to(device)
            out = model(past_values=ctx)
            preds = out.mean_predictions[:, :horizon].float().cpu().numpy()
            all_preds.append(preds)
        preds = np.concatenate(all_preds, axis=0)[:n_win]

        m = compute_nse_metrics(
            bundle["targets"],
            preds,
            bundle["target_observed"],
            bundle["target_in_flood"],
        )
        rows.append({
            "station": station.replace("liaohe_", "").replace("camels_", ""),
            "n_windows": n_win,
            **m,
        })
    df = pd.DataFrame(rows)
    df.to_csv(
        save_dir / f"per_station_{method_tag}.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return df


def summarize(df: pd.DataFrame, method: str) -> dict:
    if df.empty:
        return {"method": method, "n_stations": 0}
    return {
        "method": method,
        "n_stations": int(len(df)),
        "n_windows": int(df["n_windows"].sum()),
        "nse_flood_med": float(df["nse_flood"].dropna().median())
            if df["nse_flood"].dropna().size else float("nan"),
        "nse_dry_med": float(df["nse_dry"].dropna().median())
            if df["nse_dry"].dropna().size else float("nan"),
        "nse_overall_med": float(df["nse_overall"].dropna().median())
            if df["nse_overall"].dropna().size else float("nan"),
    }


# ---------------------------------------------------------------------------
# 单方法主入口
# ---------------------------------------------------------------------------

def run_one_method(args) -> dict:
    set_seed(args.seed)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    # ---- 数据 ----
    seg_all = build_or_load_segments(
        max_camels_basins=args.max_camels_basins,
        seed=args.seed,
        refresh=args.refresh_cache,
    )
    train_segs_d, val_segs_d = split_by_time(seg_all, train_ratio=args.train_ratio)
    train_segs = flatten_segments(train_segs_d)
    val_segs_for_loss = flatten_segments(val_segs_d)
    if args.eval_domain == "camels":
        eval_val_d = filter_camels(val_segs_d)
    else:
        eval_val_d = filter_liaohe(val_segs_d)
    logger.info(
        "训练段 %d, 验证段 %d, eval_domain=%s, eval 站 %d",
        len(train_segs), len(val_segs_for_loss), args.eval_domain, len(eval_val_d),
    )

    method_dir = OUT_ROOT / args.method
    method_dir.mkdir(parents=True, exist_ok=True)
    if args.eval_domain == "liaohe":
        metrics_path = method_dir / "metrics.json"
        eval_method_tag = args.method
    else:
        metrics_path = method_dir / f"metrics_{args.eval_domain}.json"
        eval_method_tag = f"{args.method}_{args.eval_domain}"
    train_metrics: dict = {}

    # ---- 训练 ----
    if args.method != "zero_shot" and not args.eval_only:
        train_ds = RandomWindowDataset(
            train_segs,
            context_len=args.context_len,
            horizon=args.horizon,
            num_samples=args.num_samples,
            seed=args.seed,
        )
        val_ds = RandomWindowDataset(
            val_segs_for_loss,
            context_len=args.context_len,
            horizon=args.horizon,
            num_samples=max(args.num_samples // 10, 500),
            seed=args.seed + 1,
        )
        logger.info("train windows %d, val windows %d", len(train_ds), len(val_ds))

        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("device=%s", device)
        model = build_model(args.method, args.model_id, device)
        train_metrics = train_loop(args, model, train_ds, val_ds, method_dir)
        del model
        torch.cuda.empty_cache()

    # ---- 评估 ----
    device = "cuda" if torch.cuda.is_available() else "cpu"
    eval_model = reload_for_eval(args.method, args.model_id, method_dir, device)
    df = evaluate_nse(
        eval_model,
        eval_val_d,
        device=device,
        context_len=args.context_len,
        horizon=args.horizon,
        batch_size=args.batch_size,
        save_dir=method_dir,
        method_tag=eval_method_tag,
    )
    summary = summarize(df, args.method)
    summary["eval_domain"] = args.eval_domain
    full = {**summary, **train_metrics, "args": vars(args)}
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(full, f, ensure_ascii=False, indent=2, default=str)
    logger.info("method=%s 完成: %s", args.method, summary)
    del eval_model
    torch.cuda.empty_cache()
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="TimesFM 2.5 微调（水文）")
    p.add_argument(
        "--method",
        choices=["zero_shot", "lora_r4", "lora_r16", "full"],
        default="lora_r4",
    )
    p.add_argument("--model_id", default=DEFAULT_MODEL_ID)
    p.add_argument("--context_len", type=int, default=DEFAULT_CTX)
    p.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    p.add_argument("--max_camels_basins", type=int, default=200,
                   help="使用的 CAMELS-US 站点上限；设 -1 表示全部")
    p.add_argument("--train_ratio", type=float, default=0.8)
    p.add_argument("--num_samples", type=int, default=50000)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--refresh_cache", action="store_true")
    p.add_argument("--eval_only", action="store_true")
    p.add_argument(
        "--eval_domain",
        choices=["liaohe", "camels"],
        default="liaohe",
        help="评估流域：liaohe（默认，辽河 val 段）或 camels（CAMELS-US val 段）",
    )
    args = p.parse_args()
    if args.max_camels_basins is not None and args.max_camels_basins < 0:
        args.max_camels_basins = None
    # full 微调降 batch + 降 lr 防爆
    if args.method == "full" and args.batch_size > 8:
        logger.warning("full 微调建议 batch_size<=8（12GB 显存），自动降到 8")
        args.batch_size = 8
    return args


def main() -> None:
    args = parse_args()
    run_one_method(args)


if __name__ == "__main__":
    main()
