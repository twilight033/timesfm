#!/usr/bin/env python3
"""
仅使用 CAMELS-US 数据微调 TimesFM 2.5，自动依次运行：
  zero_shot → lora_r4 → lora_r16 → full

训练数据：仅 CAMELS-US（不含辽河），切段时不区分汛期/非汛期，缺测统一前向填充
评估数据：CAMELS-US val 段，主要看 nse_overall_med

用法：
    python run_finetune_camels.py
    python run_finetune_camels.py --methods zero_shot lora_r4          # 只跑部分
    python run_finetune_camels.py --num_samples 20000 --epochs 3       # 快速试验
    python run_finetune_camels.py --max_camels_basins 100              # 只用 100 站
    python run_finetune_camels.py --max_ffill 14                       # gap 超过 14 天才切段
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from copy import deepcopy
from pathlib import Path

import pandas as pd

from finetune import OUT_ROOT, run_one_method
from finetune_data import build_or_load_segments

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ALL_METHODS = ["zero_shot", "lora_r4", "lora_r16", "full"]


def make_method_args(base_args, method: str):
    a = deepcopy(base_args)
    a.method = method
    if method == "full" and a.batch_size > 8:
        a.batch_size = 8
    return a


def main() -> None:
    p = argparse.ArgumentParser(description="仅 CAMELS-US 数据微调 TimesFM 2.5")
    p.add_argument(
        "--methods", nargs="+", default=ALL_METHODS,
        choices=ALL_METHODS,
        help="要跑的方法列表（按顺序执行）",
    )
    p.add_argument("--model_id", default="google/timesfm-2.5-200m-transformers")
    p.add_argument("--context_len", type=int, default=384)
    p.add_argument("--horizon", type=int, default=1)
    p.add_argument("--max_camels_basins", type=int, default=200,
                   help="使用的 CAMELS-US 站点数，-1 表示全部")
    p.add_argument("--train_ratio", type=float, default=0.8)
    p.add_argument("--num_samples", type=int, default=50000)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--refresh_cache", action="store_true",
                   help="强制重建数据缓存")
    p.add_argument("--eval_only", action="store_true",
                   help="只跑评估（使用各方法已有 checkpoint）")
    p.add_argument("--max_ffill", type=int, default=7,
                   help="CAMELS 缺测前向填充的最大天数（超过则切段）")
    cli = p.parse_args()
    if cli.max_camels_basins is not None and cli.max_camels_basins < 0:
        cli.max_camels_basins = None

    # 预缓存数据（只加载 CAMELS，然后过滤）
    print("=" * 60)
    print("  预构建数据缓存（CAMELS-only）")
    print("=" * 60)
    build_or_load_segments(
        max_camels_basins=cli.max_camels_basins,
        seed=cli.seed,
        refresh=cli.refresh_cache,
        max_ffill=cli.max_ffill,
    )

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    summaries = []
    t0 = time.time()
    for i, method in enumerate(cli.methods, 1):
        print()
        print("#" * 60)
        print(f"# [{i}/{len(cli.methods)}] method = {method}")
        print("#" * 60)
        args = make_method_args(cli, method)
        # 关键：训练与评估均限定为 CAMELS
        args.train_domain = "camels"
        args.eval_domain = "camels"
        try:
            summary = run_one_method(args)
            summary["status"] = "ok"
        except Exception as e:
            logger.exception("method=%s 失败: %s", method, e)
            summary = {"method": method, "status": f"error: {e}"}
        summary["elapsed_min"] = round((time.time() - t0) / 60, 2)
        summaries.append(summary)

    print()
    print("=" * 60)
    print("  最终对比（CAMELS-US val）")
    print("=" * 60)
    df = pd.DataFrame(summaries)
    cols_order = [
        "method", "status",
        "n_stations", "n_windows",
        "nse_overall_med",
        "elapsed_min",
    ]
    df = df[[c for c in cols_order if c in df.columns]]
    print(df.to_string(index=False))

    out_csv = OUT_ROOT / "comparison_camels.csv"
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    out_json = OUT_ROOT / "comparison_camels.json"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(summaries, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n  对比表已保存: {out_csv}")
    print(f"  详细 JSON  : {out_json}")


if __name__ == "__main__":
    main()
