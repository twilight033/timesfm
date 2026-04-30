#!/usr/bin/env python3
"""
一键运行多种微调方式 + 输出对比表。

默认顺序跑：zero_shot → lora_r4 → lora_r16 → full
然后在 output_finetune/comparison.csv 输出辽河 val NSE 中位数对比。

用法:
    python run_finetune_compare.py
    python run_finetune_compare.py --methods zero_shot lora_r4 full
    python run_finetune_compare.py --num_samples 20000 --epochs 3   # 快速试验
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
    """从基线 args 派生单方法 args。"""
    a = deepcopy(base_args)
    a.method = method
    if method == "full" and a.batch_size > 8:
        a.batch_size = 8
    return a


def main() -> None:
    p = argparse.ArgumentParser(description="一键多方法微调对比（水文）")
    p.add_argument(
        "--methods", nargs="+", default=ALL_METHODS,
        choices=ALL_METHODS,
        help="要跑的方法列表（顺序即执行顺序）",
    )
    p.add_argument("--model_id", default="google/timesfm-2.5-200m-transformers")
    p.add_argument("--context_len", type=int, default=365)
    p.add_argument("--horizon", type=int, default=1)
    p.add_argument("--max_camels_basins", type=int, default=200)
    p.add_argument("--train_ratio", type=float, default=0.8)
    p.add_argument("--num_samples", type=int, default=50000)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--refresh_cache", action="store_true")
    p.add_argument("--eval_only", action="store_true",
                   help="只跑评估（用各方法已有 ckpt）")
    cli = p.parse_args()
    if cli.max_camels_basins is not None and cli.max_camels_basins < 0:
        cli.max_camels_basins = None

    # 预先把数据缓存好（避免每个 method 重复构建）
    print("=" * 60)
    print("  预构建数据缓存")
    print("=" * 60)
    build_or_load_segments(
        max_camels_basins=cli.max_camels_basins,
        seed=cli.seed,
        refresh=cli.refresh_cache,
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
    print("  最终对比")
    print("=" * 60)
    df = pd.DataFrame(summaries)
    cols_order = [
        "method", "status",
        "n_stations", "n_windows",
        "nse_flood_med", "nse_dry_med", "nse_overall_med",
        "elapsed_min",
    ]
    df = df[[c for c in cols_order if c in df.columns]]
    print(df.to_string(index=False))

    out_csv = OUT_ROOT / "comparison.csv"
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    out_json = OUT_ROOT / "comparison.json"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(summaries, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n  对比表已保存: {out_csv}")
    print(f"  详细 JSON  : {out_json}")


if __name__ == "__main__":
    main()
