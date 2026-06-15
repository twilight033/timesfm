"""
LSTM-First（反向 cascade）：LSTM 先预测径流，TimesFM 再预测残差。

与 lstm_xreg.py（前向 cascade）的对照：
  前向：微调 TimesFM 出基线 → LSTM 学残差 → final = tf_hor + lstm_resid
  反向：LSTM 预测径流      → TimesFM 学残差 → final = lstm_pred + tf_resid

关键约束：LSTM 输入与前向版**完全一致**（动态强迫 ⊕ 静态属性，不含历史径流真值），
仅把预测目标从"残差"换成"径流本身"，保证两种方法的 LSTM 输入对等、对比干净。
因为 LSTM 没消费径流自相关，其残差里仍保留时序结构，留给 TimesFM 去抓。

训练目标：(动态强迫 + 静态属性) → 径流（按窗口归一化）
推理：滚动生成 lstm_pred[t]，下游用 residual = streamflow - lstm_pred 喂 TimesFM。

直接复用 lstm_xreg 的数据加载、模型结构与归一化统计逻辑。
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split

from lstm_xreg import (  # noqa: F401  —— 数据加载工具供下游脚本统一从本模块导入
    DYNAMIC_COLS,
    N_DYNAMIC,
    N_FEATURES,
    N_STATIC,
    STATIC_ATTR_COLS,
    HydroLSTM,
    align_forcing_streamflow,
    compute_stats,
    load_camels_forcing,
    load_camels_static_attrs,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 训练数据集（预测径流本身，非残差）
# ---------------------------------------------------------------------------

class _StreamflowWindowDataset(Dataset):
    """单站滑动窗口数据集（lstm-first 模式）。

    与 lstm_xreg._WindowDataset 的差异：
      - 不依赖 tf_preds（TimesFM 预计算）
      - 解码器主标签是径流本身（按窗口归一化），而非残差

    样本构成：
      ctx_feat [T_ctx, N_FEATURES]  归一化动态 ⊕ 静态
      hor_feat [T_hor, N_FEATURES]  同上（预报期）
      ctx_sf_n [T_ctx]              按窗口归一化的径流（编码器辅助损失）
      hor_sf_n [T_hor]              按窗口归一化的径流（解码器主损失）
    """

    def __init__(
        self,
        dyn: np.ndarray,         # [T, N_DYN]
        sf: np.ndarray,          # [T]
        static_n: np.ndarray,    # [N_STATIC] 已归一化
        context_len: int,
        horizon: int,
        dyn_mean: np.ndarray,
        dyn_std: np.ndarray,
        stride: int = 1,
    ):
        self.dyn = dyn
        self.sf = sf
        self.static_n = static_n
        self.ctx = context_len
        self.hor = horizon
        self.dyn_mean = dyn_mean
        self.dyn_std = dyn_std
        n = len(sf)
        self.indices = list(range(0, n - context_len - horizon + 1, stride))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        s = self.indices[idx]
        ctx_dyn = self.dyn[s : s + self.ctx]
        hor_dyn = self.dyn[s + self.ctx : s + self.ctx + self.hor]
        ctx_sf  = self.sf[s : s + self.ctx]
        hor_sf  = self.sf[s + self.ctx : s + self.ctx + self.hor]

        # 动态特征归一化 + 静态拼接
        ctx_dyn_n = (ctx_dyn - self.dyn_mean) / self.dyn_std
        hor_dyn_n = (hor_dyn - self.dyn_mean) / self.dyn_std
        static_ctx = np.broadcast_to(self.static_n, (self.ctx, len(self.static_n)))
        static_hor = np.broadcast_to(self.static_n, (self.hor, len(self.static_n)))
        ctx_feat = np.concatenate([ctx_dyn_n, static_ctx], axis=1).astype(np.float32)
        hor_feat = np.concatenate([hor_dyn_n, static_hor], axis=1).astype(np.float32)

        # 径流按上下文归一化（编码器辅助 + 解码器主损失共用同一组统计量）
        sf_mean = float(ctx_sf.mean())
        sf_std  = float(max(ctx_sf.std(), 1e-6))
        ctx_sf_n = ((ctx_sf - sf_mean) / sf_std).astype(np.float32)
        hor_sf_n = ((hor_sf - sf_mean) / sf_std).astype(np.float32)

        return (
            torch.from_numpy(ctx_feat),
            torch.from_numpy(hor_feat),
            torch.from_numpy(ctx_sf_n),
            torch.from_numpy(hor_sf_n),
            torch.tensor([sf_mean], dtype=torch.float32),
            torch.tensor([sf_std], dtype=torch.float32),
        )


# ---------------------------------------------------------------------------
# 训练
# ---------------------------------------------------------------------------

def train_lstm_streamflow(
    train_arrays: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    # 每个元素：(dyn [T,N_DYN], sf [T], static [N_STATIC])，已切到训练期
    context_len: int,
    horizon: int,
    val_arrays: list[tuple[np.ndarray, np.ndarray, np.ndarray]] | None = None,
    # 独立验证期窗口（按时间切分得到）；为 None 时回退到从训练集随机划分
    hidden: int = 128,
    n_layers: int = 2,
    dropout: float = 0.1,
    epochs: int = 20,
    batch_size: int = 512,
    lr: float = 1e-3,
    val_ratio: float = 0.15,
    stride: int = 1,
    aux_weight: float = 0.2,   # 编码器辅助损失权重（预测 ctx 径流）
    device: str | None = None,
    save_dir: Path | None = None,
) -> tuple[HydroLSTM, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """训练 HydroLSTM（径流模式），返回 (模型, dyn_mean, dyn_std, static_mean, static_std)。"""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("训练设备: %s，共 %d 个站点", device, len(train_arrays))

    # compute_stats 只用元组的 g[0]=dyn 与 g[2]=static，3 元组兼容；只用训练段，避免泄露
    dyn_mean, dyn_std, static_mean, static_std = compute_stats(train_arrays)

    def _build_concat(arrays):
        """按训练段统计量归一化静态，逐站建滑窗数据集并拼接（无有效窗口则返回 None）。"""
        dsets = []
        for dyn, sf, static_raw in arrays:
            if len(sf) < context_len + horizon:
                continue
            static_n = ((static_raw - static_mean) / static_std).astype(np.float32)
            ds = _StreamflowWindowDataset(
                dyn, sf, static_n,
                context_len, horizon, dyn_mean, dyn_std, stride=stride,
            )
            if len(ds) > 0:
                dsets.append(ds)
        return ConcatDataset(dsets) if dsets else None

    train_full = _build_concat(train_arrays)
    if train_full is None:
        raise RuntimeError("没有可用的训练窗口，请检查数据。")

    val_full = _build_concat(val_arrays) if val_arrays is not None else None
    if val_full is not None:
        # 独立验证期：验证窗口目标落在训练期之后，与训练窗口时间不重叠，
        # 故 stride=1（全窗口）也不会有自相关泄露。
        train_ds, val_ds = train_full, val_full
        logger.info("时间切分：训练窗口 %d，验证窗口 %d（独立验证期）",
                    len(train_ds), len(val_ds))
    else:
        # 兜底：从训练集随机划分（旧行为，存在自相关泄露，仅在无独立验证期时使用）
        if val_arrays is not None:
            logger.warning("验证期无可用窗口，回退到从训练集随机划分验证。")
        n_val = max(1, int(len(train_full) * val_ratio))
        train_ds, val_ds = random_split(
            train_full, [len(train_full) - n_val, n_val],
            generator=torch.Generator().manual_seed(42),
        )
        logger.info("随机划分验证集：训练窗口 %d，验证窗口 %d", len(train_ds), len(val_ds))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=(device != "cpu"))
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=0)

    model = HydroLSTM(n_features=N_FEATURES, hidden=hidden,
                      n_layers=n_layers, dropout=dropout)
    model.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    total_steps = epochs * max(1, len(train_loader))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps)

    best_val, best_state = float("inf"), None

    for epoch in range(1, epochs + 1):
        model.train()
        tr_loss, tr_n = 0.0, 0
        for ctx_feat, hor_feat, ctx_sf_n, hor_sf_n, _, _ in train_loader:
            ctx_feat = ctx_feat.to(device)
            hor_feat = hor_feat.to(device)
            ctx_sf_n = ctx_sf_n.to(device)
            hor_sf_n = hor_sf_n.to(device)

            ctx_pred, hor_pred = model(ctx_feat, hor_feat)
            # 主损失：解码器预测预报期径流
            loss_main = nn.functional.mse_loss(hor_pred, hor_sf_n)
            # 辅助损失：编码器预测上下文径流（帮助建立有效隐状态）
            loss_aux  = nn.functional.mse_loss(ctx_pred, ctx_sf_n)
            loss = loss_main + aux_weight * loss_aux

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tr_loss += loss_main.item()   # 只记录主损失
            tr_n += 1

        model.eval()
        vl, vn = 0.0, 0
        with torch.no_grad():
            for ctx_feat, hor_feat, _, hor_sf_n, _, _ in val_loader:
                ctx_feat = ctx_feat.to(device)
                hor_feat = hor_feat.to(device)
                hor_sf_n = hor_sf_n.to(device)
                _, hp = model(ctx_feat, hor_feat)
                vl += nn.functional.mse_loss(hp, hor_sf_n).item()
                vn += 1

        avg_tr = tr_loss / max(tr_n, 1)
        avg_vl = vl / max(vn, 1)
        logger.info("epoch %d/%d  train_main=%.4f  val_main=%.4f",
                    epoch, epochs, avg_tr, avg_vl)
        if avg_vl < best_val:
            best_val = avg_vl
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            logger.info("  ✓ 更新最佳 val=%.4f", best_val)

    if best_state:
        model.load_state_dict(best_state)

    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(best_state or model.state_dict(), save_dir / "hydro_lstm.pt")
        np.save(save_dir / "dyn_mean.npy", dyn_mean)
        np.save(save_dir / "dyn_std.npy", dyn_std)
        np.save(save_dir / "static_mean.npy", static_mean)
        np.save(save_dir / "static_std.npy", static_std)
        meta = {
            "hidden": hidden, "n_layers": n_layers, "dropout": dropout,
            "n_features": N_FEATURES, "n_dynamic": N_DYNAMIC, "n_static": N_STATIC,
            "context_len": context_len, "horizon": horizon,
            "mode": "streamflow",
        }
        with open(save_dir / "lstm_meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        logger.info("LSTM 已保存到 %s", save_dir)

    return model, dyn_mean, dyn_std, static_mean, static_std


# ---------------------------------------------------------------------------
# 推理封装
# ---------------------------------------------------------------------------

class LSTMStreamflowPredictor:
    """lstm-first 批量推理：返回径流预测（原始尺度）。"""

    def __init__(
        self,
        model: HydroLSTM,
        dyn_mean: np.ndarray,
        dyn_std: np.ndarray,
        static_mean: np.ndarray,
        static_std: np.ndarray,
        device: str = "cpu",
    ):
        self.model = model.to(device)
        self.model.eval()
        self.dyn_mean    = dyn_mean.astype(np.float32)
        self.dyn_std     = dyn_std.astype(np.float32)
        self.static_mean = static_mean.astype(np.float32)
        self.static_std  = static_std.astype(np.float32)
        self.device = device

    @classmethod
    def from_checkpoint(cls, save_dir: Path, device: str = "cpu") -> "LSTMStreamflowPredictor":
        with open(save_dir / "lstm_meta.json", encoding="utf-8") as f:
            meta = json.load(f)
        model = HydroLSTM(n_features=meta["n_features"], hidden=meta["hidden"],
                          n_layers=meta["n_layers"], dropout=meta["dropout"])
        model.load_state_dict(torch.load(save_dir / "hydro_lstm.pt", map_location="cpu"))
        return cls(
            model,
            dyn_mean    = np.load(save_dir / "dyn_mean.npy"),
            dyn_std     = np.load(save_dir / "dyn_std.npy"),
            static_mean = np.load(save_dir / "static_mean.npy"),
            static_std  = np.load(save_dir / "static_std.npy"),
            device=device,
        )

    def _build_feat(self, dyn: np.ndarray, static: np.ndarray) -> np.ndarray:
        """归一化动态特征 + 拼接静态属性，返回 [N, T, N_FEATURES]。"""
        dyn_n = (dyn - self.dyn_mean) / self.dyn_std
        static_n = (static - self.static_mean) / self.static_std
        T = dyn.shape[1]
        static_exp = np.broadcast_to(
            static_n[:, None, :], (static_n.shape[0], T, static_n.shape[1])
        ).copy()
        return np.concatenate([dyn_n, static_exp], axis=2).astype(np.float32)

    @torch.no_grad()
    def predict_streamflow_batch(
        self,
        ctx_dyn: np.ndarray,    # [N, T_ctx, N_DYN]
        hor_dyn: np.ndarray,    # [N, T_hor, N_DYN]
        ctx_sf: np.ndarray,     # [N, T_ctx] 用于计算窗口归一化统计量（反归一化用）
        static: np.ndarray,     # [N, N_STATIC] 原始尺度
        batch_size: int = 256,
    ) -> np.ndarray:
        """批量预测径流，返回 [N, T_hor]（原始径流尺度）。"""
        ctx_feat = self._build_feat(ctx_dyn, static)   # [N, T_ctx, N_FEATURES]
        hor_feat = self._build_feat(hor_dyn, static)   # [N, T_hor, N_FEATURES]

        # 与训练时一致：径流按上下文归一化，反归一化要乘 sf_std 并加回 sf_mean
        sf_mean = ctx_sf.mean(axis=1, keepdims=True)                       # [N, 1]
        sf_std  = np.maximum(ctx_sf.std(axis=1, keepdims=True), 1e-6)      # [N, 1]

        all_hor = []
        for i in range(0, len(ctx_feat), batch_size):
            ctx_t = torch.from_numpy(ctx_feat[i : i + batch_size]).to(self.device)
            hor_t = torch.from_numpy(hor_feat[i : i + batch_size]).to(self.device)
            _, hp_n = self.model(ctx_t, hor_t)
            all_hor.append(hp_n.cpu().numpy())

        hor_pred_n = np.concatenate(all_hor, axis=0)   # [N, T_hor]
        # 反归一化：径流 = pred_n * sf_std + sf_mean（径流非 0 均值，必须加回均值）
        return hor_pred_n * sf_std + sf_mean


# ---------------------------------------------------------------------------
# 全序列滚动预测（生成 lstm_pred，供下游算残差）
# ---------------------------------------------------------------------------

def rolling_predict_full(
    predictor: LSTMStreamflowPredictor,
    dyn: np.ndarray,        # [T, N_DYN]
    sf: np.ndarray,         # [T]
    static: np.ndarray,     # [N_STATIC]
    context_len: int,
    horizon: int = 1,
    batch_size: int = 256,
) -> np.ndarray:
    """对整条序列滚动 LSTM 预测，返回长度 T 的 lstm_pred。

    lstm_pred[t] = 以 dyn[t-context_len:t]（强迫⊕静态）为上下文预测 sf[t]。
    t < context_len 的位置填 NaN（无有效上下文）。

    horizon>1 时与 precompute_tf_forecasts 行为一致：相邻窗口重叠，
    后序窗口（更近上下文）覆盖前序，属预期。
    """
    T = len(sf)
    lstm_pred = np.full(T, np.nan, dtype=np.float32)

    starts = list(range(0, T - context_len - horizon + 1))
    if not starts:
        return lstm_pred

    ctx_dyn = np.stack([dyn[s : s + context_len] for s in starts]).astype(np.float32)
    hor_dyn = np.stack(
        [dyn[s + context_len : s + context_len + horizon] for s in starts]
    ).astype(np.float32)
    ctx_sf = np.stack([sf[s : s + context_len] for s in starts]).astype(np.float32)
    static_batch = np.broadcast_to(static[None], (len(starts), len(static))).copy()

    preds = predictor.predict_streamflow_batch(
        ctx_dyn, hor_dyn, ctx_sf, static_batch, batch_size=batch_size
    )  # [N, horizon]

    for j, s in enumerate(starts):
        lstm_pred[s + context_len : s + context_len + horizon] = preds[j]

    return lstm_pred
