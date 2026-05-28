"""
LSTM-XReg（timesfm + xreg 模式）：
  LSTM 学习预测微调后 TimesFM 无法解释的残差。

训练流程：
  1. 离线预计算：微调 TimesFM 对每个训练窗口生成 1 步超前预测 tf_pred
  2. 残差标签：residual_hor = streamflow_hor - tf_pred
  3. LSTM 训练目标：(动态强迫 + 静态属性) → residual_hor

推理流程（timesfm + xreg）：
  1. tf_hor  = 微调 TimesFM(streamflow_ctx)
  2. lstm_resid = LSTM(forcing_ctx, forcing_hor, static)
  3. final  = tf_hor + lstm_resid

动态强迫（Daymet 3 列）：prcp, tmax, tmin
静态属性（27 列，参照 Kratzert et al. 2019）：
  气候×9, 地形×3, 植被×5, 土壤×8, 地质×2
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 动态特征（Daymet 3 列）
# ---------------------------------------------------------------------------

_DYN_COL_MAP = {
    "prcp(mm/day)": "prcp",
    "tmax(C)":      "tmax",
    "tmin(C)":      "tmin",
}
DYNAMIC_COLS = list(_DYN_COL_MAP.values())
N_DYNAMIC = len(DYNAMIC_COLS)

FORCING_ROOT_REL = (
    "camels_us/CAMELS_US/basin_timeseries_v1p2_metForcing_obsFlow"
    "/basin_dataset_public_v1p2/basin_mean_forcing/daymet"
)

# ---------------------------------------------------------------------------
# 静态属性（27 列，参照论文）
# ---------------------------------------------------------------------------

_STATIC_FILE_COLS: dict[str, dict[str, str]] = {
    "camels_clim.txt": {
        "p_mean":         "p_mean",
        "pet_mean":       "pet_mean",
        "aridity":        "aridity",
        "p_seasonality":  "p_seasonality",
        "frac_snow":      "frac_snow_daily",
        "high_prec_freq": "high_prec_freq",
        "high_prec_dur":  "high_prec_dur",
        "low_prec_freq":  "low_prec_freq",
        "low_prec_dur":   "low_prec_dur",
    },
    "camels_topo.txt": {
        "elev_mean":   "elev_mean",
        "slope_mean":  "slope_mean",
        "area_gages2": "area_gages2",
    },
    "camels_vege.txt": {
        "frac_forest": "forest_frac",
        "lai_max":     "lai_max",
        "lai_diff":    "lai_diff",
        "gvf_max":     "gvf_max",
        "gvf_diff":    "gvf_diff",
    },
    "camels_soil.txt": {
        "soil_depth_pelletier": "soil_depth_pelletier",
        "soil_depth_statsgo":   "soil_depth_statsgo",
        "soil_porosity":        "soil_porosity",
        "soil_conductivity":    "soil_conductivity",
        "max_water_content":    "max_water_content",
        "sand_frac":            "sand_frac",
        "silt_frac":            "silt_frac",
        "clay_frac":            "clay_frac",
    },
    "camels_geol.txt": {
        "carbonate_rocks_frac": "carb_rocks_frac",
        "geol_permeability":    "geol_permeability",
    },
}

STATIC_ATTR_COLS: list[str] = []
for _m in _STATIC_FILE_COLS.values():
    STATIC_ATTR_COLS.extend(_m.values())
N_STATIC = len(STATIC_ATTR_COLS)          # 27
N_FEATURES = N_DYNAMIC + N_STATIC         # 30（LSTM 输入维度）


# ---------------------------------------------------------------------------
# 数据加载
# ---------------------------------------------------------------------------

def load_camels_forcing(camels_root: Path, gauge_id: str) -> pd.DataFrame:
    """加载单站 Daymet 强迫，仅保留 prcp, tmax, tmin。"""
    forcing_root = camels_root / FORCING_ROOT_REL
    files = list(forcing_root.glob(f"**/{gauge_id}_lump_cida_forcing_leap.txt"))
    if not files:
        raise FileNotFoundError(f"未找到站点 {gauge_id} 的 Daymet 强迫文件，请检查路径。")
    path = files[0]
    all_raw = ["Year", "Mnth", "Day", "Hr",
               "dayl(s)", "prcp(mm/day)", "srad(W/m2)",
               "swe(mm)", "tmax(C)", "tmin(C)", "vp(Pa)"]
    df = pd.read_csv(path, sep=r"\s+", skiprows=4, header=None,
                     names=all_raw, na_values=["-999", "-999.0"])
    df["date"] = pd.to_datetime(
        df[["Year", "Mnth", "Day"]].rename(
            columns={"Year": "year", "Mnth": "month", "Day": "day"}
        )
    )
    return df.set_index("date").rename(columns=_DYN_COL_MAP)[DYNAMIC_COLS].sort_index().astype(np.float32)


def load_camels_static_attrs(camels_root: Path) -> pd.DataFrame:
    """从 5 个 CAMELS 属性文件加载 27 个静态流域属性。"""
    attr_root = camels_root / "camels_us" / "CAMELS_US"
    dfs: list[pd.DataFrame] = []
    for fname, col_map in _STATIC_FILE_COLS.items():
        raw = pd.read_csv(attr_root / fname, sep=";", index_col=0,
                          na_values=["NA", "nan", "NaN", "-9999"])
        raw.index = raw.index.astype(str).str.zfill(8)
        available = {fc: nc for fc, nc in col_map.items() if fc in raw.columns}
        missing = set(col_map.keys()) - set(available.keys())
        if missing:
            logger.warning("%s 缺少列: %s", fname, missing)
        dfs.append(raw[list(available.keys())].rename(columns=available))
    result = pd.concat(dfs, axis=1)
    return result[[c for c in STATIC_ATTR_COLS if c in result.columns]].astype(np.float32)


def align_forcing_streamflow(forcing: pd.DataFrame, streamflow: pd.Series) -> pd.DataFrame:
    """按日期内连接，丢弃含 NaN 的行，最后一列为 streamflow。"""
    return forcing.join(streamflow.rename("streamflow"), how="inner").dropna()


# ---------------------------------------------------------------------------
# 模型
# ---------------------------------------------------------------------------

class HydroLSTM(nn.Module):
    """seq2seq LSTM：（动态强迫 ⊕ 静态属性）→ 残差预测。

    编码器处理上下文强迫（同时预测径流，辅助损失）。
    解码器利用编码器隐状态 + 预报期强迫，预测 TimesFM 残差。
    """

    def __init__(
        self,
        n_features: int = N_FEATURES,
        hidden: int = 128,
        n_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        _drop = dropout if n_layers > 1 else 0.0
        self.encoder = nn.LSTM(n_features, hidden, n_layers,
                               batch_first=True, dropout=_drop)
        self.decoder = nn.LSTM(n_features, hidden, n_layers,
                               batch_first=True, dropout=_drop)
        self.head = nn.Linear(hidden, 1)

    def forward(
        self,
        ctx_feat: torch.Tensor,  # [B, T_ctx, N_FEATURES]
        hor_feat: torch.Tensor,  # [B, T_hor, N_FEATURES]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ctx_out, (h, c) = self.encoder(ctx_feat)
        ctx_pred = self.head(ctx_out).squeeze(-1)   # [B, T_ctx]  辅助：预测径流
        hor_out, _ = self.decoder(hor_feat, (h, c))
        hor_pred = self.head(hor_out).squeeze(-1)   # [B, T_hor]  主要：预测残差
        return ctx_pred, hor_pred


# ---------------------------------------------------------------------------
# 训练数据集
# ---------------------------------------------------------------------------

class _WindowDataset(Dataset):
    """单站滑动窗口数据集（timesfm + xreg 模式）。

    gauge_data 格式：(dyn [T, N_DYN], sf [T], static_n [N_STATIC], tf_preds [T])
      tf_preds[t] = 微调 TimesFM 对第 t 天的 1 步超前预测（前 context_len 天为 NaN）。

    样本构成：
      ctx_feat [T_ctx, N_FEATURES]  归一化动态 ⊕ 静态
      hor_feat [T_hor, N_FEATURES]  同上（预报期）
      ctx_sf_n [T_ctx]              按窗口归一化的径流（编码器辅助损失）
      residual_n [T_hor]            (sf_hor - tf_pred_hor) / sf_std（解码器主损失）
    """

    def __init__(
        self,
        dyn: np.ndarray,         # [T, N_DYN]
        sf: np.ndarray,          # [T]
        static_n: np.ndarray,    # [N_STATIC] 已归一化
        tf_preds: np.ndarray,    # [T] TimesFM 预测（前 context_len 天 NaN）
        context_len: int,
        horizon: int,
        dyn_mean: np.ndarray,
        dyn_std: np.ndarray,
        stride: int = 1,
    ):
        self.dyn = dyn
        self.sf = sf
        self.static_n = static_n
        self.tf_preds = tf_preds
        self.ctx = context_len
        self.hor = horizon
        self.dyn_mean = dyn_mean
        self.dyn_std = dyn_std
        # 只保留 tf_preds 有效（非 NaN）的窗口
        n = len(sf)
        self.indices = [
            i for i in range(0, n - context_len - horizon + 1, stride)
            if not np.isnan(tf_preds[i + context_len : i + context_len + horizon]).any()
        ]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        s = self.indices[idx]
        ctx_dyn = self.dyn[s : s + self.ctx]
        hor_dyn = self.dyn[s + self.ctx : s + self.ctx + self.hor]
        ctx_sf  = self.sf[s : s + self.ctx]
        hor_sf  = self.sf[s + self.ctx : s + self.ctx + self.hor]
        hor_tf  = self.tf_preds[s + self.ctx : s + self.ctx + self.hor]

        # 动态特征归一化 + 静态拼接
        ctx_dyn_n = (ctx_dyn - self.dyn_mean) / self.dyn_std
        hor_dyn_n = (hor_dyn - self.dyn_mean) / self.dyn_std
        static_ctx = np.broadcast_to(self.static_n, (self.ctx, len(self.static_n)))
        static_hor = np.broadcast_to(self.static_n, (self.hor, len(self.static_n)))
        ctx_feat = np.concatenate([ctx_dyn_n, static_ctx], axis=1).astype(np.float32)
        hor_feat = np.concatenate([hor_dyn_n, static_hor], axis=1).astype(np.float32)

        # 径流按上下文归一化（编码器辅助损失用）
        sf_mean = float(ctx_sf.mean())
        sf_std  = float(max(ctx_sf.std(), 1e-6))
        ctx_sf_n = ((ctx_sf - sf_mean) / sf_std).astype(np.float32)

        # 残差标签（解码器主损失用）：除以 sf_std 对齐量级
        residual = (hor_sf - hor_tf).astype(np.float32)
        residual_n = (residual / sf_std).astype(np.float32)

        return (
            torch.from_numpy(ctx_feat),
            torch.from_numpy(hor_feat),
            torch.from_numpy(ctx_sf_n),
            torch.from_numpy(residual_n),
            torch.tensor([sf_std], dtype=torch.float32),  # 推理时反归一化用
        )


# ---------------------------------------------------------------------------
# 归一化统计量
# ---------------------------------------------------------------------------

def compute_stats(
    gauge_arrays: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """计算动态特征和静态属性的全局均值/标准差。"""
    all_dyn    = np.concatenate([g[0] for g in gauge_arrays], axis=0)
    all_static = np.stack([g[2] for g in gauge_arrays], axis=0)

    def _stats(arr: np.ndarray, axis=0):
        m = arr.mean(axis=axis).astype(np.float32)
        s = np.where(arr.std(axis=axis) > 1e-6, arr.std(axis=axis), 1.0).astype(np.float32)
        return m, s

    dyn_mean, dyn_std       = _stats(all_dyn)
    static_mean, static_std = _stats(all_static)
    return dyn_mean, dyn_std, static_mean, static_std


# ---------------------------------------------------------------------------
# 训练
# ---------------------------------------------------------------------------

def train_lstm(
    gauge_arrays: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    # 每个元素：(dyn [T,N_DYN], sf [T], static [N_STATIC], tf_preds [T])
    context_len: int,
    horizon: int,
    hidden: int = 128,
    n_layers: int = 2,
    dropout: float = 0.1,
    epochs: int = 20,
    batch_size: int = 512,
    lr: float = 1e-3,
    val_ratio: float = 0.15,
    stride: int = 15,
    aux_weight: float = 0.2,   # 编码器辅助损失权重（预测径流）
    device: str | None = None,
    save_dir: Path | None = None,
) -> tuple[HydroLSTM, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """训练 HydroLSTM（残差模式），返回 (模型, dyn_mean, dyn_std, static_mean, static_std)。"""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("训练设备: %s，共 %d 个站点", device, len(gauge_arrays))

    dyn_mean, dyn_std, static_mean, static_std = compute_stats(gauge_arrays)

    datasets = []
    for dyn, sf, static_raw, tf_preds in gauge_arrays:
        if len(sf) < context_len + horizon:
            continue
        static_n = ((static_raw - static_mean) / static_std).astype(np.float32)
        ds = _WindowDataset(
            dyn, sf, static_n, tf_preds,
            context_len, horizon, dyn_mean, dyn_std, stride=stride,
        )
        if len(ds) > 0:
            datasets.append(ds)

    if not datasets:
        raise RuntimeError("所有站点的 tf_preds 均为 NaN，请先预计算 TimesFM 预测。")

    full_ds = ConcatDataset(datasets)
    logger.info("有效训练窗口数: %d", len(full_ds))

    n_val = max(1, int(len(full_ds) * val_ratio))
    train_ds, val_ds = random_split(
        full_ds, [len(full_ds) - n_val, n_val],
        generator=torch.Generator().manual_seed(42),
    )
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
        for ctx_feat, hor_feat, ctx_sf_n, residual_n, _ in train_loader:
            ctx_feat   = ctx_feat.to(device)
            hor_feat   = hor_feat.to(device)
            ctx_sf_n   = ctx_sf_n.to(device)
            residual_n = residual_n.to(device)

            ctx_pred, hor_pred = model(ctx_feat, hor_feat)
            # 主损失：解码器预测残差
            loss_main = nn.functional.mse_loss(hor_pred, residual_n)
            # 辅助损失：编码器预测径流（帮助建立有效隐状态）
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
            for ctx_feat, hor_feat, ctx_sf_n, residual_n, _ in val_loader:
                ctx_feat   = ctx_feat.to(device)
                hor_feat   = hor_feat.to(device)
                residual_n = residual_n.to(device)
                _, hp = model(ctx_feat, hor_feat)
                vl += nn.functional.mse_loss(hp, residual_n).item()
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
        }
        with open(save_dir / "lstm_meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        logger.info("LSTM 已保存到 %s", save_dir)

    return model, dyn_mean, dyn_std, static_mean, static_std


# ---------------------------------------------------------------------------
# 推理封装
# ---------------------------------------------------------------------------

class LSTMXRegPredictor:
    """LSTM-XReg 批量推理：返回残差预测（原始尺度）。"""

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
    def from_checkpoint(cls, save_dir: Path, device: str = "cpu") -> "LSTMXRegPredictor":
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
    def predict_residual_batch(
        self,
        ctx_dyn: np.ndarray,    # [N, T_ctx, N_DYN]
        hor_dyn: np.ndarray,    # [N, T_hor, N_DYN]
        ctx_sf: np.ndarray,     # [N, T_ctx] 用于计算 sf_std（反归一化用）
        static: np.ndarray,     # [N, N_STATIC] 原始尺度
        batch_size: int = 256,
    ) -> np.ndarray:
        """批量预测残差，返回 [N, T_hor]（原始径流尺度）。"""
        ctx_feat = self._build_feat(ctx_dyn, static)   # [N, T_ctx, N_FEATURES]
        hor_feat = self._build_feat(hor_dyn, static)   # [N, T_hor, N_FEATURES]

        # sf_std 用于反归一化（与训练时 residual_n = residual / sf_std 对应）
        sf_std = np.maximum(ctx_sf.std(axis=1, keepdims=True), 1e-6)  # [N, 1]

        all_hor = []
        for i in range(0, len(ctx_feat), batch_size):
            ctx_t = torch.from_numpy(ctx_feat[i : i + batch_size]).to(self.device)
            hor_t = torch.from_numpy(hor_feat[i : i + batch_size]).to(self.device)
            _, hp_n = self.model(ctx_t, hor_t)
            all_hor.append(hp_n.cpu().numpy())

        hor_pred_n = np.concatenate(all_hor, axis=0)   # [N, T_hor]
        # 反归一化：residual = pred_n * sf_std（无需加均值，残差以 0 为中心）
        return hor_pred_n * sf_std
