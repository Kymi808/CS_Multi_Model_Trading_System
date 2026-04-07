"""
Time Series Transformer (TST) alpha model for cross-sectional equity ranking.

Architecture:
- Positional encoding for temporal ordering
- Multi-head self-attention captures dependencies across time steps
- Operates on rolling windows of cross-sectional features per stock
- Outputs a scalar prediction (expected relative return)
"""
import numpy as np
import pandas as pd
import logging
import os
import pickle
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 500, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: d_model // 2 + d_model % 2])
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class TimeSeriesTransformer(nn.Module):
    def __init__(
        self,
        n_features: int,
        d_model: int = 64,
        n_heads: int = 4,
        n_encoder_layers: int = 2,
        d_ff: int = 128,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_enc = PositionalEncoding(d_model, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_encoder_layers)
        self.output_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, x):
        # x: (batch, seq_len, n_features)
        x = self.input_proj(x)
        x = self.pos_enc(x)
        x = self.encoder(x)
        # Use the last time step's representation
        x = x[:, -1, :]
        return self.output_head(x).squeeze(-1)


class SequenceDataset(Dataset):
    """Builds rolling sequences from panel data for transformer input."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class TSTRanker:
    """Time Series Transformer ensemble ranker matching EnsembleRanker interface."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.models: List[TimeSeriesTransformer] = []
        self.feature_names: List[str] = []
        self.feature_importance: Optional[pd.Series] = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _build_sequences(
        self, X: pd.DataFrame, y: pd.Series, seq_len: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build rolling sequences from (date, ticker) multi-index data.
        For each (date, ticker), gather the previous seq_len days of features.
        """
        if isinstance(X.index, pd.MultiIndex):
            dates = sorted(X.index.get_level_values(0).unique())
            tickers = X.index.get_level_values(1).unique()
        else:
            # Flat index — treat as single sequence
            X_vals = X.values
            y_vals = y.values
            seqs_X, seqs_y = [], []
            for i in range(seq_len, len(X_vals)):
                seqs_X.append(X_vals[i - seq_len : i])
                seqs_y.append(y_vals[i])
            return np.array(seqs_X), np.array(seqs_y)

        # For multi-index: build per-ticker sequences
        seqs_X, seqs_y = [], []
        date_list = list(dates)
        date_to_idx = {d: i for i, d in enumerate(date_list)}

        for ticker in tickers:
            ticker_mask = X.index.get_level_values(1) == ticker
            ticker_X = X.loc[ticker_mask].copy()
            ticker_y = y.loc[y.index.isin(ticker_X.index)]

            if len(ticker_X) < seq_len + 1:
                continue

            ticker_dates = sorted(ticker_X.index.get_level_values(0).unique())
            vals = ticker_X.droplevel(1).reindex(ticker_dates).values
            targets = ticker_y.droplevel(1).reindex(ticker_dates).values

            for i in range(seq_len, len(vals)):
                if np.isnan(targets[i]):
                    continue
                seq = vals[i - seq_len : i]
                if np.isnan(seq).mean() > 0.5:
                    continue
                # Fill NaN with 0 within sequences
                seq = np.nan_to_num(seq, nan=0.0)
                seqs_X.append(seq)
                seqs_y.append(targets[i])

        if not seqs_X:
            return np.array([]).reshape(0, seq_len, X.shape[1]), np.array([])
        return np.array(seqs_X), np.array(seqs_y)

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ) -> dict:
        self.feature_names = list(X_train.columns)
        X_train = X_train.replace([np.inf, -np.inf], np.nan)
        if X_val is not None:
            X_val = X_val.replace([np.inf, -np.inf], np.nan)

        seq_len = self.cfg.sequence_length
        n_features = len(self.feature_names)

        train_X, train_y = self._build_sequences(X_train, y_train, seq_len)
        val_X, val_y = None, None
        if X_val is not None and y_val is not None:
            val_X, val_y = self._build_sequences(X_val, y_val, seq_len)

        if len(train_X) < 10:
            logger.warning("TST: Not enough sequences to train, falling back to simple model")
            return {"n_train": 0, "n_features": n_features, "n_ensemble": 0}

        self.models = []
        for seed_idx in range(min(2, getattr(self.cfg, "n_ensemble", 2))):
            seed = self.cfg.random_state + seed_idx * 100
            torch.manual_seed(seed)
            np.random.seed(seed)

            model = TimeSeriesTransformer(
                n_features=n_features,
                d_model=self.cfg.d_model,
                n_heads=self.cfg.n_heads,
                n_encoder_layers=self.cfg.n_encoder_layers,
                d_ff=self.cfg.d_ff,
                dropout=self.cfg.dropout,
            ).to(self.device)

            # JIT compile for GPU acceleration (PyTorch 2.x)
            if self.device.type == "cuda":
                try:
                    model = torch.compile(model, mode="reduce-overhead")
                    logger.info(f"  TST compiled with torch.compile")
                except Exception:
                    pass

            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=self.cfg.learning_rate,
                weight_decay=self.cfg.weight_decay,
            )
            criterion = nn.MSELoss()

            train_ds = SequenceDataset(train_X, train_y)
            train_dl = DataLoader(
                train_ds, batch_size=self.cfg.batch_size, shuffle=True,
                drop_last=False, num_workers=2, pin_memory=(self.device.type == "cuda"),
            )

            # Mixed precision for ~2x GPU speedup
            use_amp = self.device.type == "cuda"
            scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

            best_val_loss = float("inf")
            patience_counter = 0

            for epoch in range(self.cfg.epochs):
                model.train()
                epoch_loss = 0
                for batch_X, batch_y in train_dl:
                    batch_X = batch_X.to(self.device, non_blocking=True)
                    batch_y = batch_y.to(self.device, non_blocking=True)
                    optimizer.zero_grad(set_to_none=True)
                    with torch.amp.autocast("cuda", enabled=use_amp):
                        pred = model(batch_X)
                        loss = criterion(pred, batch_y)
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    epoch_loss += loss.item() * len(batch_X)
                epoch_loss /= len(train_ds)

                # Validation
                if val_X is not None and len(val_X) > 0:
                    model.eval()
                    with torch.no_grad(), torch.amp.autocast("cuda", enabled=use_amp):
                        val_tensor = torch.FloatTensor(val_X).to(self.device)
                        val_pred = model(val_tensor)
                        val_loss = criterion(
                            val_pred, torch.FloatTensor(val_y).to(self.device)
                        ).item()

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    else:
                        patience_counter += 1
                        if patience_counter >= self.cfg.patience:
                            model.load_state_dict(best_state)
                            break

            model.eval()
            self.models.append(model)

        # Compute simple feature importance via input gradient
        self._compute_feature_importance(train_X)

        metrics = {
            "n_train": len(train_X),
            "n_features": n_features,
            "n_ensemble": len(self.models),
        }

        if val_X is not None and len(val_X) > 0:
            val_pred = self._predict_sequences(val_X)
            valid = ~np.isnan(val_y) & ~np.isnan(val_pred)
            if valid.sum() > 10:
                metrics["val_ic"] = float(np.corrcoef(val_pred[valid], val_y[valid])[0, 1])
                metrics["val_rank_ic"] = float(
                    pd.Series(val_pred[valid]).corr(
                        pd.Series(val_y[valid]), method="spearman"
                    )
                )
                logger.info(
                    f"  TST IC: {metrics['val_ic']:.4f}, "
                    f"Rank IC: {metrics['val_rank_ic']:.4f}"
                )

        return metrics

    def _predict_sequences(self, X_seq: np.ndarray) -> np.ndarray:
        """Predict from pre-built sequence arrays."""
        preds = np.zeros(len(X_seq))
        for model in self.models:
            model.eval()
            with torch.no_grad():
                # Process in chunks to avoid OOM
                chunk_size = 1024
                for i in range(0, len(X_seq), chunk_size):
                    chunk = torch.FloatTensor(X_seq[i : i + chunk_size]).to(self.device)
                    preds[i : i + chunk_size] += model(chunk).cpu().numpy()
        preds /= max(len(self.models), 1)
        return preds

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Predict from flat ML-format DataFrame (builds sequences internally)."""
        if not self.models:
            raise ValueError("Model not trained")
        X = X.replace([np.inf, -np.inf], np.nan)
        missing = set(self.feature_names) - set(X.columns)
        for f in missing:
            X[f] = np.nan

        seq_len = self.cfg.sequence_length
        X_aligned = X[self.feature_names]

        # Build sequences
        if isinstance(X.index, pd.MultiIndex):
            tickers = X.index.get_level_values(1).unique()
            dates = sorted(X.index.get_level_values(0).unique())
            all_preds = {}

            for ticker in tickers:
                ticker_mask = X_aligned.index.get_level_values(1) == ticker
                ticker_X = X_aligned.loc[ticker_mask].droplevel(1)
                ticker_dates = sorted(ticker_X.index)

                if len(ticker_dates) < seq_len:
                    continue

                vals = ticker_X.reindex(ticker_dates).values
                vals = np.nan_to_num(vals, nan=0.0)

                seqs = []
                seq_dates = []
                for i in range(seq_len, len(vals)):
                    seqs.append(vals[i - seq_len : i])
                    seq_dates.append(ticker_dates[i])

                if seqs:
                    seq_preds = self._predict_sequences(np.array(seqs))
                    for d, p in zip(seq_dates, seq_preds):
                        all_preds[(d, ticker)] = p

            if all_preds:
                result = pd.Series(all_preds)
                result.index = pd.MultiIndex.from_tuples(result.index)
                return result
            return pd.Series(dtype=float)
        else:
            vals = np.nan_to_num(X_aligned.values, nan=0.0)
            if len(vals) <= seq_len:
                return pd.Series(0.0, index=X.index)
            seqs = []
            for i in range(seq_len, len(vals)):
                seqs.append(vals[i - seq_len : i])
            preds = self._predict_sequences(np.array(seqs))
            return pd.Series(preds, index=X.index[seq_len:])

    def _compute_feature_importance(self, train_X: np.ndarray):
        """Approximate feature importance via mean absolute gradient."""
        if not self.models or len(train_X) == 0:
            self.feature_importance = pd.Series(
                1.0, index=self.feature_names
            )
            return

        model = self.models[0]
        model.eval()
        sample = torch.FloatTensor(train_X[:min(500, len(train_X))]).to(self.device)
        sample.requires_grad_(True)

        pred = model(sample)
        pred.sum().backward()

        if sample.grad is not None:
            # Average absolute gradient across batch and time
            grad_imp = sample.grad.abs().mean(dim=(0, 1)).cpu().numpy()
            self.feature_importance = pd.Series(
                grad_imp, index=self.feature_names
            ).sort_values(ascending=False)
        else:
            self.feature_importance = pd.Series(
                1.0, index=self.feature_names
            )

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        save_data = {
            "feature_names": self.feature_names,
            "feature_importance": self.feature_importance,
            "config": self.cfg,
            "model_states": [m.state_dict() for m in self.models],
            "n_features": len(self.feature_names),
        }
        with open(path, "wb") as f:
            pickle.dump(save_data, f)

    def load(self, path: str):
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.feature_names = data["feature_names"]
        self.feature_importance = data["feature_importance"]
        self.models = []
        n_features = data["n_features"]
        for state in data["model_states"]:
            model = TimeSeriesTransformer(
                n_features=n_features,
                d_model=self.cfg.d_model,
                n_heads=self.cfg.n_heads,
                n_encoder_layers=self.cfg.n_encoder_layers,
                d_ff=self.cfg.d_ff,
                dropout=self.cfg.dropout,
            ).to(self.device)
            model.load_state_dict(state)
            model.eval()
            self.models.append(model)
