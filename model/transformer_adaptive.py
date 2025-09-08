# model/transformer_adaptive.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.lag_layers import LagAttention

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        # x: (B, T, D)
        T = x.size(1)
        return x + self.pe[:, :T, :]

class PerformativeTransformerAdaptive(nn.Module):
    """
    Two-head model:
     - LagAttention collapses lag cube (B, W, F, L) -> (B, W, F)
     - Encoder on (B, W, F) -> context
     - Head 1: predict future exogenous trend (B, K, perf_F)
     - Head 2: predict target (B, K)
    """
    def __init__(self, feature_dim: int, perf_F: int, horizon_k: int,
                 d_model: int = 128, nhead: int = 4, num_layers: int = 2,
                 dropout: float = 0.1, lag_dim: int = 9, conditional_lag: bool=False):
        super().__init__()
        self.perf_F = perf_F
        self.horizon_k = horizon_k

        self.lag_attn = LagAttention(feature_dim, lag_dim, conditional=conditional_lag)

        self.inp = nn.Linear(feature_dim, d_model)
        self.pos = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True, dropout=dropout)
        self.enc = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Forecast head (target)
        self.dec_target = nn.GRU(d_model, d_model, batch_first=True)
        self.out_y = nn.Linear(d_model, 1)

        # Trend head (future exogenous for first perf_F features)
        self.dec_trend = nn.GRU(d_model, d_model, batch_first=True)
        self.out_trend = nn.Linear(d_model, perf_F)

    def forward(self, x_lag, meta=None):
        """
        x_lag: (B, W, F, L) normalized lag cube
        meta:  optional (B, W, M) one-hot region meta; can be concatenated if desired
        Returns:
          pred_trend: (B, K, perf_F)
          pred_y:     (B, K)
        """
        B, W, F, L = x_lag.shape

        # 1) Adaptive lag collapse
        xW = self.lag_attn(x_lag)             # (B, W, F)

        # 2) (Optional) concat meta
        if meta is not None:
            # meta is (B, W, M); project to F and concat
            M = meta.size(-1)
            meta_proj = nn.functional.pad(meta, (0, max(0, F - M)))[:, :, :F]  # simple placeholder
            xW = xW + meta_proj  # or torch.cat and adjust 'inp' layer

        # 3) Encode lookback
        z = self.inp(xW)                      # (B, W, D)
        z = self.pos(z)
        h = self.enc(z)                       # (B, W, D)

        # 4) Prepare a simple horizon query (zeros) for both heads
        qry = torch.zeros(B, self.horizon_k, h.size(-1), device=h.device)

        # Decode target
        _, h_last = self.dec_target(h)        # (1, B, D)
        y_seq, _ = self.dec_target(qry, h_last)
        pred_y = self.out_y(y_seq).squeeze(-1)        # (B, K)

        # Decode trend
        _, h_last2 = self.dec_trend(h)
        tr_seq, _ = self.dec_trend(qry, h_last2)
        pred_trend = self.out_trend(tr_seq)            # (B, K, perf_F)

        return pred_trend, pred_y

def perform_transformer_adaptive(feature_dim=12, perf_F=6, horizon_k=8,
                                 d_model=128, nhead=4, num_layers=2, lag_dim=9,
                                 conditional_lag=False):
    return PerformativeTransformerAdaptive(feature_dim, perf_F, horizon_k,
                                           d_model, nhead, num_layers,
                                           dropout=0.1, lag_dim=lag_dim,
                                           conditional_lag=conditional_lag)