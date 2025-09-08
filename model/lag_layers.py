# model/lag_layers.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class LagAttention(nn.Module):
    """
    Learn attention over L lags for each feature. Optionally make it
    input-conditioned (via a tiny MLP) if you want sample-wise dynamics.
    """
    def __init__(self, feature_dim: int, lag_dim: int, conditional: bool = False, hidden: int = 64):
        super().__init__()
        self.feature_dim = feature_dim
        self.lag_dim = lag_dim
        self.conditional = conditional
        if conditional:
            # simple conditioning on the current x_t (lag 0)
            self.cond_mlp = nn.Sequential(
                nn.Linear(feature_dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, feature_dim * lag_dim)
            )
        else:
            self.W = nn.Parameter(torch.randn(feature_dim, lag_dim) * 0.01)

    def forward(self, x):
        """
        x: (B, T, F, L) lag cube
        returns: (B, T, F) lag-collapsed features (adaptive delay)
        """
        B, T, F, L = x.shape
        if self.conditional:
            # condition on x_t (lag 0)
            xt = x[..., 0]                       # (B, T, F)
            logits = self.cond_mlp(xt)          # (B, T, F*L)
            logits = logits.view(B, T, F, L)    # (B, T, F, L)
            attn = F.softmax(logits, dim=-1)    # (B, T, F, L)
            out = (attn * x).sum(dim=-1)        # (B, T, F)
        else:
            # global per-feature lag weights
            attn = F.softmax(self.W, dim=-1)    # (F, L)
            out = torch.einsum('btfl,fl->btf', x, attn)
        return out