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
        print("DEBUG self.W type:", type(self.W), self.W.shape if hasattr(self.W, "shape") else self.W)

    def forward(self, x):
        """
        x: (B, T, Feat, L) lag cube
        returns: (B, T, Feat) lag-collapsed features (adaptive delay)
        """
        B, T, Feat, L = x.shape
        if self.conditional:
            # condition on x_t (lag 0)
            xt = x[..., 0]                       # (B, T, Feat)
            logits = self.cond_mlp(xt)          # (B, T, Feat*L)
            logits = logits.view(B, T, Feat, L)    # (B, T, Feat, L)
            attn = F.softmax(logits, dim=-1)    # (B, T, Feat, L)
            out = (attn * x).sum(dim=-1)        # (B, T, Feat)
        else:
            # global per-feature lag weights
            attn = F.softmax(self.W, dim=-1)    # (Feat, L)
            out = torch.einsum('btfl,fl->btf', x, attn)
        return out