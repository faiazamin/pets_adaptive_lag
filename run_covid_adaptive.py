# run_covid_adaptive.py
import numpy as np
import random
import os
import argparse
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import tqdm

from dataset.covid.get_covid_adaptive import create_covid_norm_all_adaptive, all_hhs_regions
from model.transformer_adaptive import perform_transformer_adaptive

def to_one_hot(a):
    b = np.zeros((a.size, a.max() + 1))
    b[np.arange(a.size), a] = 1
    return b

def train(model, optim, x_lag, y, xf_future, x_lag_valid, y_valid, xf_future_valid,
          meta_train, meta_valid, mask_future, device, k=8, epoch=8000, aux_w=0.0):
    model.to(device)
    scheduler = MultiStepLR(optim, milestones=[10000], gamma=0.1)

    x_lag = torch.tensor(x_lag, dtype=torch.float32, device=device)
    y = torch.tensor(y, dtype=torch.float32, device=device)
    xf_future = torch.tensor(xf_future, dtype=torch.float32, device=device)
    meta_train = torch.tensor(meta_train, dtype=torch.float32, device=device)
    mask_future = torch.tensor(mask_future, dtype=torch.float32, device=device)

    loss_track = []
    mse = torch.nn.MSELoss(reduction='none')

    for i in range(epoch):
        model.train()
        pred_trend, pred_y = model(x_lag, meta_train)

        # Forecast loss (target)
        loss_y = ((pred_y - y)**2).mean()

        # Aux trend loss (only on perf_F future)
        loss_trend = (mse(pred_trend, xf_future).mean(dim=-1) * mask_future).mean()

        loss = loss_y + aux_w * loss_trend

        optim.zero_grad()
        loss.backward()
        optim.step()
        scheduler.step()

        loss_track.append(loss_y.item())

        if i % 1000 == 0:
            print(f"epoch: {i}, forecast MSE: {loss_y.item():.5f}, trend MSE: {loss_trend.item():.5f}")
            _ = test(model, x_lag_valid, y_valid, xf_future_valid, meta_valid, device)

    return model, np.array(loss_track)

def test(model, x_lag, y, xf_future, meta, device):
    model.eval()
    x_lag = torch.tensor(x_lag, dtype=torch.float32, device=device)
    meta = torch.tensor(meta, dtype=torch.float32, device=device)
    with torch.no_grad():
        pred_trend, pred_y = model(x_lag, meta)
    pred_y = pred_y.cpu().numpy()
    loss = np.mean((pred_y - y)**2)
    print("Valid/Test forecast MSE:", loss)
    return pred_y

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--k', type=int, default=8)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--dev', type=str, default='cuda:0')
    parser.add_argument('--epoch_first', type=int, default=20000)
    parser.add_argument('--epoch_rolling', type=int, default=3000)
    parser.add_argument('--conditional_lag', action='store_true')
    parser.add_argument('--aux',type=int, default=1)
    args = parser.parse_args()

    device = args.dev
    seed = args.seed
    np.random.seed(seed); random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed(seed)

    k = args.k
    start, end = 80, 108  # epiweeks
    perf_F = 6

    # Build model
    model = perform_transformer_adaptive(
        feature_dim=12, perf_F=perf_F, horizon_k=k,
        d_model=128, nhead=4, num_layers=2,
        lag_dim=k+1, conditional_lag=args.conditional_lag
    )
    optim = Adam(model.parameters(), lr=1e-3)

    loss_tracker = []

    for cur in tqdm(range(start, end)):
        # new adaptive loader
        [x_tr, y_tr, xf_tr, x_te, y_te, xf_te], [x_scalers, y_scalers], masks, meta = \
            create_covid_norm_all_adaptive(cur, window_size=16, k=k, perf_F=perf_F)

        # meta one-hot per region index (same shape pattern as original)
        num_samples, seq_len, F, L = x_tr.shape
        meta_test_idx = np.array([meta[i] for i in range(0, len(meta), int(len(meta)/48))])
        meta_oh = to_one_hot(meta.astype(int).reshape(-1)).reshape(num_samples, seq_len, -1)
        meta_test_oh = to_one_hot(meta_test_idx.astype(int).reshape(-1)).reshape(48, seq_len, -1)

        epochs = args.epoch_first if cur == start else args.epoch_rolling
        model, loss_track = train(
            model, optim,
            x_tr, y_tr, xf_tr,
            x_te, y_te, xf_te,
            meta_oh, meta_test_oh,
            masks['future_mask'], device, k=k, epoch=epochs, aux_w=args.aux
        )
        loss_tracker.append(loss_track)

        # Test (and inverse-transform per region for saving)
        pred_norm = test(model, x_te, y_te, xf_te, meta_test_oh, device)

        # Save per-region predictions in original scale
        for j, region in enumerate(all_hhs_regions):
            scaler_y = y_scalers[region]
            region_pred = pred_norm[j:j+1]                  # (1, k)
            true_pred = scaler_y.inverse_transform(region_pred)
            out_dir = f'./covid/res_fps_adaptive/transformer/seed_{seed}/{region}'
            os.makedirs(out_dir, exist_ok=True)
            np.save(f'{out_dir}/pred_{cur}.npy', true_pred)

if __name__ == '__main__':
    main()