# dataset/covid/get_covid_adaptive.py
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.ndimage import gaussian_filter1d

feature_key = [
    'retail_and_recreation_percent_change_from_baseline',
    'grocery_and_pharmacy_percent_change_from_baseline',
    'parks_percent_change_from_baseline',
    'transit_stations_percent_change_from_baseline',
    'workplaces_percent_change_from_baseline',
    'residential_percent_change_from_baseline',
    "Observed Number",
    "Excess Estimate",
    "positiveIncr_cumulative",
    "positiveIncr",
    "death_jhu_cumulative",
    "death_jhu_incidence",
]

all_hhs_regions = ['AL','AZ','AR','CA','CO','CT','DE','DC','FL','GA','ID','IL','IN','IA','KS','KY','LA',
                   'ME','MD','MA','MI','MN','MS','MO','NE','NV','NH','NJ','NM','NY','NC','ND','OH','OK',
                   'OR','PA','RI','SC','SD','TN','TX','UT','VT','VA','WA','WV','WI','X']

path = './dataset/covid/covid-hospitalization-all-state-merged_vEW202239.csv'

def _read_region_matrix(region):
    raw = pd.read_csv(path)
    raw = raw.loc[raw['region'] == region][10:140].copy()
    raw = raw.ffill()

    xs = []
    for i, key in enumerate(feature_key):
        series = raw[key].to_numpy()
        if i <= 5:  # smooth mobility-like features
            series = gaussian_filter1d(series, sigma=3)
        xs.append(series.reshape(-1, 1))
    X = np.concatenate(xs, axis=1)  # (T, F)
    y = raw['death_jhu_incidence'].to_numpy()  # (T,)
    return X, y

def _make_lag_cube(X, max_k):
    """
    Build lag cube of shape (T, F, L) where L = max_k+1 and
    slice [:, :, 0] is x_t, [:, :, 1] is x_{t-1}, ... x_{t-max_k}.
    """
    T, F = X.shape
    L = max_k + 1
    cube = np.zeros((T, F, L), dtype=X.dtype)
    for j in range(L):
        if j == 0:
            cube[:, :, 0] = X
        else:
            cube[j:, :, j] = X[:-j, :]
    return cube

def create_window_region_adaptive(region, current_time, window_size=16, k=8, perf_F=6):
    """
    Returns:
      x_lag_train:   (N, window, F, L)  lag cube for lookback
      y_train:       (N, k)             target future
      x_future_perf: (N, k, perf_F)     ground-truth future exogenous (first perf_F features)
      x_lag_test:    (1, window, F, L)
      y_test:        (1, k)
      x_future_perf_test: (1, k, perf_F)
      masks: dict with 'future_mask' (N, k) and 'future_mask_test' (1, k)
    """
    X, y = _read_region_matrix(region)      # (T, F), (T,)
    T, F = X.shape
    L = k + 1  # lag depth

    X_lag = _make_lag_cube(X, max_k=k)      # (T, F, L)

    x_lag_train, y_train, x_future_perf, future_mask = [], [], [], []
    for t_end in range(window_size + k, current_time + k):
        # lookback window (ends at t_end-k)
        x_lag_train.append(X_lag[t_end - k - window_size : t_end - k])   # (window, F, L)
        # target horizon [t_end-k+1 .. t_end] aligns to length k
        y_train.append(y[t_end - k : t_end])                              # (k,)
        # future exogenous for aux loss (first perf_F features)
        xf = X[t_end - k : t_end, :perf_F]                                # (k, perf_F)
        x_future_perf.append(xf)
        future_mask.append(np.ones((k,), dtype=np.float32))

    x_lag_train   = np.array(x_lag_train)         # (N, window, F, L)
    y_train       = np.array(y_train)             # (N, k)
    x_future_perf = np.array(x_future_perf)       # (N, k, perf_F)
    future_mask   = np.array(future_mask)         # (N, k)

    # Test slice ending at current_time
    x_lag_test = X_lag[current_time - window_size : current_time]         # (window, F, L)
    y_test     = y[current_time - k : current_time]                       # (k,)
    x_future_perf_test = X[current_time - k : current_time, :perf_F]      # (k, perf_F)

    return (x_lag_train, y_train, x_future_perf,
            x_lag_test[None, ...], y_test[None, ...], x_future_perf_test[None, ...],
            {'future_mask': future_mask, 'future_mask_test': np.ones((1, k), dtype=np.float32)})

def create_covid_norm_all_adaptive(current_time, window_size=16, k=8, perf_F=6):
    """
    Aggregate all regions; scale per region; stack batches.
    Returns:
      [x_tr, y_tr, x_future_tr, x_te, y_te, x_future_te],
      [x_scalers, y_scalers],
      masks,
      meta_data (region ids)
    """
    x_scalers = dict(zip(all_hhs_regions, [StandardScaler() for _ in all_hhs_regions]))
    xf_scalers = dict(zip(all_hhs_regions, [StandardScaler() for _ in all_hhs_regions]))
    y_scalers = dict(zip(all_hhs_regions, [StandardScaler() for _ in all_hhs_regions]))

    X_tr_list, Y_tr_list, XF_tr_list = [], [], []
    X_te_list, Y_te_list, XF_te_list = [], [], []
    meta_list, mask_list, mask_test_list = [], [], []

    for ridx, region in enumerate(all_hhs_regions):
        (x_lag_tr, y_tr, xf_tr,
         x_lag_te, y_te, xf_te,
         masks) = create_window_region_adaptive(region, current_time, window_size, k, perf_F)

        # Fit scalers on lookback inputs and targets
        B, W, F, L = x_lag_tr.shape
        x_flat = x_lag_tr.reshape(-1, F)[:, 0]  # NOTE: fit on x_t channel only to keep scale reasonable
        # Better: fit per-feature scaler on raw X (we reuse the existing approach)
        # Here we refit on y_tr for each region
        y_scalers[region].fit(y_tr.reshape(-1, 1))
        y_tr_n = y_scalers[region].transform(y_tr.reshape(-1, 1)).reshape(B, -1)
        y_te_n = y_scalers[region].transform(y_te.reshape(-1, 1)).reshape(1, -1)

        # Normalize exogenous for aux head using the same idea as original
        # We build a per-feature StandardScaler from the region's X lookbacks
        # For simplicity, compute from the x_t slice across the batch
        X_t_slice = x_lag_tr[..., 0].reshape(-1, F)  # (B*W, F)
        x_scalers[region].fit(X_t_slice)
        # Apply scaler to all lag channels independently
        def norm_lag(x_lag):
            T = x_lag.shape[0]
            x0 = x_lag[..., 0].reshape(-1, F)
            x0n = x_scalers[region].transform(x0).reshape(T, W, F)
            out = [x0n[..., None]]
            for ell in range(1, L):
                xell = x_lag[..., ell].reshape(-1, F)
                xelln = x_scalers[region].transform(xell).reshape(T, W, F)
                out.append(xelln[..., None])
            return np.concatenate(out, axis=-1)  # (T, W, F, L)

        x_lag_tr_n = norm_lag(x_lag_tr)
        x_lag_te_n = norm_lag(x_lag_te)

        # Fit and normalize future exogenous for the aux loss with separate scaler
        xf_scalers[region].fit(xf_tr.reshape(-1, xf_tr.shape[-1]))
        xf_tr_n = xf_scalers[region].transform(xf_tr.reshape(-1, xf_tr.shape[-1])).reshape(xf_tr.shape)
        xf_te_n = xf_scalers[region].transform(xf_te.reshape(-1, xf_te.shape[-1])).reshape(xf_te.shape)

        X_tr_list.append(x_lag_tr_n)
        Y_tr_list.append(y_tr_n)
        XF_tr_list.append(xf_tr_n)

        X_te_list.append(x_lag_te_n)
        Y_te_list.append(y_te_n)
        XF_te_list.append(xf_te_n)

        meta_list.append(np.ones((x_lag_tr_n.shape[0], x_lag_tr_n.shape[1], 1)) * ridx)
        mask_list.append(masks['future_mask'])
        mask_test_list.append(masks['future_mask_test'])

    x_tr = np.concatenate(X_tr_list, axis=0)
    y_tr = np.concatenate(Y_tr_list, axis=0)
    xf_tr = np.concatenate(XF_tr_list, axis=0)

    x_te = np.concatenate(X_te_list, axis=0)
    y_te = np.concatenate(Y_te_list, axis=0)
    xf_te = np.concatenate(XF_te_list, axis=0)

    meta = np.concatenate(meta_list, axis=0)
    future_mask = np.concatenate(mask_list, axis=0)
    future_mask_test = np.concatenate(mask_test_list, axis=0)

    masks = {'future_mask': future_mask, 'future_mask_test': future_mask_test}
    return [x_tr, y_tr, xf_tr, x_te, y_te, xf_te], [x_scalers, y_scalers], masks, meta