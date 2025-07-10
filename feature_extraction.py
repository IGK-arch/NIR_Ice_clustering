import numpy as np
from sklearn.linear_model import LinearRegression


def extract_features_and_targets(data, labels, time_discret=7, N=6):
    S, T, H, W = data.shape
    features = []
    targets = []

    for year_idx in range(S):
        for start_idx in range(N, T):
            history_flatten = data[year_idx, start_idx - N:start_idx].mean(axis=0).flatten()
            history = data[year_idx, start_idx - N:start_idx].mean(axis=(1, 2))

            x = np.arange(N).reshape(-1, 1)
            model = LinearRegression().fit(x, history)
            trend = model.coef_[0]

            mean_value = np.mean(history)
            min_value = np.min(history)
            max_value = np.max(history)

            mean_value_f = np.mean(history_flatten)
            min_value_f = np.min(history_flatten)
            max_value_f = np.max(history_flatten)

            last_step = data[year_idx, start_idx - 1]
            ice_area = np.sum(last_step > 0.5)
            grad_x = np.gradient(last_step, axis=0).mean()
            grad_y = np.gradient(last_step, axis=1).mean()

            features.append([mean_value_f, mean_value, trend, min_value, max_value, ice_area, grad_x, grad_y])
            targets.append(labels[year_idx * T + start_idx])

    return np.array(features), np.array(targets)
