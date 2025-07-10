import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def visualize_similar_years(data, labels, dates, n_clusters, frames_per_year=12, cmap='bone'):
    S, T, H, W = data.shape
    dates = pd.to_datetime(dates)
    monthly_indices
