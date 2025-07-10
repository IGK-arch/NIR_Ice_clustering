import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr


def evaluate_metrics(real_data, predicted_data):
    mae_per_step = []
    mse_per_step = []
    ssim_per_step = []
    psnr_per_step = []

    for t in range(real_data.shape[0]):
        real_frame = real_data[t]
        pred_frame = predicted_data[t]

        mae = mean_absolute_error(real_frame.flatten(), pred_frame.flatten())
        mse = mean_squared_error(real_frame.flatten(), pred_frame.flatten())
        mae_per_step.append(mae)
        mse_per_step.append(mse)

        ssim_value = ssim(real_frame, pred_frame, data_range=real_frame.max() - real_frame.min())
        psnr_value = psnr(real_frame, pred_frame, data_range=real_frame.max() - real_frame.min())
        ssim_per_step.append(ssim_value)
        psnr_per_step.append(psnr_value)

    results = {
        "MAE_mean": np.mean(mae_per_step),
        "MSE_mean": np.mean(mse_per_step),
        "SSIM_mean": np.mean(ssim_per_step),
        "PSNR_mean": np.mean(psnr_per_step),
        "MAE_per_step": mae_per_step,
        "MSE_per_step": mse_per_step,
        "SSIM_per_step": ssim_per_step,
        "PSNR_per_step": psnr_per_step,
    }
    return results
