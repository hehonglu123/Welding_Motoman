import numpy as np
from scipy import stats
import os

def read_and_calculate_mse(model_dir, multisteps=10):
    # Read the CSV files
    testing_loss = np.loadtxt(os.path.join(model_dir, 'testing_loss.csv'), delimiter=',')
    test_dh_error = np.loadtxt(os.path.join(model_dir, 'test_dh_error.csv'), delimiter=',')
    test_dw_error = np.loadtxt(os.path.join(model_dir, 'test_dw_error.csv'), delimiter=',')
    
    # Check shapes
    print(f"Testing Loss shape: {testing_loss.shape}")
    print(f"Test DH Error shape: {test_dh_error.shape}")
    print(f"Test DW Error shape: {test_dw_error.shape}")
    
    # Determine batch size
    batch_size = test_dh_error.shape[0] // (40 - multisteps)
    
    # Reshape errors to (batch, 40, 1) each
    reshaped_dh = test_dh_error.reshape(batch_size, 40-multisteps, 1)
    reshaped_dw = test_dw_error.reshape(batch_size, 40-multisteps, 1)
    
    # Combine to (batch, 40, 2)
    combined_error = np.abs(np.concatenate([reshaped_dh, reshaped_dw], axis=2))

    # ignore the first 10 steps for MSE calculation
    # combined_error = combined_error[:, 10:, :]  # shape (batch, 30, 2)
    
    # Calculate MSE
    mse = np.mean(combined_error ** 2)
    
    return testing_loss, combined_error, mse

if __name__ == "__main__":
    # "model_20250715_151005", "model_20250715_151228", "model_20250715_151436", "model_20250715_151650"
    model_dir = "model_20250715_151650/"  # Current directory, change as needed 
    # model_dir = "model_20250715_122716/"  # Current directory, change as needed
    testing_loss, combined_error, mse = read_and_calculate_mse(model_dir,multisteps=10)

    print(f"Min testing loss: {np.min(testing_loss):.4f}")
    print(f"Calculated MSE: {mse:.4f}")

    dh_mean_error = np.mean(combined_error[:, :, 0])
    dh_95_error = stats.expon(scale=np.std(combined_error[:, :, 0])).interval(0.95)[1]
    dh_max_error = np.max(combined_error[:, :, 0])
    dw_mean_error = np.mean(combined_error[:, :, 1])
    dw_95_error = stats.expon(scale=np.std(combined_error[:, :, 1])).interval(0.95)[1]
    dw_max_error = np.max(combined_error[:, :, 1])

    print(f"dh (mean, 95% CI, max): ({dh_mean_error:.4f}, {dh_95_error:.4f}, {dh_max_error:.4f})")
    print(f"dw (mean, 95% CI, max): ({dw_mean_error:.4f}, {dw_95_error:.4f}, {dw_max_error:.4f})")