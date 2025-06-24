import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

inch2mm = 25.4 # conversion factor from inch to mm
mm2inch = 1 / inch2mm # conversion factor from mm to inch

# for plotting
xy_label_size = 14
xy_tick_size = 12
legend_size = 12
title_size = 16
sup_title_size = 18

def plot_vt_vw_dh_dw(v_torch_range, v_wire_range, dh_pred, dw_pred, plot_title='Predicted dh and dw'):

    # make the v_torch_range and v_wire_range 2D arrays for imshow
    v_torch_range_grid = np.repeat(v_torch_range, len(v_wire_range)).reshape(len(v_torch_range), len(v_wire_range))
    v_wire_range_grid = np.tile(v_wire_range, len(v_torch_range)).reshape(len(v_torch_range), len(v_wire_range))
    v_wire_range_ipm = v_wire_range * mm2inch * 60 # convert mm/s to ipm

    
    # plot dh and dw using colormap and imshow
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    # dh
    sc = ax[0].scatter(v_torch_range_grid, v_wire_range_grid, c=dh_pred, cmap='viridis', marker='o')
    ax[0].set_xlabel('Torch Speed (mm/s)', fontsize=xy_label_size)
    ax[0].tick_params(axis='x', labelsize=xy_tick_size)
    ax[0].set_ylabel('Wire Feedrate (ipm)', fontsize=xy_label_size)
    ax[0].set_yticks(np.arange(min(v_wire_range_ipm), max(v_wire_range_ipm)+1, 20) * inch2mm / 60)
    ax[0].set_yticklabels(np.arange(min(v_wire_range_ipm), max(v_wire_range_ipm)+1, 20).astype(int), fontsize=xy_tick_size)
    ax[0].set_title('Predicted $\Delta h$', fontsize=title_size)
    ax[0].set_aspect('auto')
    ax[0].grid()
    cbar = plt.colorbar(sc, ax=ax[0])
    cbar.set_label('$\Delta h$ (mm)', fontsize=xy_label_size)
    cbar.ax.tick_params(labelsize=xy_tick_size)
    # dw
    sc = ax[1].scatter(v_torch_range_grid, v_wire_range_grid, c=dw_pred, cmap='viridis', marker='o')
    ax[1].set_xlabel('Torch Speed (mm/s)', fontsize=xy_label_size)
    ax[1].tick_params(axis='x', labelsize=xy_tick_size)
    ax[1].set_ylabel('Wire Feedrate (ipm)', fontsize=xy_label_size)
    ax[1].set_yticks(np.arange(min(v_wire_range_ipm), max(v_wire_range_ipm)+1, 20) * inch2mm / 60)
    ax[1].set_yticklabels(np.arange(min(v_wire_range_ipm), max(v_wire_range_ipm)+1, 20).astype(int), fontsize=xy_tick_size)
    ax[1].set_title('Predicted $\Delta w$', fontsize=title_size)
    ax[1].set_aspect('auto')
    ax[1].grid()
    cbar = plt.colorbar(sc, ax=ax[1])
    cbar.set_label('$\Delta w$ (mm)', fontsize=xy_label_size)
    cbar.ax.tick_params(labelsize=xy_tick_size)
    plt.suptitle(plot_title, fontsize=sup_title_size)
    plt.tight_layout()
    plt.show()

def plot_error_distribution_all(dh_error_all, dw_error_all, plot_title='Error Distribution'):

    # plot the error distribution
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    cmap = plt.get_cmap('tab10')

    for i,key in enumerate(dh_error_all.keys()):
        dh_lambda_hat = 1 / np.mean(dh_error_all[key])
        dh_exp_dist = stats.expon(scale=1/dh_lambda_hat)
        dw_lambda_hat = 1 / np.mean(dw_error_all[key])
        dw_exp_dist = stats.expon(scale=1/dw_lambda_hat)

        ax[0].hist(dh_error_all[key], bins=50, density=True, alpha=0.5, label=key, color=cmap(i))
        ax[0].plot(np.linspace(0, np.max(dh_error_all[key]), 100), dh_exp_dist.pdf(np.linspace(0, np.max(dh_error_all[key]), 100)), label=key+' pdf', color=cmap(i))
        ax[0].set_xlabel('$\Delta h$ Error (mm)', fontsize=xy_label_size)
        ax[0].set_ylabel('Density', fontsize=xy_label_size)
        ax[0].set_title('$\Delta h$ Error Distribution', fontsize=title_size)
        ax[0].legend(fontsize=legend_size)
        ax[0].tick_params(axis='x', labelsize=xy_tick_size)
        ax[0].tick_params(axis='y', labelsize=xy_tick_size)
        ax[0].grid()

        ax[1].hist(dw_error_all[key], bins=50, density=True, alpha=0.5, label=key, color=cmap(i))
        ax[1].plot(np.linspace(0, np.max(dw_error_all[key]), 100), dw_exp_dist.pdf(np.linspace(0, np.max(dw_error_all[key]), 100)), label=key+' pdf', color=cmap(i))
        ax[1].set_xlabel('$\Delta w$ Error (mm)', fontsize=xy_label_size)
        ax[1].set_ylabel('Density', fontsize=xy_label_size)
        ax[1].set_title('$\Delta w$ Error Distribution', fontsize=title_size)
        ax[1].legend(fontsize=legend_size)
        ax[1].tick_params(axis='x', labelsize=xy_tick_size)
        ax[1].tick_params(axis='y', labelsize=xy_tick_size)
        ax[1].grid()
    
    ax[0].set_xlim(-0.2, 1.2)
    ax[1].set_xlim(-0.2, 2.2)
    plt.suptitle(plot_title, fontsize=sup_title_size)
    plt.tight_layout()
    plt.show()

def plot_error_distribution(dh_train_error, dw_train_error, dh_val_error, dw_val_error, plot_title='Error Distribution'):

    # fit distribution with exponential distribution
    dh_lambda_hat_train = 1 / np.mean(dh_train_error)
    dh_exp_dist_train = stats.expon(scale=1/dh_lambda_hat_train)
    dh_lambda_hat_val = 1 / np.mean(dh_val_error)
    dh_exp_dist_val = stats.expon(scale=1/dh_lambda_hat_val)
    dw_lambda_hat_train = 1 / np.mean(dw_train_error)
    dw_exp_dist_train = stats.expon(scale=1/dw_lambda_hat_train)
    dw_lambda_hat_val = 1 / np.mean(dw_val_error)
    dw_exp_dist_val = stats.expon(scale=1/dw_lambda_hat_val)
    # fit distribution with half-normal distribution
    # dh_loc_hat, dh_sigma_hat = stats.halfnorm.fit(dh_train_error, floc=0)
    # dh_halfnorm_dist = stats.halfnorm(loc=dh_loc_hat, scale=dh_sigma_hat)
    # dw_loc_hat, dw_sigma_hat = stats.halfnorm.fit(dw_train_error, floc=0)
    # dw_halfnorm_dist = stats.halfnorm(loc=dw_loc_hat, scale=dw_sigma_hat)

    # plot the error distribution
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    # dh train error
    ax[0].hist(dh_train_error, bins=50, density=True, alpha=0.5, label='Train')
    # dh validation error
    ax[0].hist(dh_val_error, bins=50, density=True, alpha=0.5, label='Test')
    ax[0].plot(np.linspace(0, np.max(dh_train_error), 100), dh_exp_dist_train.pdf(np.linspace(0, np.max(dh_train_error), 100)), label='Train pdf')
    # ax[0].plot(np.linspace(0, np.max(dh_train_error), 100), dh_halfnorm_dist.pdf(np.linspace(0, np.max(dh_train_error), 100)), label='Half-Normal Fit')
    ax[0].plot(np.linspace(0, np.max(dh_train_error), 100), dh_exp_dist_val.pdf(np.linspace(0, np.max(dh_train_error), 100)), label='Test pdf')
    ax[0].set_xlabel('$\Delta h$ Error (mm)', fontsize=xy_label_size)
    ax[0].set_ylabel('Density', fontsize=xy_label_size)
    ax[0].set_title('$\Delta h$ Error Distribution', fontsize=title_size)
    ax[0].legend(fontsize=legend_size)
    ax[0].tick_params(axis='x', labelsize=xy_tick_size)
    ax[0].tick_params(axis='y', labelsize=xy_tick_size)
    ax[0].grid()
    # dw train error 
    ax[1].hist(dw_train_error, bins=50, density=True, alpha=0.5, label='Train')
    # dw validation error
    ax[1].hist(dw_val_error, bins=50, density=True, alpha=0.5, label='Test')
    ax[1].plot(np.linspace(0, np.max(dw_train_error), 100), dw_exp_dist_train.pdf(np.linspace(0, np.max(dw_train_error), 100)), label='Train pdf')
    # ax[1].plot(np.linspace(0, np.max(dw_train_error), 100), dw_halfnorm_dist.pdf(np.linspace(0, np.max(dw_train_error), 100)), label='Half-Normal Fit')
    ax[1].plot(np.linspace(0, np.max(dw_train_error), 100), dw_exp_dist_val.pdf(np.linspace(0, np.max(dw_train_error), 100)), label='Test pdf')
    ax[1].set_xlabel('$\Delta w$ Error (mm)', fontsize=xy_label_size)
    ax[1].set_ylabel('Density', fontsize=xy_label_size)
    ax[1].set_title('$\Delta w$ Error Distribution', fontsize=title_size)
    ax[1].legend(fontsize=legend_size)
    ax[1].grid()
    ax[1].tick_params(axis='x', labelsize=xy_tick_size)
    ax[1].tick_params(axis='y', labelsize=xy_tick_size)
    plt.suptitle(plot_title, fontsize=sup_title_size)
    plt.tight_layout()
    plt.show()

    return dh_exp_dist_train.interval(0.95), dw_exp_dist_train.interval(0.95)

def plot_error_heatmap(dh_train_error, dw_train_error, dh_val_error, dw_val_error, train_inputs, val_inputs, plot_title='Error Heatmap'):

    dh_error_all = np.abs(np.concatenate((dh_train_error, dh_val_error)))
    dw_error_all = np.abs(np.concatenate((dw_train_error, dw_val_error)))
    inputs_all = np.concatenate((train_inputs, val_inputs))
    dw_inputs_all_ipm = inputs_all[:, 1] * mm2inch * 60 # convert mm/s to ipm

    ## 3D bar error heatmap
    # fig = plt.figure(1, 2)
    # ax = fig.add_subplot(111, projection='3d')
    # # ax.bar3d(inputs_all[:, 0], dw_inputs_all_ipm, np.zeros_like(dh_error_all), 0.2, 0.2, dh_error_all, shade=True)
    # # use error as colormap with bar3d
    # ax.bar3d(inputs_all[:, 0], inputs_all[:, 1], np.zeros_like(dh_error_all), 0.2, 0.2, dh_error_all, shade=True, color=plt.cm.viridis(dh_error_all/np.max[0](dh_error_all)))
    # ax.set_xlabel('Torch Speed (mm/s)', fontsize=xy_label_size)
    # ax.set_ylabel('Wire Feedrate (ipm)', fontsize=xy_label_size)
    # ax.set_zlabel('$\Delta h$ Error (mm)', fontsize=xy_label_size)
    # ax.set_title('$\Delta h$ Error Heatmap', fontsize=title_size)
    # ax.set_xticks(np.arange(min(inputs_all[:, 0]), max(inputs_all[:, 0])+1, 2))
    # ax.set_yticks(np.arange(min(dw_inputs_all_ipm), max(dw_inputs_all_ipm)+1, 20) * inch2mm / 60)
    # ax.set_yticklabels(np.arange(min(dw_inputs_all_ipm), max(dw_inputs_all_ipm)+1, 20).astype(int), fontsize=xy_tick_size)
    # ax.set_zticks(np.arange(0, np.max(dh_error_all)+1, 0.5))
    # ax.set_zticklabels(np.arange(0, np.max(dh_error_all)+1, 0.5).astype(int), fontsize=xy_tick_size)
    # ax.set_box_aspect([1, 1, 0.5])  # aspect ratio is 1:1:0.5
    # plt.title(plot_title, fontsize=sup_title_size)
    # plt.tight_layout()
    # plt.show()

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].scatter(inputs_all[:, 0], inputs_all[:, 1], c=dh_error_all, cmap='viridis', marker='o')
    ax[0].set_xlabel('Torch Speed (mm/s)', fontsize=xy_label_size)
    ax[0].set_ylabel('Wire Feedrate (ipm)', fontsize=xy_label_size)
    ax[0].set_yticks(np.arange(min(dw_inputs_all_ipm), max(dw_inputs_all_ipm)+1, 20) * inch2mm / 60)
    ax[0].set_yticklabels(np.arange(min(dw_inputs_all_ipm), max(dw_inputs_all_ipm)+1, 20).astype(int), fontsize=xy_tick_size)
    ax[0].set_title('$\Delta h$ Error Heatmap', fontsize=title_size)
    ax[0].set_aspect('auto')
    ax[0].grid()
    cbar = plt.colorbar(ax[0].collections[0], ax=ax[0])
    cbar.set_label('$\Delta h$ Error (mm)', fontsize=xy_label_size)
    cbar.ax.tick_params(labelsize=xy_tick_size)
    # dw
    ax[1].scatter(inputs_all[:, 0], inputs_all[:, 1], c=dw_error_all, cmap='viridis', marker='o')
    ax[1].set_xlabel('Torch Speed (mm/s)', fontsize=xy_label_size)
    ax[1].set_ylabel('Wire Feedrate (ipm)', fontsize=xy_label_size)
    ax[1].set_yticks(np.arange(min(dw_inputs_all_ipm), max(dw_inputs_all_ipm)+1, 20) * inch2mm / 60)
    ax[1].set_yticklabels(np.arange(min(dw_inputs_all_ipm), max(dw_inputs_all_ipm)+1, 20).astype(int), fontsize=xy_tick_size)
    ax[1].set_title('$\Delta w$ Error Heatmap', fontsize=title_size)
    ax[1].set_aspect('auto')
    ax[1].grid()
    cbar = plt.colorbar(ax[1].collections[0], ax=ax[1])
    cbar.set_label('$\Delta w$ Error (mm)', fontsize=xy_label_size)
    cbar.ax.tick_params(labelsize=xy_tick_size)
    plt.suptitle(plot_title, fontsize=sup_title_size)
    plt.tight_layout()
    plt.show()

def get_rmse(error_array):

    error_array = np.array(error_array).flatten()

    assert error_array.ndim == 1, "Error array must be 1D"
    assert error_array.size > 0, "Error array must not be empty"

    return np.sqrt(np.mean(error_array**2))

def get_mean_std(error_array):

    error_array = np.fabs(np.array(error_array).flatten())

    assert error_array.ndim == 1, "Error array must be 1D"
    assert error_array.size > 0, "Error array must not be empty"

    mean_error = np.mean(error_array)
    std_error = np.std(error_array)

    return [mean_error, std_error]
