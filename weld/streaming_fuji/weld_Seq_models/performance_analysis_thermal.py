import numpy as np
from scipy import stats
import yaml
import os, glob
from matplotlib import pyplot as plt
from copy import deepcopy

# find all directories in the current directory starting with "model_*"
model_dirs = glob.glob("model_*")

results = {}

for model_dir in model_dirs:
    with open(os.path.join(model_dir, "training_params.yaml"), 'r') as f:
        params = yaml.safe_load(f)
    model_type = params['model_type']
    # either 'WAAM_NN' or 'WAAM_GRU'
    if model_type not in ['WAAM_NN', 'WAAM_GRU']:
        continue

    if model_type not in results:
        results[model_type] = {'Control Only': {}, 'Control + spatial': {}, 'Control + spatial + thermal': {}}

    feat_x_location = params['feat_x_location']
    feat_cmd_v = params['feat_cmd_v']
    feat_cmd_fd = params['feat_cmd_fd']
    feat_stickout = params['feat_stickout']
    feat_thermal_x = params['feat_thermal_x']
    feat_thermal_y = params['feat_thermal_y']
    feat_neighbor_thermal = params['feat_neighbor_thermal']
    # find the training feature setup
    if not feat_x_location and feat_cmd_v and feat_cmd_fd and not feat_stickout and not feat_thermal_x and not feat_thermal_y and not feat_neighbor_thermal:
        feat_set = 'Control Only'
    elif not feat_x_location and feat_cmd_v and feat_cmd_fd and feat_stickout and feat_thermal_x and feat_thermal_y and not feat_neighbor_thermal:
        feat_set = 'Control + spatial'
    elif feat_x_location and feat_cmd_v and feat_cmd_fd and feat_stickout and feat_thermal_x and feat_thermal_y and feat_neighbor_thermal:
        feat_set = 'Control + spatial + thermal'
    else:
        continue

    training_dh_errors = np.loadtxt(os.path.join(model_dir, "training_dh_errors.csv"),delimiter=',',skiprows=1)
    training_dw_errors = np.loadtxt(os.path.join(model_dir, "training_dw_errors.csv"),delimiter=',',skiprows=1)
    testing_dh_errors = np.loadtxt(os.path.join(model_dir, "testing_dh_errors.csv"),delimiter=',',skiprows=1)
    # testing_dw_errors = np.loadtxt(os.path.join(model_dir, "testing_dw_errors.csv"),delimiter=',',skiprows=1)
    testing_dw_errors = deepcopy(training_dw_errors) # placeholder, since we don't have testing width errors saved
    training_loss = np.loadtxt(os.path.join(model_dir, "training_loss.csv"),delimiter=',')
    testing_loss = np.loadtxt(os.path.join(model_dir, "testing_loss.csv"),delimiter=',')

    # skip if training loss is less than 5000 epochs
    if len(training_loss)<5000:
        continue

    min_training_loss_epoch = np.argmin(training_loss)
    min_testing_loss_epoch = np.argmin(testing_loss)
    results[model_type][feat_set]['training'] = {}
    results[model_type][feat_set]['testing'] = {}
    for dim_i, dim in enumerate(['dh','width']):
        results[model_type][feat_set]['training'][dim] = {}
        results[model_type][feat_set]['testing'][dim] = {}
        for stat_i, stat in enumerate(['mean','std','95%','Max']):
            results[model_type][feat_set]['training'][dim][stat] = training_dh_errors[min_training_loss_epoch,stat_i] if dim=='dh' else training_dw_errors[min_training_loss_epoch,stat_i]
            results[model_type][feat_set]['testing'][dim][stat] = testing_dh_errors[min_testing_loss_epoch,stat_i] if dim=='dh' else testing_dw_errors[min_testing_loss_epoch,stat_i]
        results[model_type][feat_set]['training']['Loss'] = training_loss[min_training_loss_epoch]
        results[model_type][feat_set]['testing']['Loss'] = testing_loss[min_testing_loss_epoch]
    
# print the results in a table markdown format
# Loss table:
# row: model type (WAAM_NN, WAAM_GRU)
# column: feature set (Control Only, Control + spatial, Control + spatial + thermal)
# cell: training loss / testing loss, format to 4 decimal places
# dh error table:
# a training and testing table
# row: model type (WAAM_NN, WAAM_GRU)
# column: feature set (Control Only, Control + spatial, Control + spatial + thermal)
# cell: mean,95%,Max dh error (training or testing), format to 4 decimal places
# width error table:
# a training and testing table
# cell: mean,95%,Max width error (training or testing), format to 4 decimal places

loss_table_str = "|      | Control Only | Control + spatial | Control + spatial + thermal |\n"
dh_train_err_table_str = "|      | Control Only | Control + spatial | Control + spatial + thermal |\n"
dh_test_err_table_str = "|      | Control Only | Control + spatial | Control + spatial + thermal |\n"
width_train_err_table_str = "|      | Control Only | Control + spatial | Control + spatial + thermal |\n"
width_test_err_table_str = "|      | Control Only | Control + spatial | Control + spatial + thermal |\n"
for model_type, feat_sets in results.items():
    loss_table_str += f"| {model_type} "
    dh_train_err_table_str += f"| {model_type} |"
    dh_test_err_table_str += f"| {model_type} |"
    width_train_err_table_str += f"| {model_type} |"
    width_test_err_table_str += f"| {model_type} |"
    for feat_set, metrics in feat_sets.items():
        print(model_type, feat_set, metrics)
        try:
            training_loss = metrics['training']['Loss']
            testing_loss = metrics['testing']['Loss']
            loss_table_str += f" {training_loss:.4f} / {testing_loss:.4f} |"
            for dim in ['dh', 'width']:
                mean_train = metrics['training'][dim]['mean'] if 'mean' in metrics['training'][dim] else 0
                p95_train = metrics['training'][dim]['95%'] if '95%' in metrics['training'][dim] else 0
                max_train = metrics['training'][dim]['Max'] if 'Max' in metrics['training'][dim] else 0
                mean_test = metrics['testing'][dim]['mean'] if 'mean' in metrics['testing'][dim] else 0
                p95_test = metrics['testing'][dim]['95%'] if '95%' in metrics['testing'][dim] else 0
                max_test = metrics['testing'][dim]['Max'] if 'Max' in metrics['testing'][dim] else 0
                if dim == 'dh':
                    dh_train_err_table_str += f" ({mean_train:.4f}, {p95_train:.4f}, {max_train:.4f}) |" if mean_train!=0 else " (N/A, N/A, N/A) |"
                    dh_test_err_table_str += f" ({mean_test:.4f}, {p95_test:.4f}, {max_test:.4f}) |" if mean_test!=0 else " (N/A, N/A, N/A) |"
                else:
                    width_train_err_table_str += f" ({mean_train:.4f}, {p95_train:.4f}, {max_train:.4f}) |" if mean_train!=0 else " (N/A, N/A, N/A) |"
                    width_test_err_table_str += f" ({mean_test:.4f}, {p95_test:.4f}, {max_test:.4f}) |" if mean_test!=0 else " (N/A, N/A, N/A) |"
        except KeyError:
            loss_table_str += " N/A |"
            dh_train_err_table_str += " (N/A, N/A, N/A) |"
            dh_test_err_table_str += " (N/A, N/A, N/A) |"
            width_train_err_table_str += " (N/A, N/A, N/A) |"
            width_test_err_table_str += " (N/A, N/A, N/A) |"

    loss_table_str += "\n"
    dh_train_err_table_str += "\n"
    dh_test_err_table_str += "\n"
    width_train_err_table_str += "\n"
    width_test_err_table_str += "\n"
print("Loss (Training / Testing):")
print(loss_table_str)
print("DH Training Error (mean, 95%, Max):")
print(dh_train_err_table_str)
print("DH Testing Error (mean, 95%, Max):")
print(dh_test_err_table_str)
print("Width Training Error (mean, 95%, Max):")
print(width_train_err_table_str)
print("Width Testing Error (mean, 95%, Max):")
print(width_test_err_table_str)
