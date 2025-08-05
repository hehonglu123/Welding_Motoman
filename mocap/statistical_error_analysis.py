import numpy as np
import pandas as pd
import pickle
import seaborn as sns
from tabulate import tabulate

import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors

# for plotting
xy_label_size = 14
xy_tick_size = 12
legend_size = 12
title_size = 16
sup_title_size = 18

def load_errors(pickle_file):
    """
    Load errors from a pickle file.
    
    Args:
        pickle_file: Path to pickle file containing error data
        
    Returns:
        Dictionary of errors where keys are method names and values are error arrays
    """
    try:
        with open(pickle_file, 'rb') as f:
            errors = pickle.load(f)
        return errors
    except Exception as e:
        print(f"Error loading pickle file: {e}")
        return None

def perform_statistical_tests(errors):
    """
    Perform paired t-test and Wilcoxon test between each pair of methods.
    
    Args:
        errors: Dictionary of errors
        
    Returns:
        Dataframe containing test results
    """
    methods = list(errors.keys())
    results = {}
    
    for i in range(len(methods)):
        results[methods[i]] = {}
        for j in range(len(methods)):
            if i== j:
                continue
            method1 = methods[i]
            method2 = methods[j]
            
            # Ensure arrays are of the same length
            min_len = min(len(errors[method1]), len(errors[method2]))
            err1 = np.array(errors[method1])[:min_len]
            err2 = np.array(errors[method2])[:min_len]
            
            # Paired t-test
            t_stat, p_t = stats.ttest_rel(err1, err2)
            
            # Wilcoxon signed-rank test
            w_stat, p_w = stats.wilcoxon(err1, err2)
            
            # Check if error difference distribution is normal
            diff = err1 - err2
            _, p_shapiro = stats.shapiro(diff)
            is_normal = p_shapiro > 0.05
            
            # Calculate mean difference
            mean_diff = np.mean(diff)
            
            # Recommended test
            recommended = "t-test" if is_normal else "Wilcoxon"
            
            # results.append({
            #     'Method 1': method1,
            #     'Method 2': method2,
            #     'Mean Diff': round(mean_diff, 4),
            #     't-stat': round(t_stat, 4),
            #     'p-value (t-test)': round(p_t, 4),
            #     'Wilcoxon-stat': round(w_stat, 4),
            #     'p-value (Wilcoxon)': round(p_w, 4),
            #     'Normal Dist?': is_normal,
            #     'Recommended Test': recommended,
            #     'Significant?': (p_t < 0.05 if is_normal else p_w < 0.05)
            # })
            results[methods[i]][methods[j]] = {
                'Mean Diff': round(mean_diff, 4),
                't-stat': round(t_stat, 4),
                'p-value (t-test)': round(p_t, 4),
                'Wilcoxon-stat': round(w_stat, 4),
                'p-value (Wilcoxon)': round(p_w, 4),
                'Normal Dist?': is_normal,
                'Recommended Test': recommended,
                'Significant?': (p_t < 0.05 if is_normal else p_w < 0.05)
            }

    return pd.DataFrame(results)

def plot_error_distributions(errors):
    """
    Plot error difference distributions to check normality.
    
    Args:
        errors: Dictionary of errors
    """
    methods = list(errors.keys())
    n = len(methods)
    
    # Create a grid for plotting
    fig_rows = (n * (n - 1)) // 2 // 2 + ((n * (n - 1)) // 2 % 2)
    fig, axes = plt.subplots(fig_rows, 2, figsize=(15, 4 * fig_rows))
    axes = axes.flatten() if fig_rows > 1 else [axes] if n > 2 else [axes]
    
    plot_idx = 0
    for i in range(len(methods)):
        for j in range(i+1, len(methods)):
            if plot_idx >= len(axes):
                break
                
            method1 = methods[i]
            method2 = methods[j]
            
            # Ensure arrays are of the same length
            min_len = min(len(errors[method1]), len(errors[method2]))
            err1 = np.array(errors[method1])[:min_len]
            err2 = np.array(errors[method2])[:min_len]
            
            diff = err1 - err2
            
            # Plot histogram with KDE
            sns.histplot(diff, kde=True, ax=axes[plot_idx])
            
            # Add a normal distribution reference
            x = np.linspace(np.min(diff), np.max(diff), 100)
            mean, std = np.mean(diff), np.std(diff)
            norm_pdf = stats.norm.pdf(x, mean, std) * len(diff) * (np.max(diff) - np.min(diff)) / 10
            axes[plot_idx].plot(x, norm_pdf, 'r-', lw=2)
            
            # Shapiro-Wilk test
            _, p_shapiro = stats.shapiro(diff)
            is_normal = "Normal" if p_shapiro > 0.05 else "Non-normal"
            
            axes[plot_idx].set_title(f"{method1} vs {method2}\n({is_normal}, p={p_shapiro:.4f})")
            axes[plot_idx].set_xlabel("Error Difference")
            plot_idx += 1
    
    plt.tight_layout()
    plt.savefig("error_difference_distributions.png", dpi=300)
    plt.show()

def main():

    robot_type = 'R1'

    test_data_dir = 'kinematic_raw_data/test0801_R1/' if robot_type == 'R1' else 'kinematic_raw_data/test0804_R2/'

    # Set the path to your pickle file
    pickle_file = test_data_dir+"test_error_pos.pickle"  # Update this path
    
    # Load errors
    errors = load_errors(pickle_file)
    if errors is None:
        return
    
    # load NN and AE errors and add to errors dictionary
    nn_error = np.loadtxt(test_data_dir + "testing_pos_error_NN.csv", delimiter=',')
    ae_error = np.loadtxt(test_data_dir + "testing_pos_error_AE.csv", delimiter=',')
    # errors['NN'] = nn_error
    # errors['AE'] = ae_error

    # Perform statistical tests
    results = perform_statistical_tests(errors)
    
    # Display results as a table
    # show the results in 2 markdown NxN tables format
    # one table show the t statistice and p-values with (t-stats, p-values)
    # The second table show the Wilcoxon statistics and p-values with (Wilcoxon-stat, p-values)
    # print("Statistical Test Results:")
    # for method_1 in results.keys():
    #     for method_2 in results[method_1].keys():
    #         print(f"| {method_1} vs {method_2} | ({results[method_1][method_2]['t-stat']}, {results[method_1][method_2]['p-value (t-test)']}) |")
    #         print(f"| {method_1} vs {method_2} | ({results[method_1][method_2]['Wilcoxon-stat']}, {results[method_1][method_2]['p-value (Wilcoxon)']}) |")

    # Plot error distributions
    # plot_error_distributions(errors)

    # plot error distribution in x y plane
    mocap_T = np.loadtxt(test_data_dir + "mocap_T_align.csv", delimiter=',')
    all_errors = []
    for method in errors.keys():
        print(f"Processing method: {method}")
        assert len(errors[method]) == len(mocap_T), f"Error: Length mismatch for {method} and mocap_T"
        print(errors[method][0])
        if method != 'One PH':
            all_errors.extend(errors[method])
    all_errors = np.array(all_errors)
    max_error = np.max(np.abs(all_errors))
    min_error = np.min(np.abs(all_errors))
    diff_error = max_error - min_error

    start_idx = 610
    end_idx = 1000

    skip_idx = 2
    total_length = len(mocap_T[start_idx:end_idx, 0][::skip_idx])
    marker_size = 150

    fig, ax = plt.subplots(figsize=(10, 5))
    # Color bar Normalization
    norm = colors.Normalize(vmin=min_error, vmax=min_error + diff_error)
    cmap = plt.get_cmap('inferno')
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array(np.array([]))

    ax.grid()
    nominal_errors = np.abs(errors['Nominal'][start_idx:end_idx][::skip_idx])
    cpa_errors = np.abs(errors['CPA'][start_idx:end_idx][::skip_idx])
    fbf_errors = np.abs(errors['Fourier Basis PH'][start_idx:end_idx][::skip_idx])
    # plt.scatter(mocap_T[start_idx:end_idx, 0][::skip_idx],mocap_T[start_idx:end_idx, 1][::skip_idx]+20, c=nominal_errors, cmap=cmap, norm=norm, label='Nominal')
    # plt.scatter(mocap_T[start_idx:end_idx, 0][::skip_idx], mocap_T[start_idx:end_idx, 1][::skip_idx]+10, c=cpa_errors, cmap=cmap, norm=norm, label='CPA')
    # plt.scatter(mocap_T[start_idx:end_idx, 0][::skip_idx], mocap_T[start_idx:end_idx, 1][::skip_idx], c=fbf_errors, cmap=cmap, norm=norm, label='FBF')
    ax.scatter(mocap_T[start_idx:end_idx, 0][::skip_idx], np.ones(total_length)*2.1,c=nominal_errors, cmap=cmap, norm=norm, label='Nominal', s=marker_size)
    ax.scatter(mocap_T[start_idx:end_idx, 0][::skip_idx], np.ones(total_length)*1.1,c=cpa_errors, cmap=cmap, norm=norm, label='CPA', s=marker_size)
    ax.scatter(mocap_T[start_idx:end_idx, 0][::skip_idx], np.ones(total_length)*0.1,c=fbf_errors, cmap=cmap, norm=norm, label='FBF', s=marker_size)
    # Proper colorbar using ScalarMappable
    cbar = plt.colorbar(sm, ax=ax,pad=0.02)
    cbar.set_label('Error Magnitude', fontsize=xy_label_size)         # Title font size
    cbar.ax.tick_params(labelsize=xy_tick_size)
    # Plot aesthetics
    if robot_type == 'R1':
        ax.set_title('MA2010 (R1) Error at Different Configurations (X Position)', fontsize=title_size)
    else:
        ax.set_title('MA1440 (R2) Error at Different Configurations (X Position)', fontsize=title_size)
    ax.set_xlabel('X Position (mm)', fontsize=xy_label_size)
    # ax.set_ylabel('Methods', fontsize=xy_label_size)
    ax.set_ylim(-2, 2.5)
    ax.set_yticks([0, 1, 2], ['FBF', 'CPA', 'Nominal'], fontsize=xy_label_size)
    ax.tick_params(axis='both', which='major', labelsize=xy_tick_size)
    ax.tick_params(axis='y', labelrotation=35)
    plt.tight_layout()
    plt.show()

    
if __name__ == "__main__":
    main()