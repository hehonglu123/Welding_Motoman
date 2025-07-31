import numpy as np
import pandas as pd
import pickle
import seaborn as sns
from tabulate import tabulate

import scipy.stats as stats
import matplotlib.pyplot as plt

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
    results = []
    
    for i in range(len(methods)):
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
            
            results.append({
                'Method 1': method1,
                'Method 2': method2,
                'Mean Diff': round(mean_diff, 4),
                't-stat': round(t_stat, 4),
                'p-value (t-test)': round(p_t, 4),
                'Wilcoxon-stat': round(w_stat, 4),
                'p-value (Wilcoxon)': round(p_w, 4),
                'Normal Dist?': is_normal,
                'Recommended Test': recommended,
                'Significant?': (p_t < 0.05 if is_normal else p_w < 0.05)
            })
    
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

    test_data_dir = 'kinematic_raw_data/test0801_R1/'
    # test_data_dir = 'kinematic_raw_data/test0804_R2/'

    # Set the path to your pickle file
    pickle_file = test_data_dir+"test_error_pos.pickle"  # Update this path
    
    # Load errors
    errors = load_errors(pickle_file)
    if errors is None:
        return
    
    # load NN and AE errors and add to errors dictionary
    nn_error = np.loadtxt(test_data_dir + "testing_pos_error_NN.csv", delimiter=',')
    ae_error = np.loadtxt(test_data_dir + "testing_pos_error_AE.csv", delimiter=',')
    errors['NN'] = nn_error
    errors['AE'] = ae_error

    # Perform statistical tests
    results = perform_statistical_tests(errors)
    
    # Display results as a table
    print("Statistical Test Results:")
    # print(tabulate(results, headers='keys', tablefmt='pretty'))

    # show the results in 2 markdown NxN tables format
    # one table show the t statistice and p-values with (t-stats, p-values)
    # The second table show the Wilcoxon statistics and p-values with (Wilcoxon-stat, p-values)
    

    # Save results to CSV
    # results.to_csv("statistical_test_results.csv", index=False)
    
    # Plot error distributions
    # plot_error_distributions(errors)
    
if __name__ == "__main__":
    main()