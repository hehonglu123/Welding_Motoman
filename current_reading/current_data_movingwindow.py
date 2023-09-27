import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, peak_widths

# Step 1: Read CSV and get the 'current' column
base_path = '../data/wall_weld_test/70S_model_120ipm_2023_09_23_21_27_03/layer_5/'
filename = 'current.csv'
df = pd.read_csv(base_path + filename)
currents = df['current'].values  # Assuming the title of the column is 'current'

# Step 2: Find peaks
peaks, _ = find_peaks(currents, height=10)  # find all peaks greater than 10

if not len(peaks):
    print("No peaks found!")
    exit()

# find the start and end index of the region of interest
start_index = peaks[0]   # first peak index
end_index = peaks[-1]    # last peak index

# Step 3: Extract the data between start and end index
extracted_data = currents[start_index:end_index + 1]  # extract data between first and last peak

# Optionally: Save extracted data to a new CSV file
extracted_df = pd.DataFrame(extracted_data, columns=['extracted_current'])
extracted_df.to_csv(base_path + 'extracted_current.csv', index=False)

# If you want to view or further process the extracted data, it is stored in 'extracted_data' variable

# Find peaks
peaks, properties = find_peaks(extracted_data, height=10)

# Initialize a DataFrame to store the results
results_df = pd.DataFrame()

# Iterate over each possible window
for i in range(len(extracted_data)):
    window_data = extracted_data[i:i + 10]
    window_peaks, properties = find_peaks(window_data, height=10)
    
    # Calculate peak widths if there are peaks in the window
    if len(window_peaks):
        properties['widths'] = peak_widths(window_data, window_peaks)[0]
    
    # Calculate the statistics
    statistics = {
        'Window mean': np.mean(window_data),
        'Window standard deviation': np.std(window_data),
        'Window maximum value': np.max(window_data),
        'Window minimum value': np.min(window_data),
    }
    
    if len(window_peaks):
        statistics.update({
            'Number of peaks counted in the window': len(window_peaks),
            'Mean of peak values': np.mean(window_data[window_peaks]),
            'Standard deviation of peak values': np.std(window_data[window_peaks]),
            'Mean of peak width': np.mean(properties['widths']) if 'widths' in properties else np.nan,
            'Standard deviation of peak width': np.std(properties['widths']) if 'widths' in properties else np.nan,
            'Mean distance between neighbouring peaks': np.mean(np.diff(window_peaks)) if len(window_peaks) > 1 else np.nan,
            'Standard deviation of neighbouring peak distance': np.std(np.diff(window_peaks)) if len(window_peaks) > 1 else np.nan,
        })

    # Append the results to the DataFrame
    results_df = pd.concat([results_df, pd.DataFrame([statistics])]).reset_index(drop=True)


# Plot each statistic
for col in results_df.columns:
    plt.figure(figsize=(10, 6))
    plt.plot(results_df.index, results_df[col], label=col)
    plt.title(col)
    plt.xlabel('Window Start Index')
    plt.ylabel(col)
    plt.legend()
    plt.show()
