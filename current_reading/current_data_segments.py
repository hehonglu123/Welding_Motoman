import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, peak_widths

# Step 1: Read CSV and get the 'current' column
base_path = '../data/wall_weld_test/316L_model_140ipm_2023_09_27_21_43_22/layer_3/'
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
slice_size = len(extracted_data) // 40

# Initialize a DataFrame to store the results
results_df = pd.DataFrame()

for i in range(0, len(extracted_data), slice_size):
    slice_data = extracted_data[i:i + slice_size]
    slice_peaks, properties = find_peaks(slice_data, height=10)
    
    # Calculate the statistics
    statistics = {
        'Slice mean': np.mean(slice_data),
        'Slice standard deviation': np.std(slice_data),
        'Slice maximum value': np.max(slice_data),
        'Slice minimum value': np.min(slice_data),
    }
    
    if len(slice_peaks):
        statistics.update({
            'Number of peaks counted in the slice': len(slice_peaks),
            'Mean of peak values': np.mean(slice_data[slice_peaks]),
            'Standard deviation of peak values': np.std(slice_data[slice_peaks]),
        })
        
        if 'widths' in properties:
            statistics.update({
                'Mean of peak width': np.mean(properties['widths']),
                'Standard deviation of peak width': np.std(properties['widths']),
            })

        distances = np.diff(slice_peaks)
        if len(distances):
            statistics.update({
                'Mean distance between neighbouring peaks': np.mean(distances),
                'Standard deviation of neighbouring peak distance': np.std(distances),
            })
    
    # Append the results to the DataFrame
    results_df = pd.concat([results_df, pd.DataFrame([statistics])]).reset_index(drop=True)


# Plot each statistic
for col in results_df.columns:
    plt.figure(figsize=(10, 6))
    plt.plot(results_df.index, results_df[col], label=col)
    plt.title(col)
    plt.xlabel('Slice Index')
    plt.ylabel(col)
    plt.legend()
    plt.show()