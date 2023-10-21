# Current Reading Folder

This folder contains scripts and data related to current readings and data analysis algorithm.

## Directory Structure
current_reading/
│
├── analysis.py
├── client.py
├── comparison.py
├── current_acoustic.py
├── current_data_movingwindow.py
├── current_data_segments.py
├── current.csv
├── README.md
├── RR_current_reading.py
└── test.py

## Script Descriptions

- `current_acoustic.py`: Used to display current signals, acoustic signals, and an overview of the height between layers.
- `current_data_movingwindow.py`: Displays the current signals under a moving window with 11 weight values.
- `current_data_segments.py`: Shows the current signals for a specific segment with 11 weight values.
- `RR_current_reading.py`: Script based on the RRdriver to read current values.

## Required Libraries

The scripts utilize several Python libraries, as seen in the screenshots. Ensure that the following libraries are installed:
- `numpy`
- `pandas`
- `matplotlib`
- `scipy`
- `librosa`
- `os`
- `RobotRaconteur`
- `pysmu`

## Additional Notes

- For detailed functionality or configuration related to each script, refer to inline comments or associated documentation.
- Ensure the RRdriver is correctly set up and configured for the `RR_current_reading.py` script.