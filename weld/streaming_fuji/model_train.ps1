# Arrays of candidate inputs
$firstInputs = @("RNN", "GRU", "LSTM")
$secondInputs = @("2", "4", "14")
$thirdInputs = @("3", "8", "16")

# Path to the Python script
$pythonScript = ".\estimate_dhdw_sequence.py"

# Loop through all combinations
foreach ($first in $firstInputs) {
    foreach ($second in $secondInputs) {
        foreach ($third in $thirdInputs) {
            # Print the current combination
            Write-Host "Running with inputs: $first, $second, $third"
            
            # Run the Python script with the current inputs
            python $pythonScript $first $second $third
            
            # Optional: Add a small delay between runs if needed
            # Start-Sleep -Seconds 1
        }
    }
}

Write-Host "All combinations completed."