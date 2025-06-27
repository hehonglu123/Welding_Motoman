# Arrays of candidate inputs
# $firstInputs = @("RNN", "GRU", "LSTM", "NARMA")
# $secondInputs = @("2", "4", "14")
# $thirdInputs = @("3", "8", "16", "64")
$firstInputs = @("RNN","NARMA")
$secondInputs = @("2","4")
$thirdInputs = @("3", "8", "16", "64")
$fourthInputs = @("True", "False")

# Path to the Python script
$pythonScript = ".\estimate_dhdw_sequence.py"

# Loop through all combinations
foreach ($first in $firstInputs) {
    foreach ($second in $secondInputs) {
        foreach ($third in $thirdInputs) {
            # If the first input is "NARMA", we need to handle the fourth input differently
            if ($first -eq "NARMA") {
                if ($second -eq "2") {
                    $fourth = "False"  # NARMA with input size 2 does not use open loop
                } else {
                    $fourth = "True"
                }
            }
            python $pythonScript $first $second $third $fourth
        }
    }
}

Write-Host "All combinations completed."