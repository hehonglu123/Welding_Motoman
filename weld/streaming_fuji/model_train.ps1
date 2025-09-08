# Arrays of candidate inputs
$modelsInputs = @(
"WAAM_NN",    
"WAAM_GRU"
)
# $modelsInputs = @(
# "WAAM_GRU"
# )

# Feature flags (each entry is one set of extra args)
$features = @(
    "--no_feat_neighbor_thermal --no_feat_stickout --no_feat_thermal_x --no_feat_thermal_y --no_feat_x_location",
    "--no_feat_neighbor_thermal --no_feat_x_location",
    "--no_feat_stickout --no_feat_thermal_x --no_feat_thermal_y",
    ""
)
# $features = @(
#     "--no_feat_stickout --no_feat_thermal_x --no_feat_thermal_y",
#     ""
# )

# Path to the Python script
$pythonScript = ".\estimate_dhdw_sequence.py"

# Loop through all combinations
foreach ($model in $modelsInputs) {
    foreach ($feature in $features) {
        $args = @(
            "--train",
            "--model_type", $model,
            "--thermal_emb", 32,
            "--scalar_emb", 16,
            "--epochs", 5000
        )
        if ($model -like "*NN*") {
            $args += @("--nn_hidden_size", 64, "--nn_layers", 0)
        } else {
            $args += @("--nn_hidden_size", 64, "--nn_layers", 1)
        }
        

        # Add features only if non-empty
        if ($feature -ne "") {
            $args += $feature.Split(" ")
        }
        Write-Host "Running with model: $model and features: $feature"
        Write-Host "Command: python $pythonScript $($args -join ' ')"

        python $pythonScript @args
        Write-Host "==========================================="
    }
}

Write-Host "All combinations completed."