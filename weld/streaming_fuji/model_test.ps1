# Get the path to the weld_Seq_models directory
$modelsPath = Join-Path -Path $PSScriptRoot -ChildPath "weld_Seq_models"

# Check if the directory exists
if (-not (Test-Path -Path $modelsPath -PathType Container)) {
    Write-Error "The directory 'weld_Seq_models' does not exist."
    exit 1
}

# Get all folders in the weld_Seq_models directory
# $folders = Get-ChildItem -Path $modelsPath -Directory
$folders = @("model_20250715_151005", "model_20250715_151228", "model_20250715_151436", "model_20250715_151650")

# Check if any folders were found
if ($folders.Count -eq 0) {
    Write-Warning "No folders found in 'weld_Seq_models'."
    exit 0
}

# Loop through each folder and run the Python script
foreach ($folder in $folders) {
    # $folderName = $folder.Name
    $folderName = $folder
    Write-Host "Processing folder: $folderName"
    
    # Run the Python script with the folder name as an argument
    python estimate_dhdw_sequence.py --load_pretrained --load_model_dir $folderName --multi_steps 10
    
    # Check if the Python script executed successfully
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "Python script failed for folder: $folderName"
    }

    Write-Host "=========================================="
}

Write-Host "Processing complete."