# Get the path to the weld_Seq_models directory
$modelsPath = Join-Path -Path $PSScriptRoot -ChildPath "weld_Seq_models"

# Check if the directory exists
if (-not (Test-Path -Path $modelsPath -PathType Container)) {
    Write-Error "The directory 'weld_Seq_models' does not exist."
    exit 1
}

# Get all folders in the weld_Seq_models directory
$folders = Get-ChildItem -Path $modelsPath -Directory

# Check if any folders were found
if ($folders.Count -eq 0) {
    Write-Warning "No folders found in 'weld_Seq_models'."
    exit 0
}

# Loop through each folder and run the Python script
foreach ($folder in $folders) {
    $folderName = $folder.Name
    Write-Host "Processing folder: $folderName"
    
    # Run the Python script with the folder name as an argument
    python estimate_dhdw_sequence.py false $folderName
    
    # Check if the Python script executed successfully
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "Python script failed for folder: $folderName"
    }

    Write-Host "=========================================="
}

Write-Host "Processing complete."