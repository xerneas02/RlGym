# Get the list of folders in the logs directory with names starting with "PPO_"
$ppoFolders = Get-ChildItem -Path .\logs\ -Directory | Where-Object { $_.Name -like "PPO_*" }

# Find the folder with the highest number
$highestNumber = 0
foreach ($folder in $ppoFolders) {
    $number = [int]($folder.Name -replace 'PPO_', '')
    if ($number -gt $highestNumber) {
        $highestNumber = $number
    }
}

# Construct the log directory path with the highest number
$logDir = ".\logs\PPO_$highestNumber"

# Execute the tensorboard command
Start-Process -FilePath "tensorboard" -ArgumentList "--logdir $logDir"
