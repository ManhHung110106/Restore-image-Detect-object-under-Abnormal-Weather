# PowerShell script to run full Optuna tuning
Write-Host "Starting GT Tuning (RESIDE-6K)..."
python src/filters/optuna_tune_gt.py --trials 50 --max_images 1000

Write-Host "Starting No-GT Tuning (DAWN)..."
python src/filters/optuna_tune_nogt.py --trials 50 --max_images 1000

Write-Host "Starting YOLO Tuning (DAWN)..."
python src/detection/optuna_tune_yolo.py --trials 50 --max_images 1000

Write-Host "All tuning finished!"

