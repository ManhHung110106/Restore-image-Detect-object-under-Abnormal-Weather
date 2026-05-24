# Usage: .\scripts\run_cross_eval.ps1

$Env:PYTHONPATH = ".;$Env:PYTHONPATH"

Write-Host "Running Cross Evaluation with DCP baseline..."

python src\evaluation\cross_eval.py `
  --patch_size 15 `
  --omega 0.95 `
  --t0 0.1 `
  --method "DCP" `
  --sots_split "configs/reside6k.json" `
  --reside_root "C:\Users\manh hung\.cache\kagglehub\datasets\kmljts\reside-6k\versions\1\RESIDE-6K" `
  --dawn_dir "data/processed/dawn_yolo" `
  --save_output

Write-Host "Done!"
