@echo off
echo Tomato Training Pipeline
echo 1) Label tomato data
echo 2) Train tomato model
echo 3) Train disease model
set /p choice=Enter choice (1-3): 
if %choice%==1 python tomato_training_pipeline.py --label-tomato
if %choice%==2 python tomato_training_pipeline.py --train-tomato
if %choice%==3 python tomato_training_pipeline.py --train-disease --all-diseases
pause
