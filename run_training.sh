#!/bin/bash
# Quick training script for common workflow

echo "Tomato Disease Detection Training"
echo "Choose training option:"
echo "1) Label tomato detection data"
echo "2) Label disease detection data"  
echo "3) Train tomato detection model"
echo "4) Train disease detection model"
echo "5) Full training pipeline"

read -p "Enter choice (1-5): " choice

case $choice in
    1)
        echo "Starting tomato labeling..."
        python tomato_training_pipeline.py --label-tomato
        ;;
    2)
        read -p "Enter disease name (or press enter for all): " disease
        if [ -z "$disease" ]; then
            python tomato_training_pipeline.py --label-disease
        else
            python tomato_training_pipeline.py --label-disease --disease "$disease"
        fi
        ;;
    3)
        echo "Training tomato detection model..."
        python tomato_training_pipeline.py --train-tomato --epochs 50
        ;;
    4)
        read -p "Enter disease name (or press enter for all): " disease
        if [ -z "$disease" ]; then
            python tomato_training_pipeline.py --train-disease --all-diseases --epochs 80
        else
            python tomato_training_pipeline.py --train-disease --disease "$disease" --epochs 40
        fi
        ;;
    5)
        echo "Running full training pipeline..."
        echo "Step 1: Training tomato detection..."
        python tomato_training_pipeline.py --train-tomato --epochs 50
        
        echo "Step 2: Training disease detection..."
        python tomato_training_pipeline.py --train-disease --all-diseases --epochs 80
        
        echo "Full pipeline complete!"
        ;;
    *)
        echo "Invalid choice"
        ;;
esac
