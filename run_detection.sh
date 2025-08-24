#!/bin/bash
# Quick detection script

echo "Running Tomato Disease Detection..."
echo "Processing images from input_images/"

python tomato_disease_detector.py

echo "Detection complete!"
echo "Results saved to tomato_detection_results/"
echo "Logs saved to logs/"
