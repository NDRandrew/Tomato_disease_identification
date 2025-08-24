# TOMATO DISEASE DETECTION - QUICK START

## Getting Started (5 minutes)

### 1. Add Sample Images
```bash
# Add tomato/plant images for detection training
cp your_tomato_images/* tomato_training_project/raw_images/tomato_detection/tomato/
cp your_plant_images/* tomato_training_project/raw_images/tomato_detection/tomato_plant/
cp similar_objects/* tomato_training_project/raw_images/tomato_detection/not_tomato/  # apples, etc.

# Add disease images (organize by disease)
cp healthy_tomatoes/* tomato_training_project/raw_images/disease_detection/healthy/
cp bacterial_spot_images/* tomato_training_project/raw_images/disease_detection/bacterial_spot/
# ... continue for other diseases
```

### 2. Label Your Data (Interactive)
```bash
# Label tomato vs plant detection
python tomato_training_pipeline.py --label-tomato

# Label diseases
python tomato_training_pipeline.py --label-disease --disease bacterial_spot
```

### 3. Train Models
```bash
# Train tomato detection model
python tomato_training_pipeline.py --train-tomato --epochs 50

# Train disease detection (incremental learning)
python tomato_training_pipeline.py --train-disease --disease bacterial_spot --epochs 30
python tomato_training_pipeline.py --train-disease --disease early_blight --incremental --epochs 30
```

### 4. Test Your System
```bash
# Add test images
cp test_images/* input_images/

# Run detection
python tomato_disease_detector.py
```

## Advanced Training Strategy

### Disease Training Order (Recommended):
1. **healthy** (baseline)
2. **bacterial_spot** (common)
3. **early_blight** (common)
4. **late_blight** (severe)
5. **septoria_leaf_spot** (distinctive)
6. Continue with other diseases...

### Training Commands:
```bash
# Train each disease incrementally
python tomato_training_pipeline.py --train-disease --disease healthy --epochs 40
python tomato_training_pipeline.py --train-disease --disease bacterial_spot --incremental --epochs 40
python tomato_training_pipeline.py --train-disease --disease early_blight --incremental --epochs 40
# ... continue for each disease
```

## Monitoring Progress

- **Logs**: Check `logs/` directory for detailed training logs
- **Results**: Training metrics in `tomato_training_project/training_results/`
- **Models**: Best models saved in `tomato_training_project/models/`

## Troubleshooting

### Model Not Loading?
- Check model path in logs
- Ensure model file exists and isn't corrupted
- Try retraining if needed

### Poor Detection Accuracy?
- Add more diverse training images
- Increase training epochs
- Use larger model size (--model-size l)
- Check image quality and labeling accuracy

### Out of Memory Errors?
- Reduce batch size (--batch-size 8)
- Use smaller model (--model-size n)
- Use smaller image size in training

## Support

Check logs in `logs/` directory for detailed error messages and training progress.
