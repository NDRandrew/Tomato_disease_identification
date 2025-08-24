# Tomato Disease Detection Training Project

## 🚀 Quick Start Guide

### Phase 1: Tomato/Plant Detection Training

#### 1. Add Tomato Detection Images
```
raw_images/tomato_detection/
├── tomato/           # Tomato fruit images
├── tomato_plant/     # Tomato plant images
└── not_tomato/       # Similar-looking objects (apples, other plants, etc.)
```

#### 2. Label Tomato Detection Data
```bash
python tomato_training_pipeline.py --label-tomato
```

#### 3. Train Tomato Detection Model
```bash
python tomato_training_pipeline.py --train-tomato --epochs 100
```

### Phase 2: Disease Detection Training (Incremental)

#### 1. Add Disease Images
```
raw_images/disease_detection/
├── healthy/
├── bacterial_spot/
├── early_blight/
├── septoria_leaf_spot/
└── ... (other diseases)
```

#### 2. Train Disease Model Incrementally
```bash
# Train on first disease
python tomato_training_pipeline.py --train-disease --disease bacterial_spot --epochs 50

# Add next disease
python tomato_training_pipeline.py --train-disease --disease early_blight --epochs 50 --incremental

# Continue for all diseases...
```

#### 3. Train All Diseases Together (Alternative)
```bash
python tomato_training_pipeline.py --train-disease --all-diseases --epochs 100
```

### Testing Models
```bash
# Test tomato detection
python tomato_training_pipeline.py --test-tomato --model models/tomato_detection_best.pt

# Test disease detection  
python tomato_training_pipeline.py --test-disease --model models/disease_detection_best.pt
```

## 📁 Project Structure

### Raw Images
- `raw_images/tomato_detection/tomato/`: Tomato fruit images
- `raw_images/tomato_detection/tomato_plant/`: Tomato plant images  
- `raw_images/tomato_detection/not_tomato/`: Similar-looking objects (apples, other fruits, peppers, etc.)
- `raw_images/disease_detection/`: Organized by disease folders

### Labeled Data
- `labeled_data/tomato_detection/`: Tomato detection annotations
- `labeled_data/disease_detection/`: Disease detection annotations

### Models
- `models/tomato_detection_best.pt`: Best tomato detection model
- `models/disease_detection_best.pt`: Best disease detection model

### Results
- `training_results/`: Training logs and metrics
- `disease_progress/`: Incremental disease training progress

## 🦠 Supported Diseases

1. healthy
2. bacterial_spot
3. early_blight
4. late_blight
5. leaf_mold
6. septoria_leaf_spot
7. spider_mites
8. target_spot
9. yellow_leaf_curl_virus
10. mosaic_virus
11. bacterial_canker

## 💡 Tips for Better Training

### Image Quality
- Use high-resolution images (minimum 640x640)
- Ensure good lighting and focus
- Include variety of angles and conditions

### Data Balance
- Aim for balanced classes (similar number of examples per disease)
- Include negative examples (healthy plants)

### Incremental Training
- Start with most common diseases
- Train one disease at a time for better learning
- Test after each disease addition

## 🔧 Advanced Usage

### Custom Training Parameters
```bash
# Large model with more epochs
python tomato_training_pipeline.py --train-tomato --model-size l --epochs 200 --batch-size 8

# High confidence threshold
python tomato_training_pipeline.py --train-disease --confidence 0.8
```

### Resume Training
```bash
# Resume from checkpoint
python tomato_training_pipeline.py --train-disease --resume models/disease_detection_last.pt
```
