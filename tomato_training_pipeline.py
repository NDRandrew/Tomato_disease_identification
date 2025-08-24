#!/usr/bin/env python3
"""
Tomato Disease Detection Training Pipeline
Supports incremental disease training and two-stage model training
"""

import os
import cv2
import numpy as np
import yaml
import json
import shutil
from pathlib import Path
import logging
from typing import List, Dict, Tuple, Optional
import random
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import defaultdict

# Import training dependencies
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("Warning: ultralytics not available. Install with: pip install ultralytics torch torchvision")

class TomatoTrainingPipeline:
    """
    Comprehensive training pipeline for tomato and disease detection
    Supports incremental disease learning
    """
    
    def __init__(self, project_dir: str = "tomato_training_project"):
        """
        Initialize the training pipeline
        
        Args:
            project_dir: Directory for training project
        """
        self.project_dir = Path(project_dir)
        self.setup_logging()
        self.setup_project_structure()
        
        # Disease classes for incremental training
        self.disease_classes = [
            'healthy',
            'bacterial_spot',
            'early_blight', 
            'late_blight',
            'leaf_mold',
            'septoria_leaf_spot',
            'spider_mites',
            'target_spot',
            'yellow_leaf_curl_virus',
            'mosaic_virus',
            'bacterial_canker'
        ]
        
        self.current_disease_training = None
        
    def setup_logging(self):
        """Setup comprehensive logging system"""
        # Create logs directory
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        # Create log filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"training_{timestamp}.log"
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(__name__)
        self.start_time = datetime.now() 
        
        # Log system info
        self.logger.info("="*60)
        self.logger.info("TOMATO TRAINING PIPELINE STARTED")
        self.logger.info("="*60)
        self.logger.info(f"Log file: {log_file}")
        self.logger.info(f"Project directory: {self.project_dir}")


    def log_session_end(self):
        """Log session end time and duration"""
        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds()
        self.logger.info("="*60)
        self.logger.info(f"SESSION ENDED: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info(f"Total duration: {duration:.2f}s")
        self.logger.info("="*60)
        
    def setup_project_structure(self):
        """Create the required directory structure for training"""
        
        # Create main directories
        directories = [
            # Raw data organization
            'raw_images/tomato_detection/tomato',        # Tomato fruit images
            'raw_images/tomato_detection/tomato_plant',  # Tomato plant images
            'raw_images/tomato_detection/not_tomato',    # Non-tomato images (apples, other plants, etc.)
            'raw_images/disease_detection',              # Images for disease detection
            
            # Labeled data storage
            'labeled_data/tomato_detection',     # Labeled tomato/plant data
            'labeled_data/disease_detection',    # Labeled disease data
            
            # YOLO datasets for tomato detection
            'datasets/tomato_detection/images/train',
            'datasets/tomato_detection/images/val',
            'datasets/tomato_detection/labels/train',
            'datasets/tomato_detection/labels/val',
            
            # YOLO datasets for disease detection
            'datasets/disease_detection/images/train',
            'datasets/disease_detection/images/val',
            'datasets/disease_detection/labels/train',
            'datasets/disease_detection/labels/val',
            
            # Models storage
            'models',
            
            # Training results
            'training_results/tomato_detection',
            'training_results/disease_detection',
            
            # Incremental disease training tracking
            'disease_progress'
        ]
        
        for directory in directories:
            (self.project_dir / directory).mkdir(parents=True, exist_ok=True)
            
        self.logger.info(f"Created project structure in: {self.project_dir}")
        
        # Create comprehensive README
        self.create_readme()
        
    def create_readme(self):
        """Create detailed README with instructions"""
        readme_content = """# Tomato Disease Detection Training Project

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
"""
        
        with open(self.project_dir / "README.md", "w", encoding='utf-8') as f:
            f.write(readme_content)
            
        self.logger.info("Created comprehensive README.md")
    
    def create_labeling_interface_tomato(self, image_path: str) -> List[Dict]:
        """
        Labeling interface for tomato vs plant detection
        
        Args:
            image_path: Path to image to label
            
        Returns:
            List of bounding box annotations
        """
        # Load image
        image = self.load_image_for_labeling(image_path)
        if image is None:
            return []
        
        height, width = image.shape[:2]
        annotations = []
        
        class TomatoBBoxSelector:
            def __init__(self, image):
                self.image = image
                self.annotations = []
                self.current_bbox = None
                self.start_point = None
                self.start_time = datetime.now()
                self.current_class = 'tomato'  # Default class
                
            def onclick(self, event):
                if event.inaxes and event.button == 1:  # Left click
                    if self.start_point is None:
                        # Start drawing bbox
                        self.start_point = (event.xdata, event.ydata)
                    else:
                        # Finish drawing bbox
                        end_point = (event.xdata, event.ydata)
                        
                        x1, y1 = self.start_point
                        x2, y2 = end_point
                        
                        # Ensure bbox is valid
                        x1, x2 = min(x1, x2), max(x1, x2)
                        y1, y2 = min(y1, y2), max(y1, y2)
                        
                        if x2 - x1 > 10 and y2 - y1 > 10:  # Minimum size
                            annotation = {
                                'class': self.current_class,
                                'bbox': [x1, y1, x2, y2],
                                'width': width,
                                'height': height
                            }
                            self.annotations.append(annotation)
                            
                            # Choose color based on class
                            if self.current_class == 'tomato':
                                color = 'red'
                            elif self.current_class == 'tomato_plant':
                                color = 'blue'
                            elif self.current_class == 'not_tomato':
                                color = 'yellow'
                            else:
                                color = 'gray'
                            
                            # Draw the bbox
                            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, 
                                                   linewidth=2, edgecolor=color, facecolor='none')
                            event.inaxes.add_patch(rect)
                            plt.draw()
                            
                            print(f"Added {self.current_class} annotation: {len(self.annotations)} total")
                        
                        self.start_point = None
                        
            def onkey(self, event):
                if event.key == 't':
                    self.current_class = 'tomato'
                    print("Switched to TOMATO mode (red boxes)")
                elif event.key == 'p':
                    self.current_class = 'tomato_plant'
                    print("Switched to PLANT mode (blue boxes)")
                elif event.key == 'n':
                    self.current_class = 'not_tomato'
                    print("Switched to NOT_TOMATO mode (yellow boxes)")
        
        # Create interactive plot
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.imshow(image)
        ax.set_title(f"Label Tomatoes/Plants: {Path(image_path).name}\n" +
                    "Press 't' for tomato, 'p' for plant, 'n' for not_tomato | Left click to draw bbox | Close when done")
        
        selector = TomatoBBoxSelector(image)
        fig.canvas.mpl_connect('button_press_event', selector.onclick)
        fig.canvas.mpl_connect('key_press_event', selector.onkey)
        
        print("\n" + "="*60)
        print("TOMATO/PLANT LABELING INSTRUCTIONS:")
        print("- Press 't' to switch to TOMATO mode (red boxes)")
        print("- Press 'p' to switch to PLANT mode (blue boxes)")
        print("- Press 'n' to switch to NOT_TOMATO mode (yellow boxes)")
        print("- Left click to start bbox, left click again to finish")
        print("- Close window when done")
        print("="*60)
        
        plt.show()
        
        return selector.annotations
    
    def create_labeling_interface_disease(self, image_path: str, current_disease: str) -> List[Dict]:
        """
        Labeling interface for disease detection
        
        Args:
            image_path: Path to image to label
            current_disease: Current disease being labeled
            
        Returns:
            List of bounding box annotations
        """
        # Load image
        image = self.load_image_for_labeling(image_path)
        if image is None:
            return []
        
        height, width = image.shape[:2]
        annotations = []
        
        class DiseaseBBoxSelector:
            def __init__(self, image, disease):
                self.image = image
                self.annotations = []
                self.current_bbox = None
                self.start_point = None
                self.current_disease = disease
                
            def onclick(self, event):
                if event.inaxes and event.button == 1:  # Left click
                    if self.start_point is None:
                        # Start drawing bbox
                        self.start_point = (event.xdata, event.ydata)
                    else:
                        # Finish drawing bbox
                        end_point = (event.xdata, event.ydata)
                        
                        x1, y1 = self.start_point
                        x2, y2 = end_point
                        
                        # Ensure bbox is valid
                        x1, x2 = min(x1, x2), max(x1, x2)
                        y1, y2 = min(y1, y2), max(y1, y2)
                        
                        if x2 - x1 > 10 and y2 - y1 > 10:  # Minimum size
                            annotation = {
                                'class': self.current_disease,
                                'bbox': [x1, y1, x2, y2],
                                'width': width,
                                'height': height
                            }
                            self.annotations.append(annotation)
                            
                            # Color based on health status
                            color = 'green' if self.current_disease == 'healthy' else 'red'
                            
                            # Draw the bbox
                            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, 
                                                   linewidth=2, edgecolor=color, facecolor='none')
                            event.inaxes.add_patch(rect)
                            plt.draw()
                            
                            print(f"Added {self.current_disease} annotation: {len(self.annotations)} total")
                        
                        self.start_point = None
        
        # Create interactive plot
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.imshow(image)
        ax.set_title(f"Label Disease: {current_disease}\n" +
                    f"Image: {Path(image_path).name} | Left click to draw bbox | Close when done")
        
        selector = DiseaseBBoxSelector(image, current_disease)
        fig.canvas.mpl_connect('button_press_event', selector.onclick)
        
        print(f"\n🦠 Labeling {current_disease} in {Path(image_path).name}")
        print("Left click to start bbox, left click again to finish")
        
        plt.show()
        
        return selector.annotations
    
    def load_image_for_labeling(self, image_path: str) -> Optional[np.ndarray]:
        """Load and prepare image for labeling interface"""
        try:
            if image_path.lower().endswith(('.tif', '.tiff')):
                try:
                    import rasterio
                    with rasterio.open(image_path) as src:
                        image = src.read()
                        if len(image.shape) == 3:
                            image = np.transpose(image, (1, 2, 0))
                        # Use first 3 bands for display
                        if image.shape[2] >= 3:
                            image = image[:, :, :3]
                        # Normalize for display
                        image = ((image - image.min()) / (image.max() - image.min() + 1e-8) * 255).astype(np.uint8)
                except:
                    image = cv2.imread(image_path)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image = cv2.imread(image_path)
                if image is None:
                    self.logger.error(f"❌ Could not load image: {image_path}")
                    return None
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            return image
            
        except Exception as e:
            self.logger.error(f"❌ Error loading image {image_path}: {e}")
            return None
    
    def label_tomato_detection_data(self):
        """Interactive labeling for tomato vs plant detection"""
        labeling_start = datetime.now()
        self.logger.info("Starting tomato detection labeling")
        
        raw_images_dir = self.project_dir / "raw_images" / "tomato_detection"
        labeled_data_dir = self.project_dir / "labeled_data" / "tomato_detection"
        
        if not raw_images_dir.exists():
            self.logger.error(f"Raw images directory not found: {raw_images_dir}")
            self.logger.info("Please add images to raw_images/tomato_detection/")
            return
        
        # Find all images
        image_files = self.find_images(raw_images_dir)
        
        if not image_files:
            self.logger.error(f"No images found in {raw_images_dir}")
            return
        
        self.logger.info(f"Found {len(image_files)} images to label")
        
        labeled_count = 0
        total_annotations = 0
        
        for i, image_path in enumerate(image_files):
            print(f"\n--- Labeling image {i+1}/{len(image_files)}: {image_path.name} ---")
            
            # Check if already labeled
            label_file = labeled_data_dir / f"{image_path.stem}.json"
            if label_file.exists():
                response = input(f"Image already labeled. Re-label? (y/n): ").strip().lower()
                if response != 'y':
                    continue
            
            # Label the image
            annotations = self.create_labeling_interface_tomato(str(image_path))
            
            if annotations:
                # Save annotations
                label_data = {
                    'image_path': str(image_path),
                    'image_name': image_path.name,
                    'task': 'tomato_detection',
                    'annotations': annotations,
                    'labeled_date': datetime.now().isoformat()
                }
                
                with open(label_file, 'w', encoding='utf-8') as f:
                    json.dump(label_data, f, indent=2)
                
                # Copy image to labeled data directory
                shutil.copy2(image_path, labeled_data_dir / image_path.name)
                
                labeled_count += 1
                total_annotations += len(annotations)
                self.logger.info(f"Labeled {image_path.name}: {len(annotations)} annotations")
            else:
                self.logger.info(f"No annotations for {image_path.name}")
        
        labeling_time = (datetime.now() - labeling_start).total_seconds()
        self.logger.info(f"Labeling complete: {labeled_count} images, {total_annotations} annotations in {labeling_time:.2f}s")
    
    def label_disease_detection_data(self, disease: Optional[str] = None):
        """
        Interactive labeling for disease detection
        
        Args:
            disease: Specific disease to label (None for all)
        """
        self.logger.info(f"🦠 Starting disease detection labeling...")
        
        raw_images_dir = self.project_dir / "raw_images" / "disease_detection"
        labeled_data_dir = self.project_dir / "labeled_data" / "disease_detection"
        
        if not raw_images_dir.exists():
            self.logger.error(f"❌ Raw images directory not found: {raw_images_dir}")
            self.logger.info("Please organize images in raw_images/disease_detection/[disease_name]/")
            return
        
        # Get disease folders to process
        if disease:
            disease_folders = [raw_images_dir / disease] if (raw_images_dir / disease).exists() else []
        else:
            disease_folders = [f for f in raw_images_dir.iterdir() if f.is_dir()]
        
        if not disease_folders:
            self.logger.error(f"❌ No disease folders found")
            return
        
        total_images_labeled = 0
        
        for disease_folder in disease_folders:
            disease_name = disease_folder.name
            self.logger.info(f"🦠 Labeling disease: {disease_name}")
            
            image_files = self.find_images(disease_folder)
            if not image_files:
                self.logger.warning(f"⚠️ No images found for {disease_name}")
                continue
            
            for i, image_path in enumerate(image_files):
                print(f"\n--- {disease_name} - Image {i+1}/{len(image_files)}: {image_path.name} ---")
                
                # Check if already labeled
                label_file = labeled_data_dir / f"{image_path.stem}_{disease_name}.json"
                if label_file.exists():
                    response = input(f"Image already labeled. Re-label? (y/n): ").strip().lower()
                    if response != 'y':
                        continue
                
                # Label the image
                annotations = self.create_labeling_interface_disease(str(image_path), disease_name)
                
                if annotations:
                    # Save annotations
                    label_data = {
                        'image_path': str(image_path),
                        'image_name': image_path.name,
                        'task': 'disease_detection',
                        'disease': disease_name,
                        'annotations': annotations,
                        'labeled_date': datetime.now().isoformat()
                    }
                    
                    with open(label_file, 'w', encoding='utf-8') as f:
                        json.dump(label_data, f, indent=2)
                    
                    # Copy image to labeled data directory
                    shutil.copy2(image_path, labeled_data_dir / f"{image_path.stem}_{disease_name}{image_path.suffix}")
                    
                    self.logger.info(f"✅ Saved {len(annotations)} annotations for {image_path.name}")
                    total_images_labeled += 1
                else:
                    self.logger.info(f"ℹ️ No annotations created for {image_path.name}")
        
        self.logger.info(f"✅ Disease labeling complete. Labeled {total_images_labeled} images total.")
    
    def find_images(self, directory: Path) -> List[Path]:
        """Find all image files in a directory"""
        image_extensions = ['.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(directory.glob(f"*{ext}"))
            image_files.extend(directory.glob(f"*{ext.upper()}"))
        
        return sorted(image_files)
    
    def convert_to_yolo_format(self, annotations: List[Dict], output_dir: str, image_name: str, class_mapping: Dict[str, int]):
        """Convert annotations to YOLO format"""
        if not annotations:
            return
            
        label_path = Path(output_dir) / f"{Path(image_name).stem}.txt"
        
        with open(label_path, 'w') as f:
            for ann in annotations:
                bbox = ann['bbox']
                class_name = ann['class']
                img_width = ann['width']
                img_height = ann['height']
                
                # Get class ID
                class_id = class_mapping.get(class_name, 0)
                
                # Convert to YOLO format (normalized center coordinates + width/height)
                x_center = (bbox[0] + bbox[2]) / 2 / img_width
                y_center = (bbox[1] + bbox[3]) / 2 / img_height
                width = (bbox[2] - bbox[0]) / img_width
                height = (bbox[3] - bbox[1]) / img_height
                
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
    
    def prepare_tomato_dataset(self, train_split: float = 0.8):
        """Prepare YOLO dataset for tomato detection"""
        self.logger.info("📊 Preparing tomato detection dataset...")
        
        labeled_data_dir = self.project_dir / "labeled_data" / "tomato_detection"
        dataset_dir = self.project_dir / "datasets" / "tomato_detection"
        
        # Find labeled data
        label_files = list(labeled_data_dir.glob("*.json"))
        if not label_files:
            self.logger.error("❌ No labeled tomato data found")
            return False
        
        # Class mapping
        class_mapping = {'tomato': 0, 'tomato_plant': 1, 'not_tomato': 2}
        
        # Split data
        random.shuffle(label_files)
        split_idx = int(len(label_files) * train_split)
        train_files = label_files[:split_idx]
        val_files = label_files[split_idx:]
        
        self.logger.info(f"📊 Dataset split - Train: {len(train_files)}, Val: {len(val_files)}")
        
        # Process training data
        for label_file in train_files:
            self.process_dataset_file(label_file, dataset_dir, 'train', class_mapping)
        
        # Process validation data
        for label_file in val_files:
            self.process_dataset_file(label_file, dataset_dir, 'val', class_mapping)
        
        # Create dataset config
        dataset_config = {
            'path': str(dataset_dir),
            'train': 'images/train',
            'val': 'images/val',
            'nc': 3,
            'names': ['tomato', 'tomato_plant', 'not_tomato']
        }
        
        config_path = dataset_dir / "data.yaml"
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(dataset_config, f)
        
        self.logger.info(f"✅ Tomato dataset prepared: {config_path}")
        return True
    
    def prepare_disease_dataset(self, train_split: float = 0.8, specific_disease: Optional[str] = None):
        """Prepare YOLO dataset for disease detection"""
        self.logger.info("🦠 Preparing disease detection dataset...")
        
        labeled_data_dir = self.project_dir / "labeled_data" / "disease_detection"
        dataset_dir = self.project_dir / "datasets" / "disease_detection"
        
        # Find labeled data
        if specific_disease:
            label_files = list(labeled_data_dir.glob(f"*_{specific_disease}.json"))
            self.logger.info(f"🔬 Preparing dataset for specific disease: {specific_disease}")
        else:
            label_files = list(labeled_data_dir.glob("*.json"))
            self.logger.info("🔬 Preparing dataset for all diseases")
        
        if not label_files:
            self.logger.error("❌ No labeled disease data found")
            return False
        
        # Create class mapping
        all_classes = set()
        for label_file in label_files:
            with open(label_file, 'r') as f:
                data = json.load(f)
                for ann in data['annotations']:
                    all_classes.add(ann['class'])
        
        # Sort classes for consistent mapping
        sorted_classes = sorted(list(all_classes))
        class_mapping = {cls: i for i, cls in enumerate(sorted_classes)}
        
        self.logger.info(f"📋 Disease classes found: {sorted_classes}")
        
        # Split data
        random.shuffle(label_files)
        split_idx = int(len(label_files) * train_split)
        train_files = label_files[:split_idx]
        val_files = label_files[split_idx:]
        
        self.logger.info(f"📊 Dataset split - Train: {len(train_files)}, Val: {len(val_files)}")
        
        # Process training data
        for label_file in train_files:
            self.process_dataset_file(label_file, dataset_dir, 'train', class_mapping)
        
        # Process validation data
        for label_file in val_files:
            self.process_dataset_file(label_file, dataset_dir, 'val', class_mapping)
        
        # Create dataset config
        dataset_config = {
            'path': str(dataset_dir),
            'train': 'images/train',
            'val': 'images/val',
            'nc': len(sorted_classes),
            'names': sorted_classes
        }
        
        config_path = dataset_dir / "data.yaml"
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(dataset_config, f)
        
        self.logger.info(f"✅ Disease dataset prepared: {config_path}")
        return True
    
    def process_dataset_file(self, label_file: Path, dataset_dir: Path, split: str, class_mapping: Dict[str, int]):
        """Process a single labeled file for YOLO dataset"""
        try:
            with open(label_file, 'r') as f:
                data = json.load(f)
            
            image_name = data['image_name']
            annotations = data['annotations']
            
            # Copy image
            src_image = label_file.parent / image_name
            if not src_image.exists():
                # Try with modified name for disease detection
                if 'disease' in data.get('task', ''):
                    disease = data.get('disease', '')
                    modified_name = f"{Path(image_name).stem}_{disease}{Path(image_name).suffix}"
                    src_image = label_file.parent / modified_name
            
            if src_image.exists():
                dst_image = dataset_dir / "images" / split / image_name
                shutil.copy2(src_image, dst_image)
                
                # Convert and save labels
                self.convert_to_yolo_format(
                    annotations,
                    dataset_dir / "labels" / split,
                    image_name,
                    class_mapping
                )
            else:
                self.logger.warning(f"⚠️ Source image not found: {src_image}")
                
        except Exception as e:
            self.logger.error(f"❌ Error processing {label_file}: {e}")
    
    def train_tomato_model(self, epochs: int = 100, batch_size: int = 16, model_size: str = 's'):
        """Train tomato detection model"""
        if not YOLO_AVAILABLE:
            self.logger.error("ultralytics not available")
            return None
        
        training_start = datetime.now()
        self.logger.info(f"Starting tomato detection training: {epochs} epochs, batch {batch_size}, model {model_size}")
        
        # Prepare dataset
        if not self.prepare_tomato_dataset():
            return None
        
        # Initialize model
        model = YOLO(f'yolov8{model_size}.pt')
        
        # Train
        results = model.train(
            data=str(self.project_dir / "datasets" / "tomato_detection" / "data.yaml"),
            epochs=epochs,
            batch=batch_size,
            imgsz=640,
            project=str(self.project_dir / "training_results"),
            name="tomato_detection",
            save=True,
            plots=True
        )
        
        # Save best model
        best_model_path = self.project_dir / "models" / "tomato_detection_best.pt"
        shutil.copy2(results.save_dir / "weights" / "best.pt", best_model_path)
        
        training_time = (datetime.now() - training_start).total_seconds()
        session_time = (datetime.now() - self.start_time).total_seconds()
        
        self.logger.info(f"Tomato training complete: {training_time:.2f}s")
        self.logger.info(f"Total session time: {session_time:.2f}s")
        self.logger.info(f"Model saved: {best_model_path}")
        
        return results
    
    def train_disease_model(self, epochs: int = 100, batch_size: int = 16, model_size: str = 's', 
                           specific_disease: Optional[str] = None, incremental: bool = False):
        """Train disease detection model"""
        if not YOLO_AVAILABLE:
            self.logger.error("ultralytics not available")
            return None
        
        training_start = datetime.now()
        disease_info = f" for {specific_disease}" if specific_disease else " for all diseases"
        mode_info = " (incremental)" if incremental else ""
        
        self.logger.info(f"Starting disease training{disease_info}{mode_info}: {epochs} epochs, batch {batch_size}")
        
        # Prepare dataset
        if not self.prepare_disease_dataset(specific_disease=specific_disease):
            return None
        
        # Initialize model
        if incremental and (self.project_dir / "models" / "disease_detection_best.pt").exists():
            model = YOLO(str(self.project_dir / "models" / "disease_detection_best.pt"))
            self.logger.info("Loading existing model for incremental training")
        else:
            model = YOLO(f'yolov8{model_size}.pt')
            self.logger.info(f"Starting fresh training with YOLOv8{model_size}")
        
        # Train
        training_name = f"disease_detection_{specific_disease}" if specific_disease else "disease_detection_all"
        
        results = model.train(
            data=str(self.project_dir / "datasets" / "disease_detection" / "data.yaml"),
            epochs=epochs,
            batch=batch_size,
            imgsz=640,
            project=str(self.project_dir / "training_results"),
            name=training_name,
            save=True,
            plots=True
        )
        
        # Save best model
        best_model_path = self.project_dir / "models" / "disease_detection_best.pt"
        shutil.copy2(results.save_dir / "weights" / "best.pt", best_model_path)
        
        # Update disease training progress
        self.update_disease_progress(specific_disease, results)
        
        training_time = (datetime.now() - training_start).total_seconds()
        session_time = (datetime.now() - self.start_time).total_seconds()
        
        self.logger.info(f"Disease training complete{disease_info}: {training_time:.2f}s")
        self.logger.info(f"Total session time: {session_time:.2f}s")
        self.logger.info(f"Model saved: {best_model_path}")
        
        return results
    
    def update_disease_progress(self, disease: Optional[str], results):
        """Update disease training progress tracking"""
        progress_file = self.project_dir / "disease_progress" / "training_log.json"
        
        # Load existing progress
        if progress_file.exists():
            with open(progress_file, 'r', encoding='utf-8') as f:
                progress = json.load(f)
        else:
            progress = {'diseases_trained': [], 'training_history': []}
        
        # Add current training
        training_entry = {
            'disease': disease,
            'timestamp': datetime.now().isoformat(),
            'epochs': results.epochs if hasattr(results, 'epochs') else None,
            'best_fitness': float(results.best_fitness) if hasattr(results, 'best_fitness') else None
        }
        
        progress['training_history'].append(training_entry)
        
        if disease and disease not in progress['diseases_trained']:
            progress['diseases_trained'].append(disease)
        
        # Save progress
        progress_file.parent.mkdir(exist_ok=True)
        with open(progress_file, 'w', encoding='utf-8') as f:
            json.dump(progress, f, indent=2)
        
        disease_name = disease or 'all diseases'
        self.logger.info(f"Updated training progress for {disease_name}")
    
    def test_model(self, model_path: str, test_images_dir: str, model_type: str = "tomato"):
        """Test trained model"""
        if not YOLO_AVAILABLE:
            self.logger.error("❌ ultralytics not available")
            return
        
        if not os.path.exists(model_path):
            self.logger.error(f"❌ Model not found: {model_path}")
            return
        
        self.logger.info(f"🧪 Testing {model_type} model: {model_path}")
        
        model = YOLO(model_path)
        
        # Find test images
        test_dir = Path(test_images_dir)
        if not test_dir.exists():
            self.logger.error(f"❌ Test directory not found: {test_dir}")
            return
        
        test_images = self.find_images(test_dir)
        if not test_images:
            self.logger.error(f"❌ No test images found in {test_dir}")
            return
        
        self.logger.info(f"📷 Testing on {len(test_images)} images")
        
        # Create results directory
        results_dir = self.project_dir / "training_results" / f"{model_type}_test_results"
        results_dir.mkdir(exist_ok=True)
        
        total_detections = 0
        
        for image_path in test_images:
            try:
                results = model(str(image_path))
                
                # Save annotated results
                for r in results:
                    annotated = r.plot()
                    output_path = results_dir / f"{image_path.stem}_detected.jpg"
                    cv2.imwrite(str(output_path), annotated)
                    
                    # Count detections
                    num_detections = len(r.boxes) if r.boxes is not None else 0
                    total_detections += num_detections
                    
                    self.logger.info(f"✅ {image_path.name}: {num_detections} detections")
                    
            except Exception as e:
                self.logger.error(f"❌ Error testing {image_path}: {e}")
        
        self.logger.info(f"🏁 Testing complete. Total detections: {total_detections}")
        self.logger.info(f"📁 Results saved to: {results_dir}")

def main():
    """Main function with comprehensive command-line interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Tomato Disease Detection Training Pipeline")
    parser.add_argument("--project", default="tomato_training_project", help="Project directory")
    
    # Labeling commands
    parser.add_argument("--label-tomato", action="store_true", help="Label tomato detection data")
    parser.add_argument("--label-disease", action="store_true", help="Label disease detection data")
    parser.add_argument("--disease", help="Specific disease to label/train")
    
    # Training commands
    parser.add_argument("--train-tomato", action="store_true", help="Train tomato detection model")
    parser.add_argument("--train-disease", action="store_true", help="Train disease detection model")
    parser.add_argument("--all-diseases", action="store_true", help="Train on all diseases")
    parser.add_argument("--incremental", action="store_true", help="Incremental training (add to existing model)")
    
    # Testing commands
    parser.add_argument("--test-tomato", action="store_true", help="Test tomato model")
    parser.add_argument("--test-disease", action="store_true", help="Test disease model")
    parser.add_argument("--model", help="Model path for testing")
    parser.add_argument("--test-dir", default="test_images", help="Test images directory")
    
    # Training parameters
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--model-size", default="s", choices=['n', 's', 'm', 'l', 'x'], help="Model size")
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = TomatoTrainingPipeline(args.project)
    
    # Execute commands
    if args.label_tomato:
        print("Starting tomato detection labeling...")
        trainer.label_tomato_detection_data()
    
    elif args.label_disease:
        print("Starting disease detection labeling...")
        trainer.label_disease_detection_data(args.disease)
    
    elif args.train_tomato:
        print("Training tomato detection model...")
        trainer.train_tomato_model(
            epochs=args.epochs,
            batch_size=args.batch_size,
            model_size=args.model_size
        )
    
    elif args.train_disease:
        print("Training disease detection model...")
        disease = args.disease if not args.all_diseases else None
        trainer.train_disease_model(
            epochs=args.epochs,
            batch_size=args.batch_size,
            model_size=args.model_size,
            specific_disease=disease,
            incremental=args.incremental
        )
    
    elif args.test_tomato:
        print("Testing tomato detection model...")
        model_path = args.model or f"{args.project}/models/tomato_detection_best.pt"
        trainer.test_model(model_path, args.test_dir, "tomato")
    
    elif args.test_disease:
        print("Testing disease detection model...")
        model_path = args.model or f"{args.project}/models/disease_detection_best.pt"
        trainer.test_model(model_path, args.test_dir, "disease")
    
    else:
        print("TOMATO DISEASE DETECTION TRAINING PIPELINE")
        print("=" * 50)
        print("\nQUICK START GUIDE:")
        print("\n1. PREPARE DATA:")
        print("   • Add images to raw_images/tomato_detection/")
        print("   • Add disease images to raw_images/disease_detection/[disease]/")
        print("\n2. LABEL DATA:")
        print("   python tomato_training_pipeline.py --label-tomato")
        print("   python tomato_training_pipeline.py --label-disease")
        print("\n3. TRAIN MODELS:")
        print("   python tomato_training_pipeline.py --train-tomato")
        print("   python tomato_training_pipeline.py --train-disease --all-diseases")
        print("\n4. INCREMENTAL DISEASE TRAINING:")
        print("   python tomato_training_pipeline.py --train-disease --disease bacterial_spot")
        print("   python tomato_training_pipeline.py --train-disease --disease early_blight --incremental")
        print("\n5. TEST MODELS:")
        print("   python tomato_training_pipeline.py --test-tomato")
        print("   python tomato_training_pipeline.py --test-disease")
        print("\nFor detailed instructions, see README.md")
        print("All logs are automatically saved to logs/ directory")
    
    trainer.log_session_end()

if __name__ == "__main__":
    main()