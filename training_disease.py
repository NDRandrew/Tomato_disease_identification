#!/usr/bin/env python3
"""
Tomato Disease Detection Training Pipeline - GPU Enabled
Supports incremental disease training and two-stage model training
Now with GPU acceleration and optimized performance
COMPLETE SEPARATION: Tomato Segmentation vs Disease Detection (bbox)
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
import torch
from matplotlib.patches import Polygon as MPLPolygon

# Import training dependencies
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("Warning: ultralytics not available. Install with: pip install ultralytics torch torchvision")

def check_gpu_setup():
    """
    Check GPU setup and provide recommendations
    """
    print("="*50)
    print("GPU SETUP CHECK")
    print("="*50)
    
    # Check PyTorch installation
    print(f"PyTorch version: {torch.__version__}")
    
    # Check CUDA availability
    if torch.cuda.is_available():
        print(f"CUDA available: {torch.version.cuda}")
        print(f"GPU detected: {torch.cuda.get_device_name(0)}")
        
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU memory: {gpu_memory:.1f} GB")
        
        if gpu_memory >= 8:
            print("Sufficient GPU memory for training")
        elif gpu_memory >= 4:
            print("Limited GPU memory. Use smaller batch sizes")
        else:
            print("Very limited GPU memory. CPU training recommended")
            
    else:
        print("CUDA not available")
        print("To enable GPU training:")
        print("   1. Install NVIDIA GPU drivers")
        print("   2. Install CUDA toolkit")
        print("   3. Install GPU PyTorch:")
        print("      pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
    
    print("="*50)

class TomatoTrainingPipeline:
    """
    Comprehensive training pipeline for tomato and disease detection
    Supports incremental disease learning with GPU acceleration
    """
    
    def __init__(self, project_dir: str = "tomato_training_project"):
        """
        Initialize the training pipeline with GPU support
        
        Args:
            project_dir: Directory for training project
        """
        self.project_dir = Path(project_dir)
        self.setup_logging()
        self.setup_project_structure()
        
        # GPU/Device setup
        self.device = self.setup_device()
        
        # Disease classes for incremental training
        self.disease_classes = [
            'bacterial_spot',
            'early_blight', 
            'late_blight',
            'leaf_mold',
            'septoria_leaf_spot',
            'spider_mites',
            'target_spot',
            'yellow_leaf_curl_virus',
            'mosaic_virus_plant'
        ]
        
        self.current_disease_training = None
    
    def setup_device(self):
        """
        Setup and detect the best available device (GPU/CPU)
        """
        if torch.cuda.is_available():
            device = 'cuda'
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
            self.logger.info(f"GPU detected: {gpu_name}")
            self.logger.info(f"GPU memory: {gpu_memory:.1f} GB")
            self.logger.info(f"CUDA version: {torch.version.cuda}")
            
            # Log GPU memory usage
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated(0) / 1024**3
                cached = torch.cuda.memory_reserved(0) / 1024**3
                self.logger.info(f"GPU memory - Allocated: {allocated:.2f} GB, Cached: {cached:.2f} GB")
                
        else:
            device = 'cpu'
            self.logger.warning("No GPU detected. Training will use CPU (much slower)")
            self.logger.info("To enable GPU:")
            self.logger.info("1. Install NVIDIA CUDA drivers")
            self.logger.info("2. Install GPU PyTorch: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
        
        self.logger.info(f"Training device: {device}")
        return device
    
    def get_optimal_batch_size(self, base_batch_size: int = 16) -> int:
        """
        Determine optimal batch size based on available GPU memory
        """
        if self.device == 'cpu':
            return max(4, base_batch_size // 4)  # Reduce for CPU
        
        # GPU memory-based batch size adjustment
        if torch.cuda.is_available():
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
            
            if gpu_memory_gb >= 24:  # RTX 4090, A100, etc.
                return base_batch_size * 2
            elif gpu_memory_gb >= 12:  # RTX 4080, 3080 Ti, etc.
                return base_batch_size
            elif gpu_memory_gb >= 8:   # RTX 3070, 4060 Ti, etc.
                return max(8, base_batch_size // 2)
            elif gpu_memory_gb >= 6:   # RTX 3060, etc.
                return max(4, base_batch_size // 4)
            else:  # Lower memory GPUs
                return max(2, base_batch_size // 8)
        
        return base_batch_size
    
    def monitor_gpu_usage(self):
        """
        Monitor and log GPU usage during training
        """
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(0) / 1024**3
            cached = torch.cuda.memory_reserved(0) / 1024**3
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            
            self.logger.info(f"GPU Memory - Used: {allocated:.2f}/{total:.1f} GB ({allocated/total*100:.1f}%)")
            
            # Warning for high memory usage
            if allocated / total > 0.9:
                self.logger.warning("GPU memory usage >90%. Consider reducing batch size.")
        
    def setup_logging(self):
        """Setup comprehensive logging system"""
        # Create logs directory
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        # Create log filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"training_{timestamp}.log"
        
        # Configure logging with UTF-8 encoding to handle special characters
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
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

## Quick Start Guide

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

## Project Structure

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

## Supported Diseases

1. bacterial_spot
2. early_blight
3. late_blight
4. leaf_mold
5. septoria_leaf_spot
6. spider_mites
7. target_spot
8. yellow_leaf_curl_virus
9. mosaic_virus_plant
10. bacterial_canker

## Tips for Better Training

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

## Advanced Usage

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

## GPU Support

The pipeline automatically detects and uses GPU if available:
- NVIDIA GPU with CUDA support recommended
- Automatic batch size optimization based on GPU memory
- Memory monitoring during training
- Fallback to CPU if GPU unavailable

To enable GPU support:
1. Install NVIDIA drivers and CUDA
2. Install GPU PyTorch: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118`
"""
        
        with open(self.project_dir / "README.md", "w", encoding='utf-8') as f:
            f.write(readme_content)
            
        self.logger.info("Created comprehensive README.md")

    def get_current_annotation_counts(self, task_type: str = "tomato_detection") -> Dict[str, int]:
        """
        Count all existing annotations across all labeled images
        
        Args:
            task_type: Either "tomato_detection" or "disease_detection"
            
        Returns:
            Dictionary with counts for each class
        """
        labeled_data_dir = self.project_dir / "labeled_data" / task_type
        counts = {}
        
        if not labeled_data_dir.exists():
            return counts
        
        # Find all label files
        label_files = list(labeled_data_dir.glob("*.json"))
        
        for label_file in label_files:
            try:
                with open(label_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                annotations = data.get('annotations', [])
                for annotation in annotations:
                    class_name = annotation.get('class', 'unknown')
                    if class_name not in counts:
                        counts[class_name] = 0
                    counts[class_name] += 1
                    
            except Exception as e:
                self.logger.warning(f"Error reading label file {label_file}: {e}")
                continue
        
        return counts
    
    def format_counts_display(self, counts: Dict[str, int], task_type: str = "tomato_detection") -> str:
        """
        Format annotation counts for display (updated for new classes)
        """
        if not counts:
            return "No annotations found"
        
        if task_type == "tomato_detection":
            # Updated order for segmentation classes
            class_order = ['mature_tomato', 'immature_tomato', 'tomato_plant', 'not_tomato']
            title = "TOMATO SEGMENTATION COUNTS"
        else:
            # Order for disease detection
            class_order = ['bacterial_spot'] + sorted([k for k in counts.keys() if k != 'bacterial_spot'])
            title = "DISEASE DETECTION COUNTS"
        
        lines = [f"\n{title}:"]
        lines.append("-" * len(title))
        
        total = 0
        for class_name in class_order:
            if class_name in counts:
                count = counts[class_name]
                lines.append(f"{class_name}: {count}")
                total += count
        
        # Add any classes not in the predefined order
        for class_name, count in counts.items():
            if class_name not in class_order:
                lines.append(f"{class_name}: {count}")
                total += count
        
        lines.append("-" * len(title))
        lines.append(f"TOTAL: {total}")
        
        return "\n".join(lines)
    
    def update_counts_display(self, fig, current_counts: Dict[str, int], task_type: str = "tomato_detection"):
        """
        Update the counts display on the matplotlib figure
        
        Args:
            fig: Matplotlib figure
            current_counts: Current annotation counts
            task_type: Type of task
        """
        # Remove existing text if any
        for text in fig.texts:
            text.remove()
        
        # Format and add new counts text
        counts_text = self.format_counts_display(current_counts, task_type)
        fig.text(0.02, 0.98, counts_text, transform=fig.transFigure, fontsize=9, 
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
        
        fig.canvas.draw()
    
    def create_segmentation_interface_tomato(self, image_path: str) -> List[Dict]:
        """
        Instance segmentation labeling interface with FIXED undo functionality
        """
        # Load image
        image = self.load_image_for_labeling(image_path)
        if image is None:
            return []
        
        height, width = image.shape[:2]
        
        # Get current annotation counts
        current_counts = self.get_current_annotation_counts("tomato_detection")
        
        class TomatoSegmentationSelector:
            def __init__(self, image, parent_self):
                self.image = image
                self.annotations = []
                self.completed_annotation_visuals = []  # Track visuals per completed annotation
                self.current_polygon_points = []  # Points for current polygon
                self.current_polygon_visuals = []  # Visual elements for current polygon being drawn
                self.current_class = 'mature_tomato'
                self.fig = None
                self.ax = None
                self.parent = parent_self
                self.session_counts = current_counts.copy()
                
            def onclick(self, event):
                if not event.inaxes:
                    return
                    
                if event.button == 1:  # Left click - add point
                    x, y = event.xdata, event.ydata
                    self.current_polygon_points.append([x, y])
                    
                    # Draw point
                    point = self.ax.plot(x, y, 'ro', markersize=5)[0]
                    self.current_polygon_visuals.append(point)
                    
                    # Update polygon line
                    if len(self.current_polygon_points) > 1:
                        # Remove old line if it exists
                        if len(self.current_polygon_visuals) > len(self.current_polygon_points):
                            old_line = self.current_polygon_visuals.pop()
                            old_line.remove()
                        
                        # Draw new line
                        polygon_array = np.array(self.current_polygon_points)
                        line = self.ax.plot(polygon_array[:, 0], polygon_array[:, 1], 
                                        'r-', linewidth=2)[0]
                        self.current_polygon_visuals.append(line)
                    
                    plt.draw()
                    print(f"Added point {len(self.current_polygon_points)}: ({x:.0f}, {y:.0f})")
                    
                elif event.button == 3:  # Right click - finish polygon
                    if len(self.current_polygon_points) >= 3:
                        self.finish_polygon()
                    else:
                        print("Need at least 3 points to create a polygon")
                        
            def finish_polygon(self):
                """Complete the current polygon and save annotation"""
                if len(self.current_polygon_points) < 3:
                    print("Need at least 3 points for a polygon")
                    return
                
                # Create annotation
                annotation = {
                    'class': self.current_class,
                    'segmentation': self.current_polygon_points.copy(),
                    'width': width,
                    'height': height,
                    'type': 'polygon'
                }
                self.annotations.append(annotation)
                
                # Update session counts
                if self.current_class not in self.session_counts:
                    self.session_counts[self.current_class] = 0
                self.session_counts[self.current_class] += 1
                
                # Choose color
                color_map = {
                    'mature_tomato': 'red',
                    'immature_tomato': 'orange',
                    'tomato_plant': 'green',
                    'not_tomato': 'gray'
                }
                color = color_map.get(self.current_class, 'yellow')
                
                # Clear temporary drawing visuals
                for visual in self.current_polygon_visuals:
                    visual.remove()
                self.current_polygon_visuals = []
                
                # Create final polygon visual
                polygon_array = np.array(self.current_polygon_points)
                poly = MPLPolygon(polygon_array, closed=True, 
                                edgecolor=color, facecolor=color, 
                                alpha=0.3, linewidth=2)
                self.ax.add_patch(poly)
                
                # Add label
                centroid_x = np.mean(polygon_array[:, 0])
                centroid_y = np.mean(polygon_array[:, 1])
                text = self.ax.text(centroid_x, centroid_y, 
                                f"{len(self.annotations)}\n{self.current_class}", 
                                color='white', fontweight='bold', fontsize=8,
                                ha='center', va='center',
                                bbox=dict(boxstyle='round,pad=0.3', 
                                        facecolor=color, alpha=0.7))
                
                # Store ONLY the final visuals for this completed annotation
                self.completed_annotation_visuals.append([poly, text])
                
                # Update counts display
                self.parent.update_counts_display(self.fig, self.session_counts, "tomato_detection")
                
                plt.draw()
                
                print(f"Completed {self.current_class} annotation #{len(self.annotations)}")
                print(f"Total {self.current_class}: {self.session_counts.get(self.current_class, 0)}")
                print("Start new polygon with left clicks, right click to finish")
                
                # Reset for next polygon
                self.current_polygon_points = []
                
            def onkey(self, event):
                if event.key == 'm':
                    self.current_class = 'mature_tomato'
                    print("Switched to MATURE TOMATO mode (red)")
                elif event.key == 'i':
                    self.current_class = 'immature_tomato'
                    print("Switched to IMMATURE TOMATO mode (orange)")
                elif event.key == 'p':
                    self.current_class = 'tomato_plant'
                    print("Switched to TOMATO PLANT mode (green)")
                elif event.key == 'n':
                    self.current_class = 'not_tomato'
                    print("Switched to NOT TOMATO mode (gray)")
                elif event.key == 'u':
                    self.undo_last_annotation()
                elif event.key == 'r':
                    self.reset_current_polygon()
                elif event.key == 'c':
                    self.clear_all_annotations()
                elif event.key == 'h':
                    self.show_help()
                elif event.key == 'enter':
                    if len(self.current_polygon_points) >= 3:
                        self.finish_polygon()
                        
            def reset_current_polygon(self):
                """Reset the current polygon being drawn"""
                if self.current_polygon_points:
                    # Remove all temporary visual elements
                    for visual in self.current_polygon_visuals:
                        visual.remove()
                    
                    self.current_polygon_points = []
                    self.current_polygon_visuals = []
                    plt.draw()
                    print("Reset current polygon")
                else:
                    print("No polygon to reset")
                        
            def undo_last_annotation(self):
                """Undo the last completed annotation"""
                if self.annotations:
                    # Remove last annotation data
                    removed_annotation = self.annotations.pop()
                    removed_class = removed_annotation['class']
                    
                    # Update session counts
                    if removed_class in self.session_counts and self.session_counts[removed_class] > 0:
                        self.session_counts[removed_class] -= 1
                    
                    # Remove visual elements for this annotation
                    if self.completed_annotation_visuals:
                        visuals = self.completed_annotation_visuals.pop()
                        for visual in visuals:
                            visual.remove()
                    
                    # Update counts display
                    self.parent.update_counts_display(self.fig, self.session_counts, "tomato_detection")
                    
                    plt.draw()
                    print(f"Undid last annotation ({removed_class}). {len(self.annotations)} annotations remaining.")
                    print(f"Total {removed_class}: {self.session_counts.get(removed_class, 0)}")
                else:
                    print("No annotations to undo")
                        
            def clear_all_annotations(self):
                """Clear all annotations"""
                if self.annotations:
                    response = input("Clear all annotations? (y/n): ").strip().lower()
                    if response == 'y':
                        # Restore original counts
                        for annotation in self.annotations:
                            class_name = annotation['class']
                            if class_name in self.session_counts and self.session_counts[class_name] > 0:
                                self.session_counts[class_name] -= 1
                        
                        self.annotations.clear()
                        
                        # Remove all completed annotation visuals
                        for visuals in self.completed_annotation_visuals:
                            for visual in visuals:
                                visual.remove()
                        self.completed_annotation_visuals.clear()
                        
                        # Also clear current polygon if any
                        for visual in self.current_polygon_visuals:
                            visual.remove()
                        self.current_polygon_visuals = []
                        self.current_polygon_points = []
                        
                        # Update counts display
                        self.parent.update_counts_display(self.fig, self.session_counts, "tomato_detection")
                        
                        plt.draw()
                        print("All annotations cleared")
                else:
                    print("No annotations to clear")
                        
            def show_help(self):
                """Show help information"""
                help_text = """
    SEGMENTATION LABELING CONTROLS:
    ================================
    CLASS SELECTION:
    - 'm': MATURE TOMATO (red) - ripe, ready to harvest
    - 'i': IMMATURE TOMATO (orange) - green or ripening
    - 'p': TOMATO PLANT (green) - leaves and stems
    - 'n': NOT TOMATO (gray) - other objects

    DRAWING:
    - Left click: Add point to polygon
    - Right click: Finish current polygon
    - Enter: Finish current polygon (alternative)

    EDITING:
    - 'r': Reset/cancel current polygon (removes points being drawn)
    - 'u': Undo last COMPLETED annotation (removes finished polygon)
    - 'c': Clear all annotations

    OTHER:
    - 'h': Show this help
    - Close window: Save and continue

    TIPS:
    - Click around the object's outline to create a polygon
    - More points = more accurate segmentation
    - 3 points minimum, 20-50 points recommended
    - Right-click or Enter when done with polygon
    - Use 'r' to restart if you make a mistake while drawing
    - Use 'u' to remove the last finished polygon
                """
                print(help_text)
        
        # Create interactive plot
        fig, ax = plt.subplots(figsize=(15, 10))
        ax.imshow(image)
        ax.set_title(f"Segment Tomatoes: {Path(image_path).name}\n" +
                    "'m'=mature, 'i'=immature, 'p'=plant, 'n'=not_tomato | " +
                    "Left=add point, Right=finish, 'r'=reset, 'u'=undo | 'h'=help")
        
        selector = TomatoSegmentationSelector(image, self)
        selector.fig = fig
        selector.ax = ax
        
        # Initial counts display
        self.update_counts_display(fig, current_counts, "tomato_detection")
        
        fig.canvas.mpl_connect('button_press_event', selector.onclick)
        fig.canvas.mpl_connect('key_press_event', selector.onkey)
        
        print("\n" + "="*70)
        print("TOMATO SEGMENTATION LABELING")
        print("="*70)
        print("CLASS KEYS:")
        print("  'm' = MATURE TOMATO (red) - ripe, ready to harvest")
        print("  'i' = IMMATURE TOMATO (orange) - green or ripening")  
        print("  'p' = TOMATO PLANT (green) - leaves and stems")
        print("  'n' = NOT TOMATO (gray) - other objects")
        print("\nDRAWING:")
        print("  - Left click to add points around object outline")
        print("  - Right click or Enter to finish polygon")
        print("  - 'r' to reset current polygon (while drawing)")
        print("  - 'u' to undo last completed annotation")
        print("  - 'h' for help")
        print("="*70)
        print(self.format_counts_display(current_counts, "tomato_detection"))
        print("="*70)
        
        plt.show()
        
        return selector.annotations
    
    def create_labeling_interface_disease(self, image_path: str, current_disease: str = None) -> List[Dict]:
        """
        Enhanced labeling interface for disease detection with disease selection
        """
        # Load image
        image = self.load_image_for_labeling(image_path)
        if image is None:
            return []
        
        height, width = image.shape[:2]
        
        # Get current annotation counts
        current_counts = self.get_current_annotation_counts("disease_detection")
        
        # If no disease specified, ask user to choose
        if current_disease is None:
            print("\n" + "="*60)
            print("SELECT DISEASE TO LABEL")
            print("="*60)
            for i, disease in enumerate(self.disease_classes, 1):
                count = current_counts.get(disease, 0)
                print(f"{i:2d}. {disease:<25} (current: {count})")
            print("="*60)
            
            while True:
                try:
                    choice = input("Enter disease number (1-11): ").strip()
                    disease_idx = int(choice) - 1
                    if 0 <= disease_idx < len(self.disease_classes):
                        current_disease = self.disease_classes[disease_idx]
                        break
                    else:
                        print(f"Please enter a number between 1 and {len(self.disease_classes)}")
                except ValueError:
                    print("Please enter a valid number")
        
        class DiseaseBBoxSelector:
            def __init__(self, image, disease, parent_self):
                self.image = image
                self.annotations = []
                self.annotation_patches = []
                self.current_bbox = None
                self.start_point = None
                self.current_disease = disease
                self.fig = None
                self.ax = None
                self.parent = parent_self
                self.session_counts = current_counts.copy()
                self.available_diseases = parent_self.disease_classes
                
            def onclick(self, event):
                if event.inaxes and event.button == 1:  # Left click
                    if self.start_point is None:
                        # Start drawing bbox
                        self.start_point = (event.xdata, event.ydata)
                        print(f"Started bbox at ({event.xdata:.0f}, {event.ydata:.0f}) - Disease: {self.current_disease}")
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
                                'height': height,
                                'type': 'bbox'  # Explicitly mark as bbox
                            }
                            self.annotations.append(annotation)
                            
                            # Update session counts
                            if self.current_disease not in self.session_counts:
                                self.session_counts[self.current_disease] = 0
                            self.session_counts[self.current_disease] += 1
                            
                            # Color based on disease type
                            color_map = {
                                'bacterial_spot': 'darkred',
                                'early_blight': 'orange',
                                'late_blight': 'brown',
                                'leaf_mold': 'purple',
                                'septoria_leaf_spot': 'yellow',
                                'spider_mites': 'red',
                                'target_spot': 'pink',
                                'yellow_leaf_curl_virus': 'gold',
                                'mosaic_virus_plant': 'lime'
                            }
                            color = color_map.get(self.current_disease, 'red')
                            
                            # Draw the bbox
                            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, 
                                                linewidth=2, edgecolor=color, facecolor='none')
                            self.ax.add_patch(rect)
                            self.annotation_patches.append(rect)
                            
                            # Add annotation label with disease name
                            label_text = f"{len(self.annotations)}: {self.current_disease}"
                            text = self.ax.text(x1, y1-5, label_text, 
                                            color=color, fontweight='bold', fontsize=9,
                                            bbox=dict(boxstyle='round,pad=0.3', 
                                                    facecolor='white', alpha=0.7))
                            self.annotation_patches.append(text)
                            
                            # Update title and counts
                            self.update_title()
                            self.parent.update_counts_display(self.fig, self.session_counts, "disease_detection")
                            
                            print(f"Added {self.current_disease} annotation #{len(self.annotations)}")
                            print(f"Total {self.current_disease}: {self.session_counts.get(self.current_disease, 0)}")
                        else:
                            print("Bbox too small, minimum size is 10x10 pixels")
                        
                        self.start_point = None
                        
            def onkey(self, event):
                # Disease selection keys (1-9 for quick access)
                if event.key in '123456789':
                    try:
                        idx = int(event.key) - 1
                        if idx < len(self.available_diseases):
                            self.current_disease = self.available_diseases[idx]
                            self.update_title()
                            print(f"Switched to: {self.current_disease}")
                    except:
                        pass
                elif event.key == '0':
                    if len(self.available_diseases) >= 10:
                        self.current_disease = self.available_diseases[9]
                        self.update_title()
                        print(f"Switched to: {self.current_disease}")
                elif event.key == 'd':
                    self.show_disease_menu()
                elif event.key == 'u' or event.key == 'ctrl+z':
                    self.undo_last_annotation()
                elif event.key == 'c':
                    self.clear_all_annotations()
                elif event.key == 'h':
                    self.show_help()
                    
            def show_disease_menu(self):
                """Show interactive disease selection menu"""
                print("\n" + "="*60)
                print("CHANGE DISEASE")
                print("="*60)
                for i, disease in enumerate(self.available_diseases, 1):
                    marker = " <-- CURRENT" if disease == self.current_disease else ""
                    count = self.session_counts.get(disease, 0)
                    print(f"{i:2d}. {disease:<25} (count: {count}){marker}")
                print("="*60)
                try:
                    choice = input("Enter disease number: ").strip()
                    idx = int(choice) - 1
                    if 0 <= idx < len(self.available_diseases):
                        self.current_disease = self.available_diseases[idx]
                        self.update_title()
                        print(f"Switched to: {self.current_disease}")
                except:
                    print("Invalid selection")
            
            def update_title(self):
                """Update plot title with current disease"""
                if self.ax:
                    self.ax.set_title(
                        f"DISEASE LABELING - Current: {self.current_disease.upper()}\n" +
                        f"Image: {Path(image_path).name} | Keys 1-9: quick switch | 'd': menu | 'u': undo | 'h': help",
                        fontsize=10, fontweight='bold'
                    )
                    plt.draw()
                    
            def undo_last_annotation(self):
                """Undo the last annotation"""
                if self.annotations:
                    # Get the class of the annotation being removed
                    removed_annotation = self.annotations.pop()
                    removed_class = removed_annotation['class']
                    
                    # Update session counts
                    if removed_class in self.session_counts and self.session_counts[removed_class] > 0:
                        self.session_counts[removed_class] -= 1
                    
                    # Remove last visual elements (rectangle + text)
                    if len(self.annotation_patches) >= 2:
                        text_patch = self.annotation_patches.pop()
                        rect_patch = self.annotation_patches.pop()
                        text_patch.remove()
                        rect_patch.remove()
                    
                    # Update counts display
                    self.parent.update_counts_display(self.fig, self.session_counts, "disease_detection")
                    
                    print(f"Undid last annotation ({removed_class}). {len(self.annotations)} annotations remaining.")
                    print(f"Total {removed_class}: {self.session_counts.get(removed_class, 0)}")
                else:
                    print("No annotations to undo.")
                    
            def clear_all_annotations(self):
                """Clear all annotations"""
                if self.annotations:
                    response = input("Clear all annotations? (y/n): ").strip().lower()
                    if response == 'y':
                        # Restore original counts
                        for annotation in self.annotations:
                            class_name = annotation['class']
                            if class_name in self.session_counts and self.session_counts[class_name] > 0:
                                self.session_counts[class_name] -= 1
                        
                        self.annotations.clear()
                        
                        # Remove all visual elements
                        for patch in self.annotation_patches:
                            patch.remove()
                        self.annotation_patches.clear()
                        
                        # Update counts display
                        self.parent.update_counts_display(self.fig, self.session_counts, "disease_detection")
                        
                        print("All annotations cleared.")
                else:
                    print("No annotations to clear.")
                    
            def show_help(self):
                """Show help information"""
                help_text = f"""
    ========================================
    DISEASE LABELING CONTROLS
    ========================================
    CURRENT DISEASE: {self.current_disease}
    
    DISEASE SELECTION:
    - Keys 1-9: Quick switch to disease (1=healthy, 2=bacterial_spot, etc.)
    - 'd': Show disease selection menu
    
    LABELING:
    - Left click: Start bounding box
    - Left click again: Finish bounding box
    
    EDITING:
    - 'u': Undo last annotation
    - 'c': Clear all annotations
    
    OTHER:
    - 'h': Show this help
    - Close window: Save and continue
    
    AVAILABLE DISEASES:
    """
                for i, disease in enumerate(self.available_diseases, 1):
                    help_text += f"\n    {i}. {disease}"
                
                help_text += "\n\n    Draw bounding boxes around affected areas."
                help_text += "\n    Switch diseases with number keys for multi-disease images."
                print(help_text)
        
        # Create interactive plot with larger figure to accommodate counts
        fig, ax = plt.subplots(figsize=(16, 10))
        ax.imshow(image)
        
        selector = DiseaseBBoxSelector(image, current_disease, self)
        selector.fig = fig
        selector.ax = ax
        selector.update_title()
        
        # Initial counts display
        self.update_counts_display(fig, current_counts, "disease_detection")
        
        fig.canvas.mpl_connect('button_press_event', selector.onclick)
        fig.canvas.mpl_connect('key_press_event', selector.onkey)
        
        print("\n" + "="*70)
        print("DISEASE DETECTION LABELING")
        print("="*70)
        print(f"Current disease: {current_disease}")
        print("\nQUICK KEYS:")
        for i in range(min(9, len(self.disease_classes))):
            print(f"  {i+1} = {self.disease_classes[i]}")
        print("\nOTHER CONTROLS:")
        print("  'd' = Change disease")
        print("  'u' = Undo last")
        print("  'h' = Show help")
        print("  Left click = Draw bounding box")
        print("="*70)
        print(self.format_counts_display(current_counts, "disease_detection"))
        print("="*70)
        
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
                    self.logger.error(f"Could not load image: {image_path}")
                    return None
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            return image
            
        except Exception as e:
            self.logger.error(f"Error loading image {image_path}: {e}")
            return None
    
    def label_tomato_detection_data(self):
        """Interactive labeling for tomato vs plant detection with enhanced counting"""
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
        
        # Get initial counts
        initial_counts = self.get_current_annotation_counts("tomato_detection")
        
        self.logger.info(f"Found {len(image_files)} unique images to label")
        self.logger.info("Current annotation counts:")
        for class_name, count in initial_counts.items():
            self.logger.info(f"  {class_name}: {count}")
        
        labeled_count = 0
        total_annotations = 0
        skipped_count = 0
        session_start_count = sum(initial_counts.values())
        
        for i, image_path in enumerate(image_files):
            print(f"\n--- Labeling image {i+1}/{len(image_files)}: {image_path.name} ---")
            
            # Check if already labeled
            label_file = labeled_data_dir / f"{image_path.stem}.json"
            if label_file.exists():
                print(f"Image already labeled: {label_file.name}")
                response = input(f"Re-label this image? (y/n/s to skip all remaining): ").strip().lower()
                if response == 's':
                    print("Skipping all remaining labeled images...")
                    break
                elif response != 'y':
                    skipped_count += 1
                    continue
            
            # Show current progress
            current_counts = self.get_current_annotation_counts("tomato_detection")
            current_total = sum(current_counts.values())
            session_annotations = current_total - session_start_count
            
            print(f"Progress: {session_annotations} annotations added this session")
            
            # Label the image
            try:
                annotations = self.create_segmentation_interface_tomato(str(image_path))
                
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
                    
            except Exception as e:
                self.logger.error(f"Error labeling {image_path.name}: {e}")
                continue
        
        # Final summary
        final_counts = self.get_current_annotation_counts("tomato_detection")
        final_total = sum(final_counts.values())
        session_total = final_total - session_start_count
        
        labeling_time = (datetime.now() - labeling_start).total_seconds()
        
        self.logger.info("="*60)
        self.logger.info("LABELING SESSION SUMMARY")
        self.logger.info("="*60)
        self.logger.info(f"Session time: {labeling_time:.2f}s")
        self.logger.info(f"Images labeled: {labeled_count}")
        self.logger.info(f"Images skipped: {skipped_count}")
        self.logger.info(f"Annotations added this session: {session_total}")
        self.logger.info(f"Total annotations in dataset: {final_total}")
        self.logger.info("\nFinal counts by class:")
        for class_name, count in final_counts.items():
            initial_count = initial_counts.get(class_name, 0)
            added = count - initial_count
            self.logger.info(f"  {class_name}: {count} (+{added})")
        self.logger.info("="*60)
    
    def label_disease_detection_data(self, disease: Optional[str] = None, free_labeling: bool = False):
        """
        Interactive labeling for disease detection (BBOX ONLY)
        
        Args:
            disease: Specific disease to label (if provided, uses folder-based labeling)
            free_labeling: If True, allows user to select disease for each image
        """
        labeling_start = datetime.now()
        self.logger.info(f"Starting disease detection labeling...")
        
        raw_images_dir = self.project_dir / "raw_images" / "disease_detection"
        labeled_data_dir = self.project_dir / "labeled_data" / "disease_detection"
        
        if not raw_images_dir.exists():
            self.logger.error(f"Raw images directory not found: {raw_images_dir}")
            self.logger.info("Please organize images in raw_images/disease_detection/[disease_name]/")
            return
        
        # Get initial counts
        initial_counts = self.get_current_annotation_counts("disease_detection")
        
        # Determine labeling mode
        if free_labeling:
            # Free labeling mode: all images in main folder, user selects disease per image
            self.logger.info("Free labeling mode: You'll select the disease for each image")
            image_files = self.find_images(raw_images_dir)
            if not image_files:
                self.logger.error(f"No images found in {raw_images_dir}")
                return
            
            labeled_count = 0
            skipped_count = 0
            session_start_count = sum(initial_counts.values())
            
            for i, image_path in enumerate(image_files):
                print(f"\n--- Image {i+1}/{len(image_files)}: {image_path.name} ---")
                
                # Check if already labeled (look for any disease variant)
                existing_labels = list(labeled_data_dir.glob(f"{image_path.stem}_*.json"))
                if existing_labels:
                    print(f"Image has {len(existing_labels)} existing label(s)")
                    response = input(f"Re-label this image? (y/n/s to skip remaining): ").strip().lower()
                    if response == 's':
                        print("Skipping all remaining labeled images...")
                        break
                    elif response != 'y':
                        skipped_count += 1
                        continue
                
                # Label with disease selection (None = user selects)
                annotations = self.create_labeling_interface_disease(str(image_path), current_disease=None)
                
                if annotations:
                    # Get the disease(s) from annotations
                    diseases_in_image = set(ann['class'] for ann in annotations)
                    
                    # Save separate file for each disease
                    for disease_in_annotations in diseases_in_image:
                        disease_annotations = [ann for ann in annotations if ann['class'] == disease_in_annotations]
                        
                        label_file = labeled_data_dir / f"{image_path.stem}_{disease_in_annotations}.json"
                        label_data = {
                            'image_path': str(image_path),
                            'image_name': image_path.name,
                            'task': 'disease_detection',
                            'disease': disease_in_annotations,
                            'annotations': disease_annotations,
                            'labeled_date': datetime.now().isoformat()
                        }
                        
                        with open(label_file, 'w', encoding='utf-8') as f:
                            json.dump(label_data, f, indent=2)
                    
                    # Copy image once to labeled data directory
                    dest_image = labeled_data_dir / f"{image_path.stem}{image_path.suffix}"
                    if not dest_image.exists():
                        shutil.copy2(image_path, dest_image)
                    
                    labeled_count += 1
                    self.logger.info(f"Saved annotations for {image_path.name} ({len(diseases_in_image)} disease types)")
                else:
                    self.logger.info(f"No annotations for {image_path.name}")
            
        else:
            # Folder-based labeling mode: images organized in disease folders
            if disease:
                disease_folders = [raw_images_dir / disease] if (raw_images_dir / disease).exists() else []
                if not disease_folders:
                    self.logger.error(f"Disease folder not found: {disease}")
                    return
            else:
                disease_folders = [f for f in raw_images_dir.iterdir() if f.is_dir()]
            
            if not disease_folders:
                self.logger.error(f"No disease folders found in {raw_images_dir}")
                self.logger.info("Tip: Use --free-labeling for mixed disease images")
                return
            
            labeled_count = 0
            skipped_count = 0
            session_start_count = sum(initial_counts.values())
            
            for disease_folder in disease_folders:
                disease_name = disease_folder.name
                
                # Validate disease name
                if disease_name not in self.disease_classes:
                    self.logger.warning(f"Unknown disease folder: {disease_name}")
                    response = input(f"Continue with '{disease_name}'? (y/n): ").strip().lower()
                    if response != 'y':
                        continue
                
                self.logger.info(f"Labeling disease: {disease_name}")
                
                image_files = self.find_images(disease_folder)
                if not image_files:
                    self.logger.warning(f"No images found for {disease_name}")
                    continue
                
                for i, image_path in enumerate(image_files):
                    print(f"\n--- {disease_name} - Image {i+1}/{len(image_files)}: {image_path.name} ---")
                    
                    # Check if already labeled
                    label_file = labeled_data_dir / f"{image_path.stem}_{disease_name}.json"
                    if label_file.exists():
                        response = input(f"Already labeled. Re-label? (y/n/s to skip remaining): ").strip().lower()
                        if response == 's':
                            print("Skipping remaining images in this disease...")
                            break
                        elif response != 'y':
                            skipped_count += 1
                            continue
                    
                    # Label with pre-selected disease
                    annotations = self.create_labeling_interface_disease(str(image_path), current_disease=disease_name)
                    
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
                        
                        # Copy image to labeled data directory with disease suffix
                        shutil.copy2(image_path, labeled_data_dir / f"{image_path.stem}_{disease_name}{image_path.suffix}")
                        
                        labeled_count += 1
                        self.logger.info(f"Saved {len(annotations)} annotations for {image_path.name}")
                    else:
                        self.logger.info(f"No annotations for {image_path.name}")
        
        # Final summary
        final_counts = self.get_current_annotation_counts("disease_detection")
        final_total = sum(final_counts.values())
        session_total = final_total - session_start_count
        
        labeling_time = (datetime.now() - labeling_start).total_seconds()
        
        self.logger.info("="*60)
        self.logger.info("DISEASE LABELING SESSION SUMMARY")
        self.logger.info("="*60)
        self.logger.info(f"Session time: {labeling_time:.2f}s")
        self.logger.info(f"Images labeled: {labeled_count}")
        self.logger.info(f"Images skipped: {skipped_count}")
        self.logger.info(f"Annotations added this session: {session_total}")
        self.logger.info(f"Total annotations in dataset: {final_total}")
        self.logger.info("\nFinal counts by class:")
        for class_name, count in sorted(final_counts.items()):
            initial_count = initial_counts.get(class_name, 0)
            added = count - initial_count
            self.logger.info(f"  {class_name}: {count} (+{added})")
        self.logger.info("="*60)
    
    def find_images(self, directory: Path) -> List[Path]:
        """Find all image files in a directory (fixed duplicate detection)"""
        image_extensions = ['.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp']
        image_files = []
        
        # Get all files in directory
        all_files = list(directory.glob("*"))
        
        # Filter by extension (case-insensitive)
        for file_path in all_files:
            if file_path.is_file():  # Only process files, not directories
                file_ext = file_path.suffix.lower()  # Convert to lowercase for comparison
                if file_ext in image_extensions:
                    image_files.append(file_path)
        
        return sorted(list(set(image_files)))  # Remove any remaining duplicates and sort
    
    def convert_segmentation_to_yolo(self, annotations: List[Dict], output_dir: str, 
                                 image_name: str, class_mapping: Dict[str, int]):
        """
        Convert polygon segmentation annotations to YOLO segmentation format
        YOLO format: class_id x1 y1 x2 y2 x3 y3 ... (normalized coordinates)
        """
        if not annotations:
            return
            
        label_path = Path(output_dir) / f"{Path(image_name).stem}.txt"
        
        with open(label_path, 'w') as f:
            for ann in annotations:
                # Only process polygon annotations
                if ann.get('type') != 'polygon':
                    continue
                    
                segmentation = ann.get('segmentation', [])
                if not segmentation or len(segmentation) < 3:
                    self.logger.warning(f"Skipping invalid polygon with {len(segmentation)} points")
                    continue
                    
                class_name = ann['class']
                img_width = ann['width']
                img_height = ann['height']
                
                # Get class ID
                class_id = class_mapping.get(class_name, 0)
                
                # Normalize and flatten coordinates
                normalized_coords = []
                for point in segmentation:
                    if len(point) >= 2:  # Ensure point has x, y
                        x_norm = max(0.0, min(1.0, point[0] / img_width))
                        y_norm = max(0.0, min(1.0, point[1] / img_height))
                        normalized_coords.extend([f"{x_norm:.6f}", f"{y_norm:.6f}"])
                
                # Write line: class_id x1 y1 x2 y2 x3 y3 ...
                if normalized_coords:
                    line = f"{class_id} " + " ".join(normalized_coords)
                    f.write(line + "\n")
                else:
                    self.logger.warning(f"No valid coordinates for annotation in {image_name}")
    
    def prepare_tomato_segmentation_dataset(self, train_split: float = 0.8):
        """Prepare YOLO segmentation dataset for tomato detection"""
        self.logger.info("Preparing tomato segmentation dataset...")
        
        labeled_data_dir = self.project_dir / "labeled_data" / "tomato_detection"
        dataset_dir = self.project_dir / "datasets" / "tomato_detection"
        
        # Find labeled data
        label_files = list(labeled_data_dir.glob("*.json"))
        if not label_files:
            self.logger.error("No labeled tomato data found")
            return False
        
        self.logger.info(f"Found {len(label_files)} labeled images")
        
        # Verify annotations are segmentation format
        polygon_count = 0
        bbox_count = 0
        for label_file in label_files:
            try:
                with open(label_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                for ann in data.get('annotations', []):
                    if ann.get('type') == 'polygon':
                        polygon_count += 1
                    else:
                        bbox_count += 1
            except Exception as e:
                self.logger.error(f"Error reading {label_file}: {e}")
        
        if polygon_count == 0:
            self.logger.error("No polygon annotations found! Use segmentation labeling interface.")
            return False
        
        if bbox_count > 0:
            self.logger.warning(f"Found {bbox_count} bbox annotations mixed with {polygon_count} polygons - skipping bboxes")
        
        self.logger.info(f"Processing {polygon_count} polygon annotations")
        
        # Class mapping
        class_mapping = {
            'mature_tomato': 0,
            'immature_tomato': 1,
            'tomato_plant': 2,
            'not_tomato': 3
        }
        
        # Split data
        random.shuffle(label_files)
        split_idx = int(len(label_files) * train_split)
        train_files = label_files[:split_idx]
        val_files = label_files[split_idx:]
        
        self.logger.info(f"Dataset split - Train: {len(train_files)}, Val: {len(val_files)}")
        
        # Clear existing labels
        for split in ['train', 'val']:
            label_dir = dataset_dir / "labels" / split
            if label_dir.exists():
                for old_label in label_dir.glob("*.txt"):
                    old_label.unlink()
        
        # Process training data
        train_count = 0
        for label_file in train_files:
            if self.process_segmentation_file(label_file, dataset_dir, 'train', class_mapping):
                train_count += 1
        
        # Process validation data
        val_count = 0
        for label_file in val_files:
            if self.process_segmentation_file(label_file, dataset_dir, 'val', class_mapping):
                val_count += 1
        
        self.logger.info(f"Processed {train_count} train and {val_count} val images")
        
        # Create dataset config
        dataset_config = {
            'path': str(dataset_dir.absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'nc': 4,
            'names': ['mature_tomato', 'immature_tomato', 'tomato_plant', 'not_tomato']
        }
        
        config_path = dataset_dir / "data.yaml"
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(dataset_config, f)
        
        self.logger.info(f"Tomato segmentation dataset prepared: {config_path}")
        
        # Verify a sample label file
        sample_label = dataset_dir / "labels" / "train" / f"{train_files[0].stem}.txt"
        if sample_label.exists():
            with open(sample_label, 'r') as f:
                first_line = f.readline().strip()
                parts = first_line.split()
                if len(parts) >= 7:  # class_id + at least 3 points (6 coords)
                    self.logger.info(f"Sample label verified: {len(parts)-1} coordinates")
                else:
                    self.logger.error(f"Sample label format incorrect: {first_line}")
        
        return True
    

    def process_segmentation_file(self, label_file: Path, dataset_dir: Path, 
                              split: str, class_mapping: Dict[str, int]) -> bool:
        """Process a single labeled file for YOLO segmentation dataset"""
        try:
            with open(label_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            image_name = data['image_name']
            annotations = data['annotations']
            
            # Filter only polygon annotations
            polygon_annotations = [ann for ann in annotations if ann.get('type') == 'polygon']
            
            if not polygon_annotations:
                self.logger.warning(f"No polygon annotations in {label_file.name}")
                return False
            
            # Copy image
            src_image = label_file.parent / image_name
            if not src_image.exists():
                self.logger.warning(f"Source image not found: {src_image}")
                return False
            
            dst_image = dataset_dir / "images" / split / image_name
            dst_image.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_image, dst_image)
            
            # Convert and save segmentation labels
            label_dir = dataset_dir / "labels" / split
            label_dir.mkdir(parents=True, exist_ok=True)
            self.convert_segmentation_to_yolo(
                polygon_annotations,
                label_dir,
                image_name,
                class_mapping
            )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error processing {label_file}: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False

    def prepare_disease_dataset(self, train_split: float = 0.8, specific_disease: Optional[str] = None):
        """
        Prepare YOLO dataset for disease detection (BBOX ONLY - NOT SEGMENTATION)
        This is completely separate from tomato segmentation
        """
        self.logger.info("="*60)
        self.logger.info("PREPARING DISEASE DETECTION DATASET (BBOX FORMAT)")
        self.logger.info("="*60)
        
        labeled_data_dir = self.project_dir / "labeled_data" / "disease_detection"
        dataset_dir = self.project_dir / "datasets" / "disease_detection"
        
        # Find labeled data
        if specific_disease:
            label_files = list(labeled_data_dir.glob(f"*_{specific_disease}.json"))
            self.logger.info(f"Preparing dataset for specific disease: {specific_disease}")
        else:
            label_files = list(labeled_data_dir.glob("*.json"))
            self.logger.info("Preparing dataset for all diseases")
        
        if not label_files:
            self.logger.error("No labeled disease data found!")
            self.logger.info("Please label disease data first using: --label-disease")
            return False
        
        self.logger.info(f"Found {len(label_files)} labeled image files")
        
        # Verify all annotations are BBOX format
        bbox_count = 0
        polygon_count = 0
        all_classes = set()
        
        for label_file in label_files:
            try:
                with open(label_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Verify task type
                if data.get('task') != 'disease_detection':
                    self.logger.warning(f"Skipping non-disease file: {label_file.name}")
                    continue
                
                annotations = data.get('annotations', [])
                for ann in annotations:
                    ann_type = ann.get('type', 'bbox')  # Default to bbox if not specified
                    if ann_type == 'bbox' or 'bbox' in ann:
                        bbox_count += 1
                        all_classes.add(ann['class'])
                    elif ann_type == 'polygon':
                        polygon_count += 1
                        self.logger.warning(f"Found polygon in disease file {label_file.name} - will skip")
                        
            except Exception as e:
                self.logger.error(f"Error reading {label_file}: {e}")
                continue
        
        if bbox_count == 0:
            self.logger.error("No bounding box annotations found!")
            self.logger.error("Disease detection uses BBOX format, not segmentation.")
            self.logger.info("Use --label-disease to create proper bbox annotations")
            return False
        
        if polygon_count > 0:
            self.logger.warning(f"Found {polygon_count} polygon annotations - these will be skipped")
            self.logger.warning("Disease detection uses BBOX format only")
        
        self.logger.info(f"Processing {bbox_count} bounding box annotations")
        
        # Create class mapping
        sorted_classes = sorted(list(all_classes))
        class_mapping = {cls: i for i, cls in enumerate(sorted_classes)}
        
        self.logger.info(f"Disease classes found: {sorted_classes}")
        self.logger.info("Class mapping:")
        for cls, idx in class_mapping.items():
            self.logger.info(f"  {idx}: {cls}")
        
        # Split data
        random.shuffle(label_files)
        split_idx = int(len(label_files) * train_split)
        train_files = label_files[:split_idx]
        val_files = label_files[split_idx:]
        
        self.logger.info(f"Dataset split - Train: {len(train_files)}, Val: {len(val_files)}")
        
        # Clear existing labels in disease detection dataset
        for split in ['train', 'val']:
            label_dir = dataset_dir / "labels" / split
            if label_dir.exists():
                for old_label in label_dir.glob("*.txt"):
                    old_label.unlink()
        
        # Process training data
        train_count = 0
        for label_file in train_files:
            if self.process_disease_bbox_file(label_file, dataset_dir, 'train', class_mapping):
                train_count += 1
        
        # Process validation data
        val_count = 0
        for label_file in val_files:
            if self.process_disease_bbox_file(label_file, dataset_dir, 'val', class_mapping):
                val_count += 1
        
        self.logger.info(f"Successfully processed: Train={train_count}, Val={val_count}")
        
        # Create dataset config FOR DETECTION (not segmentation)
        dataset_config = {
            'path': str(dataset_dir.absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'nc': len(sorted_classes),
            'names': sorted_classes
        }
        
        config_path = dataset_dir / "data.yaml"
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(dataset_config, f)
        
        self.logger.info(f"Disease detection dataset prepared: {config_path}")
        self.logger.info("="*60)
        
        # Verify a sample label file
        sample_labels = list((dataset_dir / "labels" / "train").glob("*.txt"))
        if sample_labels:
            with open(sample_labels[0], 'r') as f:
                first_line = f.readline().strip()
                parts = first_line.split()
                if len(parts) == 5:  # class_id x_center y_center width height
                    self.logger.info(f"Sample BBOX label verified: class={parts[0]}, coords={parts[1:5]}")
                else:
                    self.logger.warning(f"Unexpected label format: {first_line}")
        
        return True
    
    def process_disease_bbox_file(self, label_file: Path, dataset_dir: Path, 
                              split: str, class_mapping: Dict[str, int]) -> bool:
        """
        Process a single disease detection label file (BBOX ONLY)
        Completely separate from segmentation processing
        """
        try:
            with open(label_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Verify this is disease detection data
            if data.get('task') != 'disease_detection':
                return False
            
            image_name = data['image_name']
            annotations = data.get('annotations', [])
            disease = data.get('disease', '')
            
            # Filter ONLY bbox annotations
            bbox_annotations = []
            for ann in annotations:
                ann_type = ann.get('type', 'bbox')
                if ann_type == 'bbox' or ('bbox' in ann and 'segmentation' not in ann):
                    bbox_annotations.append(ann)
            
            if not bbox_annotations:
                self.logger.warning(f"No bbox annotations in {label_file.name}")
                return False
            
            # Find source image
            src_image = label_file.parent / image_name
            if not src_image.exists():
                # Try with disease suffix
                modified_name = f"{Path(image_name).stem}_{disease}{Path(image_name).suffix}"
                src_image = label_file.parent / modified_name
            
            if not src_image.exists():
                # Try without disease suffix
                base_name = Path(image_name).stem
                if '_' in base_name:
                    # Remove disease suffix
                    base_name_parts = base_name.rsplit('_', 1)
                    clean_name = base_name_parts[0] + Path(image_name).suffix
                    src_image = label_file.parent / clean_name
            
            if not src_image.exists():
                self.logger.warning(f"Source image not found for {label_file.name}")
                return False
            
            # Copy image to dataset
            dst_image = dataset_dir / "images" / split / image_name
            dst_image.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_image, dst_image)
            
            # Convert and save BBOX labels
            label_dir = dataset_dir / "labels" / split
            label_dir.mkdir(parents=True, exist_ok=True)
            self.convert_bbox_to_yolo_format(
                bbox_annotations,
                label_dir,
                image_name,
                class_mapping
            )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error processing {label_file}: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False
    
    def convert_bbox_to_yolo_format(self, annotations: List[Dict], output_dir: str, 
                                 image_name: str, class_mapping: Dict[str, int]):
        """
        Convert BBOX annotations to YOLO detection format
        Format: class_id x_center y_center width height (all normalized)
        This is SEPARATE from segmentation format
        """
        if not annotations:
            return
            
        label_path = Path(output_dir) / f"{Path(image_name).stem}.txt"
        
        lines_written = 0
        with open(label_path, 'w') as f:
            for ann in annotations:
                # Only process bbox annotations
                ann_type = ann.get('type', 'bbox')
                if ann_type != 'bbox' and 'bbox' not in ann:
                    continue
                
                bbox = ann.get('bbox')
                if not bbox or len(bbox) != 4:
                    continue
                
                class_name = ann['class']
                img_width = ann['width']
                img_height = ann['height']
                
                # Get class ID
                class_id = class_mapping.get(class_name, 0)
                
                # Convert to YOLO format (normalized center coordinates + width/height)
                x_center = ((bbox[0] + bbox[2]) / 2) / img_width
                y_center = ((bbox[1] + bbox[3]) / 2) / img_height
                width = (bbox[2] - bbox[0]) / img_width
                height = (bbox[3] - bbox[1]) / img_height
                
                # Ensure values are in valid range
                x_center = max(0.0, min(1.0, x_center))
                y_center = max(0.0, min(1.0, y_center))
                width = max(0.0, min(1.0, width))
                height = max(0.0, min(1.0, height))
                
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
                lines_written += 1
        
        if lines_written == 0:
            # Remove empty label file
            label_path.unlink()
            self.logger.warning(f"No valid bbox annotations written for {image_name}")

    def train_tomato_segmentation_model(self, epochs: int = 100, batch_size: int = 16, 
                                    model_size: str = 's'):
        """
        Train tomato segmentation model (not detection)
        Uses YOLOv8-seg for instance segmentation
        """
        if not YOLO_AVAILABLE:
            self.logger.error("ultralytics not available")
            return None
        
        training_start = datetime.now()
        
        # Optimize batch size for available hardware
        optimal_batch_size = self.get_optimal_batch_size(batch_size)
        if optimal_batch_size != batch_size:
            self.logger.info(f"Adjusted batch size from {batch_size} to {optimal_batch_size} for {self.device}")
            batch_size = optimal_batch_size
        
        self.logger.info(f"Starting tomato SEGMENTATION training: {epochs} epochs, batch {batch_size}, model {model_size}")
        self.logger.info(f"Training device: {self.device}")
        
        # Prepare dataset
        if not self.prepare_tomato_segmentation_dataset():
            return None
        
        # Initialize SEGMENTATION model (note the -seg suffix)
        model = YOLO(f'yolov8{model_size}-seg.pt')
        
        # Monitor initial GPU state
        if self.device == 'cuda':
            self.monitor_gpu_usage()
        
        try:
            # Train with device specification
            results = model.train(
                data=str(self.project_dir / "datasets" / "tomato_detection" / "data.yaml"),
                epochs=epochs,
                batch=batch_size,
                imgsz=640,
                device=self.device,
                project=str(self.project_dir / "training_results"),
                name="tomato_segmentation",
                save=True,
                plots=True,
                patience=50,
                save_period=10,
            )
            
            # Save best model
            best_model_path = self.project_dir / "models" / "tomato_segmentation_best.pt"
            shutil.copy2(results.save_dir / "weights" / "best.pt", best_model_path)
            
            training_time = (datetime.now() - training_start).total_seconds()
            session_time = (datetime.now() - self.start_time).total_seconds()
            
            self.logger.info(f"Tomato segmentation training complete: {training_time:.2f}s")
            self.logger.info(f"Total session time: {session_time:.2f}s")
            self.logger.info(f"Model saved: {best_model_path}")
            
            # Final GPU memory check
            if self.device == 'cuda':
                self.monitor_gpu_usage()
            
            return results
            
        except torch.cuda.OutOfMemoryError:
            self.logger.error("GPU out of memory! Try reducing batch size.")
            self.logger.info(f"Current batch size: {batch_size}. Try: {batch_size // 2}")
            return None
        except Exception as e:
            self.logger.error(f"Training error: {e}")
            return None
    
    def train_disease_model(self, epochs: int = 100, batch_size: int = 16, model_size: str = 's', 
                           specific_disease: Optional[str] = None, incremental: bool = False):
        """
        Train disease detection model with GPU support (DETECTION MODEL - NOT SEGMENTATION)
        Uses standard YOLOv8 for bounding box detection
        """
        if not YOLO_AVAILABLE:
            self.logger.error("ultralytics not available")
            return None
        
        training_start = datetime.now()
        
        # Optimize batch size for available hardware
        optimal_batch_size = self.get_optimal_batch_size(batch_size)
        if optimal_batch_size != batch_size:
            self.logger.info(f"Adjusted batch size from {batch_size} to {optimal_batch_size} for {self.device}")
            batch_size = optimal_batch_size
        
        disease_info = f" for {specific_disease}" if specific_disease else " for all diseases"
        mode_info = " (incremental)" if incremental else ""
        
        self.logger.info("="*60)
        self.logger.info(f"TRAINING DISEASE DETECTION MODEL{disease_info}{mode_info}")
        self.logger.info("="*60)
        self.logger.info(f"Model type: DETECTION (bbox)")
        self.logger.info(f"Epochs: {epochs}, Batch size: {batch_size}, Model size: {model_size}")
        self.logger.info(f"Training device: {self.device}")
        
        # Prepare dataset
        if not self.prepare_disease_dataset(specific_disease=specific_disease):
            self.logger.error("Failed to prepare disease dataset")
            return None
        
        # Initialize DETECTION model (standard YOLOv8, NOT -seg)
        if incremental and (self.project_dir / "models" / "disease_detection_best.pt").exists():
            model = YOLO(str(self.project_dir / "models" / "disease_detection_best.pt"))
            self.logger.info("Loading existing model for incremental training")
        else:
            model = YOLO(f'yolov8{model_size}.pt')  # Standard detection model
            self.logger.info(f"Starting fresh training with YOLOv8{model_size} DETECTION model")
        
        # Monitor initial GPU state
        if self.device == 'cuda':
            self.monitor_gpu_usage()
        
        try:
            # Train
            training_name = f"disease_detection_{specific_disease}" if specific_disease else "disease_detection_all"
            
            results = model.train(
                data=str(self.project_dir / "datasets" / "disease_detection" / "data.yaml"),
                epochs=epochs,
                batch=batch_size,
                imgsz=640,
                device=self.device,
                project=str(self.project_dir / "training_results"),
                name=training_name,
                save=True,
                plots=True,
                patience=50,
                save_period=10,
            )
            
            # Save best model
            best_model_path = self.project_dir / "models" / "disease_detection_best.pt"
            shutil.copy2(results.save_dir / "weights" / "best.pt", best_model_path)
            
            # Update disease training progress
            self.update_disease_progress(specific_disease, results)
            
            training_time = (datetime.now() - training_start).total_seconds()
            session_time = (datetime.now() - self.start_time).total_seconds()
            
            self.logger.info("="*60)
            self.logger.info(f"DISEASE TRAINING COMPLETE{disease_info}")
            self.logger.info("="*60)
            self.logger.info(f"Training time: {training_time:.2f}s")
            self.logger.info(f"Total session time: {session_time:.2f}s")
            self.logger.info(f"Model saved: {best_model_path}")
            self.logger.info("="*60)
            
            # Final GPU memory check
            if self.device == 'cuda':
                self.monitor_gpu_usage()
            
            return results
            
        except torch.cuda.OutOfMemoryError:
            self.logger.error("GPU out of memory! Try reducing batch size.")
            self.logger.info(f"Current batch size: {batch_size}. Try: {batch_size // 2}")
            return None
        except Exception as e:
            self.logger.error(f"Training error: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None
    
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
            self.logger.error("ultralytics not available")
            return
        
        if not os.path.exists(model_path):
            self.logger.error(f"Model not found: {model_path}")
            return
        
        self.logger.info(f"Testing {model_type} model: {model_path}")
        
        model = YOLO(model_path)
        
        # Find test images
        test_dir = Path(test_images_dir)
        if not test_dir.exists():
            self.logger.error(f"Test directory not found: {test_dir}")
            return
        
        test_images = self.find_images(test_dir)
        if not test_images:
            self.logger.error(f"No test images found in {test_dir}")
            return
        
        self.logger.info(f"Testing on {len(test_images)} images")
        
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
                    
                    self.logger.info(f"{image_path.name}: {num_detections} detections")
                    
            except Exception as e:
                self.logger.error(f"Error testing {image_path}: {e}")
        
        self.logger.info(f"Testing complete. Total detections: {total_detections}")
        self.logger.info(f"Results saved to: {results_dir}")



def main():
    """Main function with comprehensive command-line interface"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Tomato Disease Detection Training Pipeline - GPU Enabled\n" +
                   "Separate training for TOMATO SEGMENTATION and DISEASE DETECTION (bbox)",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--project", default="tomato_training_project", help="Project directory")
    
    # System check
    parser.add_argument("--check-gpu", action="store_true", help="Check GPU setup")
    
    # Labeling commands
    parser.add_argument("--label-tomato", action="store_true", 
                       help="Label tomato detection data (SEGMENTATION)")
    parser.add_argument("--label-disease", action="store_true", 
                       help="Label disease detection data (BBOX)")
    parser.add_argument("--disease", help="Specific disease to label/train")
    parser.add_argument("--free-labeling", action="store_true",
                       help="Free labeling mode: select disease per image (for mixed disease images)")
    
    # Training commands
    parser.add_argument("--train-tomato", action="store_true", 
                       help="Train tomato SEGMENTATION model")
    parser.add_argument("--train-disease", action="store_true", 
                       help="Train disease DETECTION model (bbox)")
    parser.add_argument("--all-diseases", action="store_true", help="Train on all diseases")
    parser.add_argument("--incremental", action="store_true", 
                       help="Incremental training (add to existing model)")
    
    # Testing commands
    parser.add_argument("--test-tomato", action="store_true", help="Test tomato model")
    parser.add_argument("--test-disease", action="store_true", help="Test disease model")
    parser.add_argument("--model", help="Model path for testing")
    parser.add_argument("--test-dir", default="test_images", help="Test images directory")
    
    # Training parameters
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--model-size", default="s", choices=['n', 's', 'm', 'l', 'x'], 
                       help="Model size")
    
    args = parser.parse_args()
    
    # Check GPU setup if requested
    if args.check_gpu:
        check_gpu_setup()
        return
    
    # Initialize trainer
    trainer = TomatoTrainingPipeline(args.project)
    
    # Execute commands
    if args.label_tomato:
        print("="*70)
        print("TOMATO SEGMENTATION LABELING")
        print("="*70)
        print("Mode: SEGMENTATION (polygons)")
        print("Classes: mature_tomato, immature_tomato, tomato_plant, not_tomato")
        print("="*70)
        trainer.label_tomato_detection_data()
    
    elif args.label_disease:
        print("="*70)
        print("DISEASE DETECTION LABELING")
        print("="*70)
        print("Mode: BOUNDING BOXES")
        if args.free_labeling:
            print("Labeling mode: FREE (select disease per image)")
            print("Tip: Use this for images with multiple diseases")
        elif args.disease:
            print(f"Labeling mode: SINGLE DISEASE ({args.disease})")
        else:
            print("Labeling mode: FOLDER-BASED (one disease per folder)")
        print("="*70)
        trainer.label_disease_detection_data(
            disease=args.disease,
            free_labeling=args.free_labeling
        )
    
    elif args.train_tomato:
        print("="*70)
        print("TRAINING TOMATO SEGMENTATION MODEL")
        print("="*70)
        print("Model type: YOLOv8-seg (instance segmentation)")
        print("Output: Precise polygon masks for each tomato/plant")
        print("="*70)
        trainer.train_tomato_segmentation_model(  
            epochs=args.epochs,
            batch_size=args.batch_size,
            model_size=args.model_size
        )
    
    elif args.train_disease:
        print("="*70)
        print("TRAINING DISEASE DETECTION MODEL")
        print("="*70)
        print("Model type: YOLOv8 (bounding box detection)")
        print("Output: Bounding boxes around diseased areas")
        disease = args.disease if not args.all_diseases else None
        print(f"Training: {disease if disease else 'ALL DISEASES'}")
        print("="*70)
        trainer.train_disease_model(
            epochs=args.epochs,
            batch_size=args.batch_size,
            model_size=args.model_size,
            specific_disease=disease,
            incremental=args.incremental
        )
    
    elif args.test_tomato:
        print("Testing tomato segmentation model...")
        model_path = args.model or f"{args.project}/models/tomato_segmentation_best.pt"
        trainer.test_model(model_path, args.test_dir, "tomato")
    
    elif args.test_disease:
        print("Testing disease detection model...")
        model_path = args.model or f"{args.project}/models/disease_detection_best.pt"
        trainer.test_model(model_path, args.test_dir, "disease")
    
    else:
        print("="*70)
        print("TOMATO DISEASE DETECTION TRAINING PIPELINE - GPU ENABLED")
        print("="*70)
        print("\nCHECK GPU SETUP:")
        print("   python tomato_training_pipeline.py --check-gpu")
        print("\n" + "="*70)
        print("TWO SEPARATE TRAINING PATHS:")
        print("="*70)
        print("\n1. TOMATO SEGMENTATION (precise polygon masks)")
        print("   - For: Counting, sizing, maturity classification")
        print("   - Model: YOLOv8-seg")
        print("   - Classes: mature_tomato, immature_tomato, tomato_plant, not_tomato")
        print("\n2. DISEASE DETECTION (bounding boxes)")
        print("   - For: Disease identification and localization")
        print("   - Model: YOLOv8")
        print("   - Classes: healthy + 10 disease types")
        print("\n" + "="*70)
        print("LABELING:")
        print("="*70)
        print("\nTOMATO SEGMENTATION:")
        print("   python tomato_training_pipeline.py --label-tomato")
        print("   -> Draw polygons around tomatoes and plants")
        print("\nDISEASE DETECTION:")
        print("   # Folder-based (one disease per folder)")
        print("   python tomato_training_pipeline.py --label-disease")
        print("   ")
        print("   # Single disease")
        print("   python tomato_training_pipeline.py --label-disease --disease bacterial_spot")
        print("   ")
        print("   # Free labeling (select disease per image)")
        print("   python tomato_training_pipeline.py --label-disease --free-labeling")
        print("   -> Draw bounding boxes, switch diseases with 1-9 keys")
        print("\n" + "="*70)
        print("TRAINING:")
        print("="*70)
        print("\nTOMATO SEGMENTATION:")
        print("   python tomato_training_pipeline.py --train-tomato --epochs 100")
        print("\nDISEASE DETECTION:")
        print("   # All diseases")
        print("   python tomato_training_pipeline.py --train-disease --all-diseases --epochs 100")
        print("   ")
        print("   # Single disease")
        print("   python tomato_training_pipeline.py --train-disease --disease bacterial_spot")
        print("   ")
        print("   # Incremental (add new disease)")
        print("   python tomato_training_pipeline.py --train-disease --disease early_blight --incremental")
        print("\n" + "="*70)
        print("TESTING:")
        print("="*70)
        print("   python tomato_training_pipeline.py --test-tomato --test-dir test_images/")
        print("   python tomato_training_pipeline.py --test-disease --test-dir test_images/")
        print("\n" + "="*70)
        print("KEY FEATURES:")
        print("="*70)
        print("   - Completely separate tomato and disease pipelines")
        print("   - Automatic GPU detection and optimization")
        print("   - Batch size adjustment based on GPU memory")
        print("   - Memory monitoring during training")
        print("   - Disease quick-switch keys (1-9)")
        print("   - Session annotation tracking")
        print("\nDIRECTORY STRUCTURE:")
        print("="*70)
        print("   raw_images/")
        print("   ├── tomato_detection/          # Tomato/plant images")
        print("   └── disease_detection/          # Disease images")
        print("       ├── healthy/")
        print("       ├── bacterial_spot/")
        print("       └── [other diseases]/")
        print("\nFor detailed instructions, see README.md")
        print("All logs saved to logs/ directory")
        print("="*70)
    
    trainer.log_session_end()

if __name__ == "__main__":
    # Check GPU setup on startup
    check_gpu_setup()
    
    # Run main program
    main()