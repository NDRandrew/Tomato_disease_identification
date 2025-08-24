#!/usr/bin/env python3
"""
Standalone Image Labeling Tool
Interactive labeling interface for tomato detection and disease annotation
"""

import os
import cv2
import numpy as np
import json
from pathlib import Path
from datetime import datetime
import logging
from typing import List, Dict, Optional
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class StandaloneLabelingTool:
    """
    Standalone tool for labeling images for tomato detection and disease classification
    """
    
    def __init__(self, 
                 input_dir: str = "images_to_label",
                 output_dir: str = "labeled_annotations",
                 log_dir: str = "logs"):
        """
        Initialize the labeling tool
        
        Args:
            input_dir: Directory containing images to label
            output_dir: Directory for saving labeled annotations
            log_dir: Directory for log files
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.log_dir = Path(log_dir)
        
        # Create directories
        self.input_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(exist_ok=True)
        self.log_dir.mkdir(exist_ok=True)
        
        # Set default mode before logging setup
        self.current_mode = 'tomato_detection'
        
        # Supported labeling modes
        self.labeling_modes = {
            'tomato_detection': {
                'classes': ['tomato', 'tomato_plant', 'not_tomato'],
                'colors': ['red', 'blue', 'yellow'],
                'keys': ['t', 'p', 'n'],
                'description': 'Label tomatoes, plants, and similar objects'
            },
            'disease_detection': {
                'classes': ['healthy', 'bacterial_spot', 'early_blight', 'late_blight', 
                           'leaf_mold', 'septoria_leaf_spot', 'spider_mites', 'target_spot',
                           'yellow_leaf_curl_virus', 'mosaic_virus', 'bacterial_canker'],
                'colors': ['green', 'red', 'orange', 'darkred', 'brown', 'purple', 
                          'pink', 'cyan', 'magenta', 'lime', 'coral'],
                'keys': ['h', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0'],
                'description': 'Label diseases on tomato plants/fruits'
            }
        }
        
        # Setup logging after everything is initialized
        self.setup_logging()
        
    def setup_logging(self):
        """Setup logging system"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.log_dir / f"labeling_session_{timestamp}.log"
        
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
        
        self.logger.info("="*60)
        self.logger.info("STANDALONE LABELING TOOL STARTED")
        self.logger.info("="*60)
        self.logger.info(f"Start time: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info(f"Input directory: {self.input_dir}")
        self.logger.info(f"Output directory: {self.output_dir}")
        self.logger.info(f"Current mode: {self.current_mode}")
    
    def log_session_end(self):
        """Log session end time and duration"""
        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds()
        self.logger.info("="*60)
        self.logger.info(f"SESSION ENDED: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info(f"Total duration: {duration:.2f}s")
        self.logger.info("="*60)
    
    def find_images(self, directory: Path) -> List[Path]:
        """Find all image files in a directory"""
        image_extensions = ['.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(directory.glob(f"*{ext}"))
            image_files.extend(directory.glob(f"*{ext.upper()}"))
        
        return sorted(image_files)
    
    def load_image_for_labeling(self, image_path: Path) -> Optional[np.ndarray]:
        """Load and prepare image for labeling interface"""
        try:
            if image_path.suffix.lower() in ['.tif', '.tiff']:
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
                except ImportError:
                    self.logger.warning("rasterio not available, using OpenCV for TIFF files")
                    image = cv2.imread(str(image_path))
                    if image is not None:
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image = cv2.imread(str(image_path))
                if image is None:
                    self.logger.error(f"Could not load image: {image_path}")
                    return None
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            return image
            
        except Exception as e:
            self.logger.error(f"Error loading image {image_path}: {e}")
            return None
    
    def create_labeling_interface(self, image_path: Path, mode: str) -> List[Dict]:
        """
        Create interactive labeling interface
        
        Args:
            image_path: Path to image to label
            mode: Labeling mode ('tomato_detection' or 'disease_detection')
            
        Returns:
            List of bounding box annotations
        """
        # Load image
        image = self.load_image_for_labeling(image_path)
        if image is None:
            return []
        
        height, width = image.shape[:2]
        mode_config = self.labeling_modes[mode]
        
        class BBoxSelector:
            def __init__(self, image, mode_config):
                self.image = image
                self.mode_config = mode_config
                self.annotations = []
                self.current_bbox = None
                self.start_point = None
                self.current_class = mode_config['classes'][0]  # Default to first class
                self.current_class_index = 0
                
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
                            
                            # Get color for current class
                            color = self.mode_config['colors'][self.current_class_index]
                            
                            # Draw the bbox
                            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, 
                                                   linewidth=2, edgecolor=color, facecolor='none')
                            event.inaxes.add_patch(rect)
                            
                            # Add class label
                            event.inaxes.text(x1, y1-5, self.current_class, 
                                            color=color, fontsize=8, weight='bold')
                            
                            plt.draw()
                            
                            print(f"Added {self.current_class} annotation: {len(self.annotations)} total")
                        
                        self.start_point = None
                        
            def onkey(self, event):
                # Handle key presses for class switching
                if event.key in self.mode_config['keys']:
                    key_index = self.mode_config['keys'].index(event.key)
                    if key_index < len(self.mode_config['classes']):
                        self.current_class = self.mode_config['classes'][key_index]
                        self.current_class_index = key_index
                        color = self.mode_config['colors'][key_index]
                        print(f"Switched to {self.current_class.upper()} mode ({color})")
                elif event.key == 'u':  # Undo last annotation
                    if self.annotations:
                        removed = self.annotations.pop()
                        print(f"Undid last annotation ({removed['class']})")
                        # Clear and redraw all annotations
                        event.inaxes.clear()
                        event.inaxes.imshow(self.image)
                        self.redraw_annotations(event.inaxes)
                        plt.draw()
            
            def redraw_annotations(self, ax):
                """Redraw all existing annotations"""
                for ann in self.annotations:
                    class_idx = self.mode_config['classes'].index(ann['class'])
                    color = self.mode_config['colors'][class_idx]
                    bbox = ann['bbox']
                    
                    rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2]-bbox[0], bbox[3]-bbox[1], 
                                           linewidth=2, edgecolor=color, facecolor='none')
                    ax.add_patch(rect)
                    ax.text(bbox[0], bbox[1]-5, ann['class'], 
                           color=color, fontsize=8, weight='bold')
        
        # Create interactive plot
        fig, ax = plt.subplots(figsize=(15, 10))
        ax.imshow(image)
        
        # Create title with instructions
        title_lines = [
            f"Labeling: {image_path.name} | Mode: {mode}",
            "Left click to draw bbox | 'u' to undo | Close window when done"
        ]
        
        # Add key instructions
        key_instructions = []
        for i, (key, class_name) in enumerate(zip(mode_config['keys'], mode_config['classes'])):
            if i < len(mode_config['colors']):
                key_instructions.append(f"'{key}': {class_name}")
        
        title_lines.append(" | ".join(key_instructions[:6]))  # First 6 instructions
        if len(key_instructions) > 6:
            title_lines.append(" | ".join(key_instructions[6:]))  # Remaining instructions
        
        ax.set_title("\n".join(title_lines), fontsize=10)
        
        selector = BBoxSelector(image, mode_config)
        fig.canvas.mpl_connect('button_press_event', selector.onclick)
        fig.canvas.mpl_connect('key_press_event', selector.onkey)
        
        # Print instructions to console
        print("\n" + "="*80)
        print(f"LABELING: {image_path.name} | MODE: {mode.upper()}")
        print("="*80)
        print("CONTROLS:")
        print("- Left click to start bbox, left click again to finish")
        print("- Press 'u' to undo last annotation")
        print("- Close window when done")
        print("\nCLASS KEYS:")
        for key, class_name, color in zip(mode_config['keys'], mode_config['classes'], mode_config['colors']):
            print(f"- Press '{key}' for {class_name.upper()} ({color})")
        print("="*80)
        
        plt.show()
        
        return selector.annotations
    
    def save_annotations(self, image_path: Path, annotations: List[Dict], mode: str) -> bool:
        """Save annotations to JSON file"""
        if not annotations:
            return False
        
        # Create filename based on image and mode
        base_name = image_path.stem
        if mode == 'disease_detection':
            # For disease detection, we might want to specify which disease
            json_filename = f"{base_name}_disease.json"
        else:
            json_filename = f"{base_name}.json"
        
        json_path = self.output_dir / json_filename
        
        # Create annotation data structure
        annotation_data = {
            'image_path': str(image_path),
            'image_name': image_path.name,
            'task': mode,
            'annotations': annotations,
            'labeled_date': datetime.now().isoformat(),
            'labeling_tool': 'standalone_labeling_tool',
            'mode': mode,
            'total_annotations': len(annotations)
        }
        
        try:
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(annotation_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Saved {len(annotations)} annotations: {json_filename}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving annotations to {json_path}: {e}")
            return False
    
    def label_images(self, mode: str = None) -> Dict:
        """
        Main labeling function
        
        Args:
            mode: Labeling mode ('tomato_detection' or 'disease_detection')
            
        Returns:
            Labeling session statistics
        """
        if mode:
            self.current_mode = mode
        
        self.logger.info(f"Starting labeling session in {self.current_mode} mode")
        
        # Find images to label
        image_files = self.find_images(self.input_dir)
        
        if not image_files:
            self.logger.error(f"No images found in {self.input_dir}")
            print(f"No images found in {self.input_dir}")
            print("Please add images to label and run again")
            return {'error': 'No images found'}
        
        self.logger.info(f"Found {len(image_files)} images to label")
        
        # Initialize statistics
        stats = {
            'total_images': len(image_files),
            'labeled_images': 0,
            'skipped_images': 0,
            'total_annotations': 0,
            'class_counts': {},
            'labeling_mode': self.current_mode,
            'session_start': self.start_time.isoformat(),
            'labeled_files': []
        }
        
        # Process each image
        for i, image_path in enumerate(image_files, 1):
            print(f"\n--- Labeling image {i}/{len(image_files)}: {image_path.name} ---")
            
            # Check if already labeled
            expected_json = self.output_dir / f"{image_path.stem}.json"
            if self.current_mode == 'disease_detection':
                expected_json = self.output_dir / f"{image_path.stem}_disease.json"
            
            if expected_json.exists():
                response = input(f"Image already labeled. Re-label? (y/n/s to skip): ").strip().lower()
                if response == 's':
                    stats['skipped_images'] += 1
                    continue
                elif response != 'y':
                    stats['skipped_images'] += 1
                    continue
            
            # Label the image
            try:
                annotations = self.create_labeling_interface(image_path, self.current_mode)
                
                if annotations:
                    # Save annotations
                    success = self.save_annotations(image_path, annotations, self.current_mode)
                    
                    if success:
                        stats['labeled_images'] += 1
                        stats['total_annotations'] += len(annotations)
                        stats['labeled_files'].append(image_path.name)
                        
                        # Count classes
                        for annotation in annotations:
                            class_name = annotation['class']
                            stats['class_counts'][class_name] = stats['class_counts'].get(class_name, 0) + 1
                        
                        print(f"Successfully labeled {image_path.name} with {len(annotations)} annotations")
                    else:
                        self.logger.error(f"Failed to save annotations for {image_path.name}")
                else:
                    print(f"No annotations created for {image_path.name}")
                    stats['skipped_images'] += 1
                    
            except Exception as e:
                self.logger.error(f"Error labeling {image_path.name}: {e}")
                stats['skipped_images'] += 1
                continue
            
            # Ask if user wants to continue
            if i < len(image_files):
                continue_response = input("Continue to next image? (y/n): ").strip().lower()
                if continue_response == 'n':
                    break
        
        # Calculate session time
        session_time = (datetime.now() - self.start_time).total_seconds()
        stats['session_end'] = datetime.now().isoformat()
        stats['session_time'] = f"{session_time:.2f}s"
        
        # Save session summary
        summary_path = self.output_dir / f"labeling_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        # Log summary
        self.logger.info("="*60)
        self.logger.info("LABELING SESSION SUMMARY")
        self.logger.info("="*60)
        self.logger.info(f"Images labeled: {stats['labeled_images']}/{stats['total_images']}")
        self.logger.info(f"Total annotations: {stats['total_annotations']}")
        
        if stats['class_counts']:
            class_summary = [f"{name}({count})" for name, count in stats['class_counts'].items()]
            self.logger.info(f"Classes: {', '.join(class_summary)}")
        
        self.logger.info(f"Session time: {session_time:.2f}s")
        self.logger.info(f"Annotations saved to: {self.output_dir}")
        self.logger.info("="*60)

        end_time = datetime.now()
        total_time = (end_time - self.start_time).total_seconds()
        self.logger.info(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info(f"Total session time: {total_time:.2f}s")

        
        return stats

def main():
    """Main function with command-line interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Standalone Image Labeling Tool")
    parser.add_argument("--input-dir", default="images_to_label", 
                       help="Directory containing images to label")
    parser.add_argument("--output-dir", default="labeled_annotations", 
                       help="Directory for saving annotations")
    parser.add_argument("--mode", choices=['tomato_detection', 'disease_detection'], 
                       default='tomato_detection', help="Labeling mode")
    parser.add_argument("--list-modes", action="store_true", 
                       help="List available labeling modes")
    
    args = parser.parse_args()
    
    # Initialize labeling tool
    labeler = StandaloneLabelingTool(
        input_dir=args.input_dir,
        output_dir=args.output_dir
    )
    
    if args.list_modes:
        print("AVAILABLE LABELING MODES")
        print("="*50)
        for mode, config in labeler.labeling_modes.items():
            print(f"\nMode: {mode}")
            print(f"Description: {config['description']}")
            print(f"Classes: {', '.join(config['classes'])}")
            print(f"Keys: {', '.join([f'{k}:{c}' for k, c in zip(config['keys'], config['classes'])])}")
        return
    
    # Check if input directory exists
    if not labeler.input_dir.exists() or not any(labeler.find_images(labeler.input_dir)):
        print(f"No images found in {labeler.input_dir}")
        print(f"Please add images to {labeler.input_dir} and run again")
        print("\nSupported formats: .jpg, .jpeg, .png, .tif, .tiff, .bmp")
        return
    
    print("STANDALONE IMAGE LABELING TOOL")
    print("="*50)
    print(f"Mode: {args.mode}")
    print(f"Input: {args.input_dir}")
    print(f"Output: {args.output_dir}")
    print()
    
    # Start labeling session
    stats = labeler.label_images(args.mode)
    
    if 'error' in stats:
        print(f"Error: {stats['error']}")
        return
    
    print(f"\nLABELING COMPLETE!")
    print(f"Images labeled: {stats['labeled_images']}")
    print(f"Total annotations: {stats['total_annotations']}")
    
    if stats['class_counts']:
        print("\nClass distribution:")
        for class_name, count in stats['class_counts'].items():
            print(f"  {class_name}: {count}")
    
    print(f"\nAnnotations saved to: {args.output_dir}")
    
    # Instructions for next steps
    print("\n" + "="*50)
    print("NEXT STEPS:")
    print("="*50)
    if args.mode == 'tomato_detection':
        print("1. Copy annotations to training project:")
        print(f"   cp {args.output_dir}/*.json tomato_training_project/labeled_data/tomato_detection/")
        print("2. Copy images to training project:")
        print(f"   cp {args.input_dir}/* tomato_training_project/labeled_data/tomato_detection/")
        print("3. Train model:")
        print("   python tomato_training_pipeline.py --train-tomato")
    else:
        print("1. Copy annotations to training project:")
        print(f"   cp {args.output_dir}/*.json tomato_training_project/labeled_data/disease_detection/")
        print("2. Copy images to training project:")
        print(f"   cp {args.input_dir}/* tomato_training_project/labeled_data/disease_detection/")
        print("3. Train model:")
        print("   python tomato_training_pipeline.py --train-disease")
    
    labeler.log_session_end()

if __name__ == "__main__":
    main()