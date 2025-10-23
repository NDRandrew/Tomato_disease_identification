#!/usr/bin/env python3
"""
Tomato Disease Detection with Segmentation Support
Two-stage detection: 1) Tomato segmentation 2) Disease identification
Supports: mature_tomato, immature_tomato, tomato_plant, not_tomato
"""

import os
import cv2
import numpy as np
import json
import torch
from pathlib import Path
from datetime import datetime
import logging
from typing import List, Dict, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')

class TomatoSegmentationDetector:
    """
    Two-stage segmentation system for tomatoes and their diseases
    Stage 1: Segment tomatoes/tomato plants with maturity classification
    Stage 2: Identify diseases in detected tomatoes/plants
    """
    
    def __init__(self, 
                 tomato_model_path: Optional[str] = None,
                 disease_model_path: Optional[str] = None,
                 confidence_threshold: float = 0.5,
                 disease_confidence_threshold: float = 0.6,
                 log_dir: str = "logs"):
        """
        Initialize the segmentation-based detector
        """
        self.confidence_threshold = confidence_threshold
        self.disease_confidence_threshold = disease_confidence_threshold
        self.tomato_model = None
        self.disease_model = None
        
        # Setup device
        self.device = self.setup_device()
        
        # Setup logging
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.setup_logging()
        
        # Updated disease classes
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
        
        # Load models if provided
        self.load_models(tomato_model_path, disease_model_path)
    
    def setup_device(self):
        """Setup and detect the best available device for inference"""
        if torch.cuda.is_available():
            device = 'cuda'
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"Inference device: {device} ({gpu_name}, {gpu_memory:.1f} GB)")
        else:
            device = 'cpu'
            print(f"Inference device: {device}")
        return device
    
    def setup_logging(self):
        """Setup comprehensive logging system"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.log_dir / f"tomato_segmentation_{timestamp}.log"
        
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
        self.logger.info("TOMATO SEGMENTATION DETECTION SYSTEM STARTED")
        self.logger.info("="*60)
        self.logger.info(f"Log file: {log_file}")
        self.logger.info(f"Inference device: {self.device}")
        self.logger.info(f"Tomato confidence: {self.confidence_threshold}")
        self.logger.info(f"Disease confidence: {self.disease_confidence_threshold}")

    def log_session_end(self):
        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds()
        self.logger.info("="*60)
        self.logger.info(f"SESSION ENDED: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info(f"Total duration: {duration:.2f}s")
        self.logger.info("="*60)
            
    def load_models(self, tomato_model_path: Optional[str], disease_model_path: Optional[str]):
        """Load the segmentation models"""
        try:
            from ultralytics import YOLO
            
            # Load tomato segmentation model
            if tomato_model_path and os.path.exists(tomato_model_path):
                try:
                    self.tomato_model = YOLO(tomato_model_path)
                    if hasattr(self.tomato_model, 'to'):
                        self.tomato_model.to(self.device)
                    self.logger.info(f"Loaded tomato segmentation model: {tomato_model_path}")
                except Exception as e:
                    self.logger.error(f"Failed to load tomato model: {e}")
            else:
                self.logger.warning("No tomato segmentation model provided or file not found")
                
            # Load disease detection model  
            if disease_model_path and os.path.exists(disease_model_path):
                try:
                    self.disease_model = YOLO(disease_model_path)
                    if hasattr(self.disease_model, 'to'):
                        self.disease_model.to(self.device)
                    self.logger.info(f"Loaded disease model: {disease_model_path}")
                except Exception as e:
                    self.logger.error(f"Failed to load disease model: {e}")
            else:
                self.logger.warning("No disease model provided or file not found")
                
        except ImportError:
            self.logger.error("ultralytics not available. Install with: pip install ultralytics torch torchvision")
    
    def preprocess_image(self, image_path: str) -> Tuple[np.ndarray, Dict]:
        """Load and preprocess image for detection"""
        try:
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            metadata = {
                'height': image.shape[0],
                'width': image.shape[1],
                'channels': image.shape[2],
                'original_path': image_path
            }
            
            self.logger.info(f"Loaded image: {Path(image_path).name} ({image.shape[0]}x{image.shape[1]})")
            return image, metadata
            
        except Exception as e:
            self.logger.error(f"Error preprocessing image {image_path}: {e}")
            raise
    
    def detect_tomatoes(self, image: np.ndarray) -> List[Dict]:
        """
        Stage 1: Segment tomatoes with maturity classification
        """
        if self.tomato_model is None:
            self.logger.error("No tomato segmentation model loaded")
            return []
        
        try:
            self.logger.info("Stage 1: Segmenting tomatoes/plants")
            stage1_start = datetime.now()
            
            # Clear GPU cache
            if self.device == 'cuda':
                torch.cuda.empty_cache()
            
            # Run segmentation
            results = self.tomato_model(image, conf=self.confidence_threshold, device=self.device, verbose=False)
            
            detections = []
            
            for result in results:
                # Check if masks exist (segmentation output)
                if hasattr(result, 'masks') and result.masks is not None:
                    masks = result.masks
                    boxes = result.boxes
                    
                    for i in range(len(masks)):
                        confidence = float(boxes.conf[i])
                        bbox = boxes.xyxy[i].cpu().numpy()
                        class_id = int(boxes.cls[i])
                        
                        # Get mask polygon
                        mask = masks.xy[i]  # Polygon coordinates
                        
                        # Get class name
                        class_names = ['mature_tomato', 'immature_tomato', 'tomato_plant', 'not_tomato']
                        class_name = class_names[class_id] if class_id < len(class_names) else 'unknown'
                        
                        # Calculate mask area
                        if len(mask) > 0:
                            mask_area = cv2.contourArea(mask.astype(np.float32))
                        else:
                            mask_area = 0
                        
                        detection = {
                            'bbox': [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])],
                            'mask': mask.tolist() if len(mask) > 0 else [],
                            'confidence': confidence,
                            'class': class_name,
                            'class_id': class_id,
                            'area': float(mask_area),
                            'centroid': [float((bbox[0] + bbox[2]) / 2), float((bbox[1] + bbox[3]) / 2)]
                        }
                        detections.append(detection)
                
                # Fallback to bounding boxes if no masks
                elif result.boxes is not None:
                    boxes = result.boxes
                    self.logger.warning("Model returned boxes instead of masks - using detection mode")
                    
                    for i in range(len(boxes)):
                        confidence = float(boxes.conf[i])
                        bbox = boxes.xyxy[i].cpu().numpy()
                        class_id = int(boxes.cls[i])
                        
                        class_names = ['mature_tomato', 'immature_tomato', 'tomato_plant', 'not_tomato']
                        class_name = class_names[class_id] if class_id < len(class_names) else 'unknown'
                        
                        detection = {
                            'bbox': [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])],
                            'mask': [],
                            'confidence': confidence,
                            'class': class_name,
                            'class_id': class_id,
                            'area': float((bbox[2] - bbox[0]) * (bbox[3] - bbox[1])),
                            'centroid': [float((bbox[0] + bbox[2]) / 2), float((bbox[1] + bbox[3]) / 2)]
                        }
                        detections.append(detection)
            
            stage1_time = (datetime.now() - stage1_start).total_seconds()
            self.logger.info(f"Stage 1 complete: {len(detections)} detections in {stage1_time:.2f}s")
            
            # Log detection summary by maturity
            if detections:
                mature_count = sum(1 for d in detections if d['class'] == 'mature_tomato')
                immature_count = sum(1 for d in detections if d['class'] == 'immature_tomato')
                plant_count = sum(1 for d in detections if d['class'] == 'tomato_plant')
                not_tomato_count = sum(1 for d in detections if d['class'] == 'not_tomato')
                self.logger.info(f"Found: {mature_count} mature, {immature_count} immature, {plant_count} plants, {not_tomato_count} not-tomatoes")
            
            return detections
            
        except torch.cuda.OutOfMemoryError:
            self.logger.error("GPU out of memory during tomato segmentation!")
            return []
        except Exception as e:
            self.logger.error(f"Error in tomato segmentation: {e}")
            return []
    
    def detect_diseases(self, image: np.ndarray, tomato_detections: List[Dict]) -> List[Dict]:
        """
        Stage 2: Detect diseases in segmented tomatoes/plants
        """
        if self.disease_model is None:
            self.logger.warning("No disease model loaded - skipping disease detection")
            return tomato_detections
        
        if not tomato_detections:
            self.logger.info("No tomatoes detected - skipping disease detection")
            return []
        
        try:
            self.logger.info(f"Stage 2: Analyzing diseases in {len(tomato_detections)} detections...")
            stage2_start = datetime.now()
            
            enhanced_detections = []
            disease_summary = {}
            
            if self.device == 'cuda':
                torch.cuda.empty_cache()
            
            for i, detection in enumerate(tomato_detections):
                # Skip disease detection for 'not_tomato' class
                if detection.get('class') == 'not_tomato':
                    enhanced_detection = detection.copy()
                    enhanced_detection.update({
                        'diseases': [],
                        'primary_disease': None,
                        'health_status': 'not_applicable',
                        'disease_count': 0
                    })
                    enhanced_detections.append(enhanced_detection)
                    continue
                
                try:
                    # Extract ROI using bbox
                    bbox = detection['bbox']
                    x1, y1, x2, y2 = [int(coord) for coord in bbox]
                    
                    # Add padding
                    height, width = image.shape[:2]
                    padding = 0.1
                    pad_x = int((x2 - x1) * padding)
                    pad_y = int((y2 - y1) * padding)
                    
                    x1 = max(0, x1 - pad_x)
                    y1 = max(0, y1 - pad_y)
                    x2 = min(width, x2 + pad_x)
                    y2 = min(height, y2 + pad_y)
                    
                    roi = image[y1:y2, x1:x2]
                    
                    if roi.size == 0:
                        self.logger.warning(f"Empty ROI for detection {i+1}")
                        continue
                    
                    # Run disease detection
                    disease_results = self.disease_model(roi, conf=self.disease_confidence_threshold, 
                                                       device=self.device, verbose=False)
                    
                    diseases_found = []
                    max_confidence_disease = None
                    max_confidence = 0
                    
                    for disease_result in disease_results:
                        disease_boxes = disease_result.boxes
                        if disease_boxes is not None:
                            for j in range(len(disease_boxes)):
                                disease_conf = float(disease_boxes.conf[j])
                                disease_class_id = int(disease_boxes.cls[j])
                                
                                if disease_class_id < len(self.disease_classes):
                                    disease_name = self.disease_classes[disease_class_id]
                                    
                                    disease_info = {
                                        'disease': disease_name,
                                        'confidence': disease_conf,
                                        'severity': self.classify_severity(disease_conf)
                                    }
                                    diseases_found.append(disease_info)
                                    
                                    if disease_conf > max_confidence:
                                        max_confidence = disease_conf
                                        max_confidence_disease = disease_name
                                    
                                    if disease_name not in disease_summary:
                                        disease_summary[disease_name] = 0
                                    disease_summary[disease_name] += 1
                    
                    # Add disease information
                    enhanced_detection = detection.copy()
                    enhanced_detection.update({
                        'diseases': diseases_found,
                        'primary_disease': max_confidence_disease if max_confidence_disease != 'healthy' else None,
                        'health_status': 'healthy' if max_confidence_disease == 'healthy' or not diseases_found else 'diseased',
                        'disease_count': len([d for d in diseases_found if d['disease'] != 'healthy'])
                    })
                    
                    enhanced_detections.append(enhanced_detection)
                    
                except Exception as e:
                    self.logger.error(f"Error analyzing detection {i+1}: {e}")
                    enhanced_detections.append(detection)
            
            stage2_time = (datetime.now() - stage2_start).total_seconds()
            
            # Log disease summary
            if disease_summary:
                disease_list = [f"{disease}({count})" for disease, count in disease_summary.items() if disease != 'healthy']
                if disease_list:
                    self.logger.info(f"Diseases found: {', '.join(disease_list)}")
                healthy_count = disease_summary.get('healthy', 0)
                if healthy_count > 0:
                    self.logger.info(f"Healthy detections: {healthy_count}")
            
            self.logger.info(f"Stage 2 complete: {len(enhanced_detections)} analyzed in {stage2_time:.2f}s")
            return enhanced_detections
            
        except torch.cuda.OutOfMemoryError:
            self.logger.error("GPU out of memory during disease detection!")
            return tomato_detections
        except Exception as e:
            self.logger.error(f"Error in disease detection: {e}")
            return tomato_detections
    
    def classify_severity(self, confidence: float) -> str:
        """Classify disease severity based on confidence score"""
        if confidence >= 0.8:
            return "severe"
        elif confidence >= 0.6:
            return "moderate" 
        elif confidence >= 0.4:
            return "mild"
        else:
            return "uncertain"
    
    def visualize_detections(self, image: np.ndarray, detections: List[Dict], 
                           save_path: Optional[str] = None) -> np.ndarray:
        """
        Visualize segmentation detections with masks
        """
        try:
            annotated = image.copy()
            
            for i, detection in enumerate(detections):
                class_name = detection['class']
                confidence = detection['confidence']
                health_status = detection.get('health_status', 'unknown')
                primary_disease = detection.get('primary_disease')
                mask = detection.get('mask', [])
                
                # Choose colors
                if class_name == 'mature_tomato':
                    base_color = (255, 0, 0)  # Red
                elif class_name == 'immature_tomato':
                    base_color = (255, 165, 0)  # Orange
                elif class_name == 'tomato_plant':
                    base_color = (0, 255, 0)  # Green
                elif class_name == 'not_tomato':
                    base_color = (128, 128, 128)  # Gray
                else:
                    base_color = (255, 255, 0)  # Yellow
                
                # Modify color based on health
                if health_status == 'diseased':
                    base_color = (255, 0, 0)  # Override to red for diseased
                
                # Draw mask if available
                if mask and len(mask) > 0:
                    mask_array = np.array(mask, dtype=np.int32)
                    overlay = annotated.copy()
                    cv2.fillPoly(overlay, [mask_array], base_color)
                    annotated = cv2.addWeighted(annotated, 0.7, overlay, 0.3, 0)
                    cv2.polylines(annotated, [mask_array], True, base_color, 2)
                else:
                    # Draw bbox if no mask
                    bbox = detection['bbox']
                    cv2.rectangle(annotated, 
                                 (int(bbox[0]), int(bbox[1])), 
                                 (int(bbox[2]), int(bbox[3])), 
                                 base_color, 2)
                
                # Prepare label
                bbox = detection['bbox']
                label_parts = [f"{class_name} ({confidence:.2f})"]
                if primary_disease:
                    label_parts.append(f"{primary_disease}")
                label = " - ".join(label_parts)
                
                # Draw label
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                cv2.rectangle(annotated,
                             (int(bbox[0]), int(bbox[1]) - label_size[1] - 10),
                             (int(bbox[0]) + label_size[0], int(bbox[1])),
                             base_color, -1)
                cv2.putText(annotated, label,
                           (int(bbox[0]), int(bbox[1]) - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            if save_path:
                annotated_bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
                cv2.imwrite(save_path, annotated_bgr)
                self.logger.info(f"Saved visualization: {save_path}")
            
            return annotated
            
        except Exception as e:
            self.logger.error(f"Error creating visualization: {e}")
            return image
    
    def process_image(self, image_path: str, output_dir: str) -> Dict:
        """Process a single image through the complete pipeline"""
        try:
            image_start_time = datetime.now()
            self.logger.info(f"Processing: {Path(image_path).name}")
            
            os.makedirs(output_dir, exist_ok=True)
            
            # Load and preprocess
            image, metadata = self.preprocess_image(image_path)
            
            # Stage 1: Segment tomatoes
            tomato_detections = self.detect_tomatoes(image)
            
            # Stage 2: Detect diseases
            enhanced_detections = self.detect_diseases(image, tomato_detections)
            
            # Create output paths
            image_name = Path(image_path).stem
            vis_path = os.path.join(output_dir, f"{image_name}_results.jpg")
            json_path = os.path.join(output_dir, f"{image_name}_results.json")
            
            # Visualize
            annotated = self.visualize_detections(image, enhanced_detections, vis_path)
            
            # Compile results
            results = {
                'image_path': image_path,
                'metadata': metadata,
                'processing_timestamp': datetime.now().isoformat(),
                'detections': enhanced_detections,
                'summary': {
                    'total_detections': len(enhanced_detections),
                    'mature_tomato_count': sum(1 for d in enhanced_detections if d['class'] == 'mature_tomato'),
                    'immature_tomato_count': sum(1 for d in enhanced_detections if d['class'] == 'immature_tomato'),
                    'plant_count': sum(1 for d in enhanced_detections if d['class'] == 'tomato_plant'),
                    'not_tomato_count': sum(1 for d in enhanced_detections if d['class'] == 'not_tomato'),
                    'healthy_count': sum(1 for d in enhanced_detections if d.get('health_status') == 'healthy'),
                    'diseased_count': sum(1 for d in enhanced_detections if d.get('health_status') == 'diseased'),
                    'diseases_found': list(set([d.get('primary_disease') for d in enhanced_detections 
                                              if d.get('primary_disease') is not None]))
                }
            }
            
            # Save results
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, default=str)
            
            processing_time = (datetime.now() - image_start_time).total_seconds()
            self.logger.info(f"Completed {Path(image_path).name}: {results['summary']['total_detections']} detections in {processing_time:.2f}s")
            
            summary = results['summary']
            if summary['diseased_count'] > 0:
                diseases = ', '.join(summary['diseases_found'])
                self.logger.info(f"Found diseases: {diseases}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error processing image {image_path}: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return {'error': str(e), 'image_path': image_path}
    
    def process_folder(self, input_folder: str, output_folder: str) -> Dict:
        """Process all images in a folder"""
        try:
            self.logger.info(f"Processing folder: {input_folder}")
            
            os.makedirs(output_folder, exist_ok=True)
            
            # Find images
            extensions = ['.jpg', '.jpeg', '.png', '.tif', '.tiff']
            image_files = []
            for ext in extensions:
                image_files.extend(Path(input_folder).glob(f'**/*{ext}'))
                image_files.extend(Path(input_folder).glob(f'**/*{ext.upper()}'))
            
            self.logger.info(f"Found {len(image_files)} images to process")
            
            # Initialize summary
            summary = {
                'total_images': len(image_files),
                'processed_images': 0,
                'failed_images': 0,
                'total_detections': 0,
                'mature_tomatoes': 0,
                'immature_tomatoes': 0,
                'total_plants': 0,
                'total_not_tomatoes': 0,
                'healthy_detections': 0,
                'diseased_detections': 0,
                'diseases_summary': {},
                'processing_start': datetime.now().isoformat(),
                'results_per_image': [],
                'inference_device': self.device
            }
            
            # Process each image
            for image_path in image_files:
                try:
                    result = self.process_image(str(image_path), output_folder)
                    
                    if 'error' not in result:
                        summary['processed_images'] += 1
                        summary['total_detections'] += result['summary']['total_detections']
                        summary['mature_tomatoes'] += result['summary']['mature_tomato_count']
                        summary['immature_tomatoes'] += result['summary']['immature_tomato_count']
                        summary['total_plants'] += result['summary']['plant_count']
                        summary['total_not_tomatoes'] += result['summary'].get('not_tomato_count', 0)
                        summary['healthy_detections'] += result['summary']['healthy_count']
                        summary['diseased_detections'] += result['summary']['diseased_count']
                        
                        for disease in result['summary']['diseases_found']:
                            if disease not in summary['diseases_summary']:
                                summary['diseases_summary'][disease] = 0
                            summary['diseases_summary'][disease] += 1
                        
                        summary['results_per_image'].append(result)
                    else:
                        summary['failed_images'] += 1
                        
                except Exception as e:
                    self.logger.error(f"Failed to process {image_path}: {e}")
                    summary['failed_images'] += 1
            
            summary['processing_end'] = datetime.now().isoformat()
            
            # Save summary
            summary_path = os.path.join(output_folder, 'processing_summary.json')
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, default=str)
            
            self.logger.info("="*60)
            self.logger.info("PROCESSING SUMMARY")
            self.logger.info("="*60)
            self.logger.info(f"Images processed: {summary['processed_images']}/{summary['total_images']}")
            self.logger.info(f"Total detections: {summary['total_detections']}")
            self.logger.info(f"Mature tomatoes: {summary['mature_tomatoes']}")
            self.logger.info(f"Immature tomatoes: {summary['immature_tomatoes']}")
            self.logger.info(f"Plants: {summary['total_plants']}")
            self.logger.info(f"Healthy: {summary['healthy_detections']}, Diseased: {summary['diseased_detections']}")
            
            if summary['diseases_summary']:
                self.logger.info("Diseases found:")
                for disease, count in summary['diseases_summary'].items():
                    self.logger.info(f"  - {disease}: {count} instances")
            
            self.logger.info(f"Results saved to: {output_folder}")
            self.logger.info("="*60)
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error processing folder: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return {'error': str(e)}

def main():
    """Main function"""
    
    INPUT_FOLDER = "input_images"
    OUTPUT_FOLDER = "tomato_segmentation_results"
    LOG_DIR = "logs"
    
    # Model paths for segmentation
    TOMATO_MODEL_PATH = "tomato_training_project/models/tomato_segmentation_best.pt"
    DISEASE_MODEL_PATH = "tomato_training_project/models/disease_detection_best.pt"
    
    os.makedirs(INPUT_FOLDER, exist_ok=True)
    
    if not os.listdir(INPUT_FOLDER):
        print(f"No images found in {INPUT_FOLDER}.")
        print("Please add images to process and run again.")
        return
    
    tomato_model_exists = os.path.exists(TOMATO_MODEL_PATH) if TOMATO_MODEL_PATH else False
    disease_model_exists = os.path.exists(DISEASE_MODEL_PATH) if DISEASE_MODEL_PATH else False
    
    if not tomato_model_exists:
        print("Tomato segmentation model not found!")
        print(f"Expected: {TOMATO_MODEL_PATH}")
        print("Train with: python tomato_training_pipeline.py --train-tomato")
        TOMATO_MODEL_PATH = None
    
    if not disease_model_exists:
        print("Disease model not found!")
        print(f"Expected: {DISEASE_MODEL_PATH}")
        DISEASE_MODEL_PATH = None
    
    if not tomato_model_exists and not disease_model_exists:
        print("No models available. Please train models first.")
        return
    
    print("Initializing Tomato Segmentation Detector...")
    detector = TomatoSegmentationDetector(
        tomato_model_path=TOMATO_MODEL_PATH,
        disease_model_path=DISEASE_MODEL_PATH,
        confidence_threshold=0.5,
        disease_confidence_threshold=0.6,
        log_dir=LOG_DIR
    )
    
    print(f"Processing images from {INPUT_FOLDER}...")
    summary = detector.process_folder(INPUT_FOLDER, OUTPUT_FOLDER)
    
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    
    if 'error' in summary:
        print(f"Processing failed: {summary['error']}")
    else:
        print(f"Images processed: {summary['processed_images']}/{summary['total_images']}")
        print(f"Inference device: {summary['inference_device']}")
        print(f"Total detections: {summary['total_detections']}")
        print(f"Mature tomatoes: {summary['mature_tomatoes']}")
        print(f"Immature tomatoes: {summary['immature_tomatoes']}")
        print(f"Plants: {summary['total_plants']}")
        print(f"Healthy: {summary['healthy_detections']}")
        print(f"Diseased: {summary['diseased_detections']}")
        
        if summary['diseases_summary']:
            print("\nDiseases detected:")
            for disease, count in summary['diseases_summary'].items():
                print(f"   - {disease}: {count} instances")
        
        print(f"\nResults saved to: {OUTPUT_FOLDER}")
        print(f"Logs saved to: {LOG_DIR}")
    
    print("="*60)
    
    detector.log_session_end()

if __name__ == "__main__":
    main()