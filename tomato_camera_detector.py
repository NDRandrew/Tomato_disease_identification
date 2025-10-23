#!/usr/bin/env python3
"""
Tomato Disease Detection with Camera Interface
Supports real-time camera processing and headless mode
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
import time
import threading
import sys

warnings.filterwarnings('ignore')

class TomatoSegmentationDetector:
    """
    Two-stage segmentation system for tomatoes and their diseases
    Now with real-time camera support
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
        
        # Camera mode attributes
        self.camera_running = False
        self.current_frame = None
        self.annotated_frame = None
        self.last_inference_time = 0
        self.inference_interval = 1.0  # seconds
        self.frame_lock = threading.Lock()
        
        # Check if display is available
        self.has_display = self.check_display_available()
        
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
    
    def check_display_available(self) -> bool:
        """Check if display is available (for headless systems)"""
        try:
            if sys.platform == 'linux':
                # Check if DISPLAY environment variable is set
                if 'DISPLAY' not in os.environ or not os.environ['DISPLAY']:
                    return False
                # Try to create a window to verify display works
                try:
                    import tkinter
                    tkinter.Tk().destroy()
                    return True
                except:
                    return False
            return True  # Assume Windows has display
        except:
            return False
    
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
        self.logger.info(f"Display available: {self.has_display}")
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
    
    def detect_tomatoes(self, image: np.ndarray) -> List[Dict]:
        """Stage 1: Segment tomatoes with maturity classification"""
        if self.tomato_model is None:
            return []
        
        try:
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
                        mask = masks.xy[i]
                        
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
            
            return detections
            
        except Exception as e:
            self.logger.error(f"Error in tomato segmentation: {e}")
            return []
    
    def detect_diseases(self, image: np.ndarray, tomato_detections: List[Dict]) -> List[Dict]:
        """Stage 2: Detect diseases in segmented tomatoes/plants"""
        if self.disease_model is None or not tomato_detections:
            return tomato_detections
        
        try:
            enhanced_detections = []
            
            if self.device == 'cuda':
                torch.cuda.empty_cache()
            
            for detection in tomato_detections:
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
                        enhanced_detections.append(detection)
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
                    self.logger.error(f"Error analyzing detection: {e}")
                    enhanced_detections.append(detection)
            
            return enhanced_detections
            
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
    
    def visualize_detections(self, image: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """Visualize segmentation detections with masks - only showing diseases"""
        try:
            annotated = image.copy()
            
            for detection in detections:
                health_status = detection.get('health_status', 'unknown')
                primary_disease = detection.get('primary_disease')
                mask = detection.get('mask', [])
                
                # Only visualize if diseased
                if health_status != 'diseased' or not primary_disease:
                    continue
                
                # Red color for diseased detections
                base_color = (0, 0, 255)  # Red (BGR format)
                
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
                
                # Only show disease name (no tomato class)
                bbox = detection['bbox']
                disease_conf = detection.get('diseases', [{}])[0].get('confidence', 0) if detection.get('diseases') else 0
                label = f"{primary_disease} ({disease_conf:.2f})"
                
                # Draw label
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                cv2.rectangle(annotated,
                             (int(bbox[0]), int(bbox[1]) - label_size[1] - 10),
                             (int(bbox[0]) + label_size[0], int(bbox[1])),
                             base_color, -1)
                cv2.putText(annotated, label,
                           (int(bbox[0]), int(bbox[1]) - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            return annotated
            
        except Exception as e:
            self.logger.error(f"Error creating visualization: {e}")
            return image
    
    def run_camera_mode(self, camera_id: int = 0, save_frames: bool = False, output_dir: str = "camera_output"):
        """
        Run real-time camera detection
        Args:
            camera_id: Camera device ID (default 0)
            save_frames: Save annotated frames periodically
            output_dir: Directory to save frames (if save_frames=True)
        """
        self.logger.info("="*60)
        self.logger.info("STARTING CAMERA MODE")
        self.logger.info("="*60)
        self.logger.info(f"Camera ID: {camera_id}")
        self.logger.info(f"Inference interval: {self.inference_interval}s")
        self.logger.info(f"Display mode: {'GUI' if self.has_display else 'Headless'}")
        
        if save_frames:
            os.makedirs(output_dir, exist_ok=True)
            self.logger.info(f"Saving frames to: {output_dir}")
        
        # Open camera
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            self.logger.error(f"Failed to open camera {camera_id}")
            print(f"ERROR: Could not open camera {camera_id}")
            print("Please check:")
            print("  - Camera is connected")
            print("  - Camera permissions are granted")
            print("  - No other application is using the camera")
            return
        
        # Get camera properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        self.logger.info(f"Camera opened: {width}x{height} @ {fps}fps")
        print(f"\nCamera opened: {width}x{height} @ {fps}fps")
        print(f"Running inference every {self.inference_interval} seconds")
        
        if self.has_display:
            print("\nControls:")
            print("  - Press 'q' or ESC to quit")
            print("  - Press 's' to save current frame")
            cv2.namedWindow('Tomato Detection', cv2.WINDOW_NORMAL)
        else:
            print("\nHeadless mode - no display available")
            print("  - Press Ctrl+C to quit")
            print("  - Frames will be saved to", output_dir if save_frames else "nowhere")
        
        self.camera_running = True
        self.has_disease_in_frame = False
        last_save_time = time.time()
        frame_count = 0
        inference_count = 0
        
        try:
            while self.camera_running:
                ret, frame = cap.read()
                
                if not ret:
                    self.logger.warning("Failed to read frame from camera")
                    break
                
                frame_count += 1
                current_time = time.time()
                
                # Run inference every N seconds
                if current_time - self.last_inference_time >= self.inference_interval:
                    self.last_inference_time = current_time
                    inference_count += 1
                    
                    self.logger.info(f"Running inference #{inference_count}...")
                    
                    # Convert BGR to RGB for model
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Stage 1: Detect tomatoes
                    tomato_detections = self.detect_tomatoes(frame_rgb)
                    
                    # Stage 2: Detect diseases
                    enhanced_detections = self.detect_diseases(frame_rgb, tomato_detections)
                    
                    # Check if there are diseased detections
                    diseased_detections = [d for d in enhanced_detections if d.get('health_status') == 'diseased']
                    has_disease = len(diseased_detections) > 0
                    
                    # Visualize (returns BGR for display)
                    with self.frame_lock:
                        self.annotated_frame = self.visualize_detections(frame, enhanced_detections)
                        self.has_disease_in_frame = has_disease
                    
                    # Log results
                    if enhanced_detections:
                        mature = sum(1 for d in enhanced_detections if d['class'] == 'mature_tomato')
                        immature = sum(1 for d in enhanced_detections if d['class'] == 'immature_tomato')
                        plants = sum(1 for d in enhanced_detections if d['class'] == 'tomato_plant')
                        diseased = len(diseased_detections)
                        
                        self.logger.info(f"Detected: {len(enhanced_detections)} objects "
                                       f"(Mature: {mature}, Immature: {immature}, Plants: {plants}, Diseased: {diseased})")
                        
                        diseases = [d.get('primary_disease') for d in enhanced_detections if d.get('primary_disease')]
                        if diseases:
                            self.logger.info(f"Diseases: {', '.join(set(diseases))}")
                    else:
                        self.logger.info("No detections")
                
                # Display frame
                if self.has_display:
                    with self.frame_lock:
                        display_frame = self.annotated_frame if self.annotated_frame is not None else frame
                    
                    # Add info overlay
                    info_text = f"FPS: {fps} | Frame: {frame_count} | Inferences: {inference_count}"
                    cv2.putText(display_frame, info_text, (10, 30), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    next_inference = int(self.inference_interval - (current_time - self.last_inference_time))
                    if next_inference > 0:
                        timer_text = f"Next inference in: {next_inference}s"
                        cv2.putText(display_frame, timer_text, (10, 60), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    
                    cv2.imshow('Tomato Detection', display_frame)
                    
                    # Handle keyboard input
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q') or key == 27:  # q or ESC
                        print("\nQuitting...")
                        break
                    elif key == ord('s'):  # Save frame (only if diseased)
                        with self.frame_lock:
                            if hasattr(self, 'has_disease_in_frame') and self.has_disease_in_frame:
                                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                save_path = os.path.join(output_dir, f"frame_{timestamp}.jpg")
                                os.makedirs(output_dir, exist_ok=True)
                                cv2.imwrite(save_path, display_frame)
                                print(f"Saved frame to: {save_path}")
                                self.logger.info(f"Manual save: {save_path}")
                            else:
                                print("No diseased detections in current frame - not saving")
                
                # Auto-save frames only when diseases are detected
                if save_frames and current_time - last_save_time >= self.inference_interval:
                    last_save_time = current_time
                    
                    with self.frame_lock:
                        # Only save if there are diseased detections
                        if hasattr(self, 'has_disease_in_frame') and self.has_disease_in_frame:
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            save_path = os.path.join(output_dir, f"auto_{timestamp}.jpg")
                            save_frame = self.annotated_frame if self.annotated_frame is not None else frame
                            cv2.imwrite(save_path, save_frame)
                            
                            if not self.has_display:
                                print(f"Saved diseased frame: {save_path}")
                            self.logger.info(f"Auto-saved diseased detection: {save_path}")
        
        except KeyboardInterrupt:
            print("\n\nInterrupted by user")
            self.logger.info("Camera mode interrupted by user")
        
        finally:
            self.camera_running = False
            cap.release()
            if self.has_display:
                cv2.destroyAllWindows()
            
            self.logger.info("="*60)
            self.logger.info("CAMERA MODE ENDED")
            self.logger.info(f"Total frames: {frame_count}")
            self.logger.info(f"Total inferences: {inference_count}")
            self.logger.info("="*60)
            
            print(f"\nCamera session ended")
            print(f"Total frames processed: {frame_count}")
            print(f"Total inferences: {inference_count}")

def main():
    """Main function with camera mode"""
    
    LOG_DIR = "logs"
    
    # Model paths
    TOMATO_MODEL_PATH = "tomato_training_project/models/tomato_segmentation_best.pt"
    DISEASE_MODEL_PATH = "tomato_training_project/models/disease_detection_best.pt"
    
    # Check models
    tomato_model_exists = os.path.exists(TOMATO_MODEL_PATH) if TOMATO_MODEL_PATH else False
    disease_model_exists = os.path.exists(DISEASE_MODEL_PATH) if DISEASE_MODEL_PATH else False
    
    if not tomato_model_exists:
        print("WARNING: Tomato segmentation model not found!")
        print(f"Expected: {TOMATO_MODEL_PATH}")
        TOMATO_MODEL_PATH = None
    
    if not disease_model_exists:
        print("WARNING: Disease model not found!")
        print(f"Expected: {DISEASE_MODEL_PATH}")
        DISEASE_MODEL_PATH = None
    
    if not tomato_model_exists and not disease_model_exists:
        print("ERROR: No models available. Please train models first.")
        return
    
    print("="*60)
    print("TOMATO DISEASE DETECTION - CAMERA MODE")
    print("="*60)
    
    # Initialize detector
    detector = TomatoSegmentationDetector(
        tomato_model_path=TOMATO_MODEL_PATH,
        disease_model_path=DISEASE_MODEL_PATH,
        confidence_threshold=0.5,
        disease_confidence_threshold=0.6,
        log_dir=LOG_DIR
    )
    
    # Run camera mode
    detector.run_camera_mode(
        camera_id=0,  # Use camera 0 (default webcam)
        save_frames=True,  # Save frames every 3 seconds
        output_dir="camera_output"
    )
    
    detector.log_session_end()
    print("\nGoodbye!")

if __name__ == "__main__":
    main()