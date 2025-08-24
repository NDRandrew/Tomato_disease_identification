#!/usr/bin/env python3
"""
XML to JSON Annotation Converter
Converts Pascal VOC XML annotation files to JSON format for tomato detection training
"""

import os
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from datetime import datetime
import logging
from typing import Dict, List, Optional

class XMLToJSONConverter:
    """
    Converts XML annotation files (Pascal VOC format) to JSON format
    for tomato detection training pipeline
    """
    
    def __init__(self, 
                 xml_input_dir: str = "dataset_xml_annotations",
                 json_output_dir: str = "dataset_json_annotations",
                 log_dir: str = "logs"):
        """
        Initialize the converter
        
        Args:
            xml_input_dir: Directory containing XML annotation files
            json_output_dir: Directory for converted JSON files
            log_dir: Directory for log files
        """
        self.xml_input_dir = Path(xml_input_dir)
        self.json_output_dir = Path(json_output_dir)
        self.log_dir = Path(log_dir)
        
        # Create directories
        self.xml_input_dir.mkdir(exist_ok=True)
        self.json_output_dir.mkdir(exist_ok=True)
        self.log_dir.mkdir(exist_ok=True)
        
        # Setup logging
        self.setup_logging()
        
        # Class mapping - map XML class names to our system's class names
        self.class_mapping = {
            'green': 'tomato',           # Green tomatoes -> tomato
            'red': 'tomato',             # Red tomatoes -> tomato
            'tomato': 'tomato',          # Already correct
            'tomato_plant': 'tomato_plant',  # Already correct
            'plant': 'tomato_plant',     # Plant -> tomato_plant
            'not_tomato': 'not_tomato'   # Already correct
        }
        
        # Default class if mapping not found
        self.default_class = 'tomato'
        
        self.logger.info("XML to JSON Converter initialized")
        self.logger.info(f"XML input directory: {self.xml_input_dir}")
        self.logger.info(f"JSON output directory: {self.json_output_dir}")
    
    def setup_logging(self):
        """Setup logging system"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.log_dir / f"xml_conversion_{timestamp}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("="*60)
        self.logger.info("XML TO JSON CONVERTER STARTED")
        self.logger.info("="*60)
    
    def parse_xml_annotation(self, xml_path: Path) -> Optional[Dict]:
        """
        Parse a single XML annotation file
        
        Args:
            xml_path: Path to XML file
            
        Returns:
            Dictionary with parsed annotation data or None if error
        """
        try:
            # Parse XML
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            # Extract image information
            filename = root.find('filename')
            if filename is not None:
                image_name = filename.text
            else:
                # Use XML filename as image name (change extension)
                image_name = xml_path.stem + '.jpg'
                self.logger.warning(f"No filename found in {xml_path.name}, using {image_name}")
            
            # Extract image dimensions
            size = root.find('size')
            if size is not None:
                width = int(size.find('width').text)
                height = int(size.find('height').text)
            else:
                self.logger.error(f"No size information found in {xml_path.name}")
                return None
            
            # Extract all objects/annotations
            annotations = []
            objects = root.findall('object')
            
            for obj in objects:
                # Get class name
                name_elem = obj.find('name')
                if name_elem is None:
                    self.logger.warning(f"Object without name in {xml_path.name}")
                    continue
                
                xml_class = name_elem.text.lower().strip()
                
                # Map class name
                mapped_class = self.class_mapping.get(xml_class, self.default_class)
                
                # Get bounding box
                bndbox = obj.find('bndbox')
                if bndbox is None:
                    self.logger.warning(f"Object without bndbox in {xml_path.name}")
                    continue
                
                try:
                    xmin = float(bndbox.find('xmin').text)
                    ymin = float(bndbox.find('ymin').text)
                    xmax = float(bndbox.find('xmax').text)
                    ymax = float(bndbox.find('ymax').text)
                    
                    # Validate bounding box
                    if xmin >= xmax or ymin >= ymax:
                        self.logger.warning(f"Invalid bounding box in {xml_path.name}: {xmin},{ymin},{xmax},{ymax}")
                        continue
                    
                    # Check if bbox is within image bounds
                    if xmin < 0 or ymin < 0 or xmax > width or ymax > height:
                        self.logger.warning(f"Bounding box outside image bounds in {xml_path.name}")
                        # Clip to image bounds
                        xmin = max(0, xmin)
                        ymin = max(0, ymin)
                        xmax = min(width, xmax)
                        ymax = min(height, ymax)
                    
                    # Create annotation
                    annotation = {
                        'class': mapped_class,
                        'bbox': [xmin, ymin, xmax, ymax],
                        'width': width,
                        'height': height
                    }
                    
                    
                    annotations.append(annotation)
                    
                except (ValueError, AttributeError) as e:
                    self.logger.error(f"Error parsing bounding box in {xml_path.name}: {e}")
                    continue
            
            if not annotations:
                self.logger.warning(f"No valid annotations found in {xml_path.name}")
                return None
            
            # Create JSON structure matching our training pipeline format
            json_data = {
                'image_path': str(Path('tomato_training_project\\raw_images\\tomato_detection') / image_name),  
                'image_name': image_name,
                'task': 'tomato_detection',
                'annotations': annotations,
                'labeled_date': datetime.now().isoformat()
            }
            
            self.logger.debug(f"Parsed {xml_path.name}: {len(annotations)} annotations")
            return json_data
            
        except ET.ParseError as e:
            self.logger.error(f"XML parsing error in {xml_path.name}: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Unexpected error parsing {xml_path.name}: {e}")
            return None
    
    def convert_single_file(self, xml_path: Path) -> bool:
        """
        Convert a single XML file to JSON
        
        Args:
            xml_path: Path to XML file
            
        Returns:
            True if successful, False otherwise
        """
        # Parse XML
        json_data = self.parse_xml_annotation(xml_path)
        if json_data is None:
            return False
        
        # Create output JSON file path
        json_filename = xml_path.stem + '.json'
        json_path = self.json_output_dir / json_filename
        
        # Save JSON file
        try:
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Converted {xml_path.name} -> {json_filename} ({len(json_data['annotations'])} objects)")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving JSON file {json_path}: {e}")
            return False
    
    def convert_all_files(self) -> Dict:
        """
        Convert all XML files in the input directory
        
        Returns:
            Summary statistics
        """
        # Find all XML files
        xml_files = list(self.xml_input_dir.glob('*.xml'))
        xml_files.extend(self.xml_input_dir.glob('*.XML'))
        
        if not xml_files:
            self.logger.error(f"No XML files found in {self.xml_input_dir}")
            return {'error': 'No XML files found'}
        
        self.logger.info(f"Found {len(xml_files)} XML files to convert")
        
        # Initialize statistics
        stats = {
            'total_files': len(xml_files),
            'converted_files': 0,
            'failed_files': 0,
            'total_objects': 0,
            'class_counts': {},
            'failed_file_list': [],
            'conversion_start': datetime.now().isoformat()
        }
        
        # Convert each file
        for xml_path in xml_files:
            success = self.convert_single_file(xml_path)
            
            if success:
                stats['converted_files'] += 1
                
                # Count objects and classes (re-parse for stats)
                json_data = self.parse_xml_annotation(xml_path)
                if json_data:
                    stats['total_objects'] += len(json_data['annotations'])
                    
                    # Count classes
                    for annotation in json_data['annotations']:
                        class_name = annotation['class']
                        stats['class_counts'][class_name] = stats['class_counts'].get(class_name, 0) + 1
            else:
                stats['failed_files'] += 1
                stats['failed_file_list'].append(xml_path.name)
        
        stats['conversion_end'] = datetime.now().isoformat()
        
        # Save conversion summary
        summary_path = self.json_output_dir / 'conversion_summary.json'
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        # Log summary
        self.logger.info("="*60)
        self.logger.info("CONVERSION SUMMARY")
        self.logger.info("="*60)
        self.logger.info(f"Total XML files: {stats['total_files']}")
        self.logger.info(f"Successfully converted: {stats['converted_files']}")
        self.logger.info(f"Failed conversions: {stats['failed_files']}")
        self.logger.info(f"Total objects: {stats['total_objects']}")
        
        if stats['class_counts']:
            self.logger.info("Object classes found:")
            for class_name, count in stats['class_counts'].items():
                self.logger.info(f"  - {class_name}: {count} objects")
        
        if stats['failed_file_list']:
            self.logger.warning(f"Failed files: {', '.join(stats['failed_file_list'])}")
        
        self.logger.info(f"JSON files saved to: {self.json_output_dir}")
        self.logger.info(f"Summary saved to: {summary_path}")
        
        return stats
    
    def update_class_mapping(self, mapping: Dict[str, str]):
        """
        Update the class mapping dictionary
        
        Args:
            mapping: Dictionary mapping XML class names to system class names
        """
        self.class_mapping.update(mapping)
        self.logger.info(f"Updated class mapping: {self.class_mapping}")
    
    def preview_files(self, num_files: int = 5) -> List[Dict]:
        """
        Preview a few XML files to see what classes and structure they have
        
        Args:
            num_files: Number of files to preview
            
        Returns:
            List of preview data
        """
        xml_files = list(self.xml_input_dir.glob('*.xml'))[:num_files]
        
        previews = []
        
        for xml_path in xml_files:
            try:
                json_data = self.parse_xml_annotation(xml_path)
                if json_data:
                    preview = {
                        'file': xml_path.name,
                        'image': json_data['image_name'],
                        'objects': len(json_data['annotations']),
                        'classes': list(set(ann['class'] for ann in json_data['annotations'])),
                        'sample_bbox': json_data['annotations'][0]['bbox'] if json_data['annotations'] else None
                    }
                    previews.append(preview)
            except Exception as e:
                self.logger.error(f"Error previewing {xml_path.name}: {e}")
        
        return previews

def main():
    """Main function with command-line interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert XML annotations to JSON format")
    parser.add_argument("--xml-dir", default="dataset_xml_annotations", help="Directory with XML files")
    parser.add_argument("--json-dir", default="dataset_json_annotations", help="Output directory for JSON files")
    parser.add_argument("--preview", action="store_true", help="Preview XML files without converting")
    parser.add_argument("--preview-count", type=int, default=5, help="Number of files to preview")
    parser.add_argument("--class-mapping", help="JSON file with class mapping (optional)")
    
    args = parser.parse_args()
    
    # Initialize converter
    converter = XMLToJSONConverter(
        xml_input_dir=args.xml_dir,
        json_output_dir=args.json_dir
    )
    
    # Load custom class mapping if provided
    if args.class_mapping and os.path.exists(args.class_mapping):
        try:
            with open(args.class_mapping, 'r', encoding='utf-8') as f:
                custom_mapping = json.load(f)
            converter.update_class_mapping(custom_mapping)
        except Exception as e:
            print(f"Error loading class mapping: {e}")
    
    if args.preview:
        print("PREVIEWING XML FILES")
        print("="*50)
        
        # Check if XML directory exists and has files
        if not converter.xml_input_dir.exists():
            print(f"XML directory doesn't exist: {converter.xml_input_dir}")
            print("Please create it and add your XML files")
            return
        
        xml_files = list(converter.xml_input_dir.glob('*.xml'))
        if not xml_files:
            print(f"No XML files found in {converter.xml_input_dir}")
            print("Please add XML annotation files to this directory")
            return
        
        previews = converter.preview_files(args.preview_count)
        
        if not previews:
            print("No valid XML files could be previewed")
            return
        
        for i, preview in enumerate(previews, 1):
            print(f"\n{i}. File: {preview['file']}")
            print(f"   Image: {preview['image']}")
            print(f"   Objects: {preview['objects']}")
            print(f"   Classes: {', '.join(preview['classes'])}")
            if preview['sample_bbox']:
                print(f"   Sample bbox: {preview['sample_bbox']}")
        
        print(f"\nPreview complete. Found {len(xml_files)} XML files total.")
        print("\nTo convert all files, run:")
        print(f"python {__file__} --xml-dir {args.xml_dir} --json-dir {args.json_dir}")
        
    else:
        print("CONVERTING XML TO JSON")
        print("="*50)
        
        # Check if XML directory exists
        if not converter.xml_input_dir.exists():
            print(f"XML directory doesn't exist: {converter.xml_input_dir}")
            print("Creating directory...")
            converter.xml_input_dir.mkdir(exist_ok=True)
            print("Please add your XML files to this directory and run again")
            return
        
        # Convert all files
        stats = converter.convert_all_files()
        
        if 'error' in stats:
            print(f"Error: {stats['error']}")
            return
        
        print(f"\nCONVERSION COMPLETE!")
        print(f"Converted: {stats['converted_files']}/{stats['total_files']} files")
        print(f"Total objects: {stats['total_objects']}")
        
        if stats['class_counts']:
            print("\nClass distribution:")
            for class_name, count in stats['class_counts'].items():
                print(f"  {class_name}: {count}")
        
        print(f"\nJSON files saved to: {converter.json_output_dir}")
        
        # Instructions for next steps
        print("\n" + "="*50)
        print("NEXT STEPS:")
        print("="*50)
        print("1. Copy JSON files to your training project:")
        print(f"   cp {converter.json_output_dir}/*.json tomato_training_project/labeled_data/tomato_detection/")
        print("2. Copy corresponding images to:")
        print("   tomato_training_project/labeled_data/tomato_detection/")
        print("3. Train your model:")
        print("   python tomato_training_pipeline.py --train-tomato")

if __name__ == "__main__":
    main()