#!/usr/bin/env python3
"""
Setup Script for Tomato Disease Detection System
Installs dependencies and creates initial project structure
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def safe_print(text):
    """Print text safely, handling encoding issues"""
    try:
        print(text)
    except UnicodeEncodeError:
        # Remove emojis and try again
        safe_text = ''.join(char for char in text if ord(char) < 128)
        print(safe_text)

def print_header():
    """Print setup header"""
    try:
        print("🍅" * 20)
        print("🍅 TOMATO DISEASE DETECTION SYSTEM SETUP")
        print("🍅" * 20)
    except UnicodeEncodeError:
        print("=" * 50)
        print("TOMATO DISEASE DETECTION SYSTEM SETUP")
        print("=" * 50)
    print()

def check_python_version():
    """Check if Python version is compatible"""
    safe_print("🐍 Checking Python version...")
    
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        safe_print("❌ Python 3.8+ is required")
        safe_print(f"   Current version: {version.major}.{version.minor}.{version.micro}")
        safe_print("   Please upgrade Python and try again")
        sys.exit(1)
    
    safe_print(f"✅ Python {version.major}.{version.minor}.{version.micro} - Compatible")
    print()

def install_requirements():
    """Install required packages"""
    safe_print("📦 Installing required packages...")
    
    # Core requirements
    requirements = [
        "ultralytics>=8.0.0",
        "opencv-python>=4.5.0",
        "numpy>=1.21.0",
        "matplotlib>=3.3.0",
        "scipy>=1.7.0",
        "scikit-image>=0.18.0",
        "scikit-learn>=1.0.0",
        "rasterio>=1.2.0",
        "pyyaml>=5.4.0",
        "torch>=1.12.0",
        "torchvision>=0.13.0"
    ]
    
    # Try to install each package
    failed_packages = []
    
    for package in requirements:
        try:
            safe_print(f"   Installing {package}...")
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", package, "--upgrade"
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            safe_print(f"   ✅ {package}")
        except subprocess.CalledProcessError:
            safe_print(f"   ❌ Failed to install {package}")
            failed_packages.append(package)
    
    if failed_packages:
        safe_print(f"\n⚠️ Failed to install: {', '.join(failed_packages)}")
        safe_print("   Try installing manually with:")
        for package in failed_packages:
            safe_print(f"   pip install {package}")
        print()
    else:
        safe_print("✅ All packages installed successfully!")
        print()

def create_project_structure():
    """Create initial project structure"""
    safe_print("📁 Creating project structure...")
    
    # Define project structure
    directories = [
        "input_images",
        "tomato_detection_results", 
        "logs",
        "tomato_training_project/raw_images/tomato_detection/tomato",
        "tomato_training_project/raw_images/tomato_detection/tomato_plant",
        "tomato_training_project/raw_images/tomato_detection/not_tomato",
        "tomato_training_project/raw_images/disease_detection/healthy",
        "tomato_training_project/raw_images/disease_detection/bacterial_spot",
        "tomato_training_project/raw_images/disease_detection/early_blight",
        "tomato_training_project/raw_images/disease_detection/late_blight",
        "tomato_training_project/raw_images/disease_detection/leaf_mold",
        "tomato_training_project/raw_images/disease_detection/septoria_leaf_spot",
        "tomato_training_project/raw_images/disease_detection/spider_mites",
        "tomato_training_project/raw_images/disease_detection/target_spot",
        "tomato_training_project/raw_images/disease_detection/yellow_leaf_curl_virus",
        "tomato_training_project/raw_images/disease_detection/mosaic_virus",
        "tomato_training_project/raw_images/disease_detection/bacterial_canker",
        "test_images"
    ]
    
    # Create directories
    created_count = 0
    for directory in directories:
        dir_path = Path(directory)
        if not dir_path.exists():
            dir_path.mkdir(parents=True, exist_ok=True)
            created_count += 1
    
    safe_print(f"✅ Created {created_count} directories")
    print()

def create_example_files():
    """Create example configuration and documentation files"""
    safe_print("📄 Creating example files...")
    
    # Create example config file
    config_content = """# Tomato Disease Detection Configuration

# Detection Settings
TOMATO_CONFIDENCE_THRESHOLD: 0.5
DISEASE_CONFIDENCE_THRESHOLD: 0.6

# Model Paths
TOMATO_MODEL: "tomato_training_project/models/tomato_detection_best.pt"
DISEASE_MODEL: "tomato_training_project/models/disease_detection_best.pt"

# Input/Output Directories
INPUT_FOLDER: "input_images"
OUTPUT_FOLDER: "tomato_detection_results"
LOG_DIRECTORY: "logs"

# Training Settings
DEFAULT_EPOCHS: 100
DEFAULT_BATCH_SIZE: 16
DEFAULT_MODEL_SIZE: "s"  # n, s, m, l, x

# Supported Disease Classes
DISEASES:
  - healthy
  - bacterial_spot
  - early_blight
  - late_blight
  - leaf_mold
  - septoria_leaf_spot
  - spider_mites
  - target_spot
  - yellow_leaf_curl_virus
  - mosaic_virus
  - bacterial_canker
"""
    
    with open("config.yaml", "w", encoding='utf-8') as f:
        f.write(config_content)
    
    # Create quick start guide (without emojis for compatibility)
    quickstart_content = """# TOMATO DISEASE DETECTION - QUICK START

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
"""
    
    with open("QUICKSTART.md", "w", encoding='utf-8') as f:
        f.write(quickstart_content)
    
    safe_print("✅ Created configuration and documentation files")
    print()

def create_run_scripts():
    """Create convenient run scripts"""
    safe_print("📜 Creating run scripts...")
    
    # Detection script
    detection_script = """#!/bin/bash
# Quick detection script

echo "Running Tomato Disease Detection..."
echo "Processing images from input_images/"

python tomato_disease_detector.py

echo "Detection complete!"
echo "Results saved to tomato_detection_results/"
echo "Logs saved to logs/"
"""
    
    with open("run_detection.sh", "w", encoding='utf-8') as f:
        f.write(detection_script)
    
    # Make executable on Unix systems
    if platform.system() != "Windows":
        os.chmod("run_detection.sh", 0o755)
    
    # Training script
    training_script = """#!/bin/bash
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
"""
    
    with open("run_training.sh", "w", encoding='utf-8') as f:
        f.write(training_script)
    
    if platform.system() != "Windows":
        os.chmod("run_training.sh", 0o755)
    
    # Windows batch files
    if platform.system() == "Windows":
        with open("run_detection.bat", "w", encoding='utf-8') as f:
            f.write("@echo off\n")
            f.write("echo Running Tomato Disease Detection...\n")
            f.write("python tomato_disease_detector.py\n")
            f.write("pause\n")
        
        with open("run_training.bat", "w", encoding='utf-8') as f:
            f.write("@echo off\n")
            f.write("echo Tomato Training Pipeline\n")
            f.write("echo 1) Label tomato data\n")
            f.write("echo 2) Train tomato model\n") 
            f.write("echo 3) Train disease model\n")
            f.write("set /p choice=Enter choice (1-3): \n")
            f.write("if %choice%==1 python tomato_training_pipeline.py --label-tomato\n")
            f.write("if %choice%==2 python tomato_training_pipeline.py --train-tomato\n")
            f.write("if %choice%==3 python tomato_training_pipeline.py --train-disease --all-diseases\n")
            f.write("pause\n")
    
    safe_print("✅ Created run scripts")
    print()

def verify_installation():
    """Verify installation by importing key packages"""
    safe_print("🔍 Verifying installation...")
    
    test_imports = [
        ("cv2", "OpenCV"),
        ("numpy", "NumPy"), 
        ("matplotlib", "Matplotlib"),
        ("sklearn", "Scikit-learn"),
        ("ultralytics", "Ultralytics YOLO"),
        ("torch", "PyTorch")
    ]
    
    failed_imports = []
    
    for module, name in test_imports:
        try:
            __import__(module)
            safe_print(f"   ✅ {name}")
        except ImportError:
            safe_print(f"   ❌ {name} - Import failed")
            failed_imports.append(name)
    
    if failed_imports:
        safe_print(f"\n⚠️ Some packages failed to import: {', '.join(failed_imports)}")
        safe_print("   This may affect functionality. Try reinstalling these packages.")
    else:
        safe_print("✅ All packages imported successfully!")
    
    print()

def show_next_steps():
    """Show next steps to user"""
    safe_print("🎉 SETUP COMPLETE!")
    safe_print("=" * 40)
    print()
    safe_print("📋 NEXT STEPS:")
    print()
    safe_print("1. ADD YOUR IMAGES:")
    safe_print("   Tomato images → tomato_training_project/raw_images/tomato_detection/tomato/")
    safe_print("   Plant images → tomato_training_project/raw_images/tomato_detection/tomato_plant/")
    safe_print("   Non-tomato images → tomato_training_project/raw_images/tomato_detection/not_tomato/")
    safe_print("   Disease images → tomato_training_project/raw_images/disease_detection/[disease]/")
    print()
    safe_print("2️⃣ START TRAINING:")
    safe_print("   🏷️ Label data: python tomato_training_pipeline.py --label-tomato")
    safe_print("   🚀 Train model: python tomato_training_pipeline.py --train-tomato")
    print()
    safe_print("3️⃣ RUN DETECTION:")
    safe_print("   📷 Add images → input_images/")
    safe_print("   🔍 Run: python tomato_disease_detector.py")
    print()
    safe_print("📚 DOCUMENTATION:")
    safe_print("   • QUICKSTART.md - Quick start guide")
    safe_print("   • config.yaml - Configuration settings")
    safe_print("   • logs/ - Automatic logging directory")
    print()
    safe_print("🆘 NEED HELP?")
    safe_print("   Check the logs/ directory for detailed error messages and progress.")
    print()
    safe_print("🍅 Happy tomato disease detection! 🍅")

def main():
    """Main setup function"""
    print_header()
    
    try:
        # Run setup steps
        check_python_version()
        install_requirements()
        create_project_structure()
        create_example_files()
        create_run_scripts()
        verify_installation()
        show_next_steps()
        
    except KeyboardInterrupt:
        safe_print("\n\n❌ Setup interrupted by user")
        sys.exit(1)
    except UnicodeEncodeError as e:
        safe_print(f"\n❌ Character encoding error: {e}")
        safe_print("This is typically a Windows console encoding issue.")
        safe_print("Setup should still work. Check the created files.")
        sys.exit(0)
    except Exception as e:
        safe_print(f"\n❌ Setup failed with error: {e}")
        safe_print("Please check the error and try again")
        sys.exit(1)

if __name__ == "__main__":
    main()