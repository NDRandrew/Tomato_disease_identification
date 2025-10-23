# 🍅 Tomato Disease Detection System
### Sistema de Detecção de Doenças em Tomateiros

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00FFFF.svg)](https://github.com/ultralytics/ultralytics)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
<img src="https://img.shields.io/badge/TCC-UFABC-green.svg" alt="TCC Badge"/>
<img src="https://img.shields.io/badge/Computer%20Vision-Deep%20Learning-orange.svg" alt="CV Badge"/>

---

## 📋 Sobre o Projeto | About the Project

**[PT-BR]** Este é um sistema de detecção de doenças em tomateiros desenvolvido como Trabalho de Conclusão de Curso (TCC) onde o Desenvolvimento foi feito por **André Carbonieri Silva**. O sistema utiliza técnicas avançadas de visão computacional e deep learning para identificar automaticamente doenças em plantas de tomate através de imagens, auxiliando produtores na detecção precoce de problemas fitossanitários.

**[EN]** This is a tomato disease detection system developed as a Thesis Completion Project (TCC) where the software development was made by **André Carbonieri Silva**. The system uses advanced computer vision and deep learning techniques to automatically identify diseases in tomato plants through images, helping farmers with early detection of phytosanitary problems.

### 🎯 Características Principais | Key Features

- **🔍 Detecção em Dois Estágios | Two-Stage Detection**
  - Estágio 1: Segmentação de tomates (maduros, imaturos, plantas)
  - Stage 1: Tomato segmentation (mature, immature, plants)
  - Estágio 2: Identificação de 11 classes de doenças
  - Stage 2: Identification of 11 disease classes

- **⚡ Processamento em Tempo Real | Real-Time Processing**
  - Suporte para webcam/câmera USB
  - Webcam/USB camera support
  - Inferência otimizada com GPU/CPU
  - GPU/CPU optimized inference

- **📊 Modo Batch | Batch Mode**
  - Processamento de múltiplas imagens
  - Multiple image processing
  - Relatórios detalhados e visualizações
  - Detailed reports and visualizations

- **🎓 Pipeline de Treinamento Incremental | Incremental Training Pipeline**
  - Treinamento progressivo de doenças
  - Progressive disease training
  - Otimização automática de batch size baseada em GPU
  - Automatic GPU-based batch size optimization

---

## 🦠 Classes de Doenças Detectadas | Detected Disease Classes

O sistema é capaz de identificar as seguintes condições:
The system can identify the following conditions:

| Classe | Nome em Português | English Name |
|--------|-------------------|--------------|
| `healthy` | Saudável | Healthy |
| `bacterial_spot` | Mancha Bacteriana | Bacterial Spot |
| `early_blight` | Pinta Preta | Early Blight |
| `late_blight` | Requeima | Late Blight |
| `leaf_mold` | Mofo das Folhas | Leaf Mold |
| `septoria_leaf_spot` | Mancha de Septória | Septoria Leaf Spot |
| `spider_mites` | Ácaros | Spider Mites |
| `target_spot` | Mancha de Alvo | Target Spot |
| `yellow_leaf_curl_virus` | Vírus da Folha Amarela | Yellow Leaf Curl Virus |
| `mosaic_virus` | Vírus do Mosaico | Mosaic Virus |
| `bacterial_canker` | Cancro Bacteriano | Bacterial Canker |

---

## 🏗️ Arquitetura do Sistema | System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    INPUT IMAGE / CAMERA                  │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│          STAGE 1: Tomato Segmentation (YOLOv8-seg)      │
│  ┌──────────────────────────────────────────────────┐  │
│  │  • Mature Tomato Detection                        │  │
│  │  • Immature Tomato Detection                      │  │
│  │  • Tomato Plant Detection                         │  │
│  │  • Non-Tomato Filtering                           │  │
│  └──────────────────────────────────────────────────┘  │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│       STAGE 2: Disease Detection (YOLOv8-detect)        │
│  ┌──────────────────────────────────────────────────┐  │
│  │  • Crop detected regions                          │  │
│  │  • Classify health status                         │  │
│  │  • Identify disease types                         │  │
│  │  • Confidence scoring                             │  │
│  └──────────────────────────────────────────────────┘  │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│          OUTPUT: Annotated Images + JSON Reports         │
│  • Bounding boxes and masks                             │
│  • Disease labels and confidence                        │
│  • Health status classification                         │
│  • Statistical summary                                  │
└─────────────────────────────────────────────────────────┘
```

---

## 🚀 Instalação | Installation

### Requisitos | Requirements

- Python 3.8+
- CUDA 11.8+ (opcional, para aceleração GPU | optional, for GPU acceleration)
- 8GB+ RAM (16GB+ recomendado | recommended)
- GPU NVIDIA com 6GB+ VRAM (opcional | optional)

### Instalação Rápida | Quick Install

```bash
# Clone o repositório | Clone the repository
git clone https://github.com/yourusername/tomato-disease-detection.git
cd tomato-disease-detection

# Crie um ambiente virtual | Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou | or
venv\Scripts\activate  # Windows

# Instale as dependências | Install dependencies
pip install -r requirements.txt

# Para suporte GPU (opcional) | For GPU support (optional)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### requirements.txt

```txt
ultralytics>=8.0.0
opencv-python>=4.8.0
numpy>=1.24.0
torch>=2.0.0
torchvision>=0.15.0
matplotlib>=3.7.0
pyyaml>=6.0
pillow>=10.0.0
```

---

## 📖 Como Usar | How to Use

### 1️⃣ Detecção em Tempo Real | Real-Time Detection

Execute o sistema com sua webcam:
Run the system with your webcam:

```bash
python tomato_camera_detection.py
```

**Controles | Controls:**
- `Q` ou `ESC`: Sair | Quit
- `S`: Salvar frame atual (apenas se doença detectada) | Save current frame (only if disease detected)

**Modo Headless (sem display):**
```bash
DISPLAY="" python tomato_camera_detection.py
```

### 2️⃣ Processamento em Lote | Batch Processing

Processe múltiplas imagens de uma vez:
Process multiple images at once:

```bash
# Adicione imagens em | Add images to: input_images/
python tomato_segmentation_detection.py
```

**Saída | Output:**
- `tomato_segmentation_results/`: Imagens anotadas | Annotated images
- `processing_summary.json`: Relatório detalhado | Detailed report
- `logs/`: Logs de processamento | Processing logs

### 3️⃣ Treinamento de Modelos | Model Training

#### Preparar Dataset

```bash
# Estrutura de diretórios | Directory structure
tomato_training_project/
├── raw_images/
│   ├── tomato_detection/
│   │   ├── tomato/          # Tomates maduros | Mature tomatoes
│   │   ├── tomato_plant/    # Plantas | Plants
│   │   └── not_tomato/      # Não-tomates | Non-tomatoes
│   └── disease_detection/
│       ├── healthy/
│       ├── bacterial_spot/
│       └── ...              # Outras doenças | Other diseases
```

#### Rotular Dados | Label Data

```bash
# Rotular detecção de tomates | Label tomato detection
python tomato_training_pipeline.py --label-tomato

# Rotular doenças | Label diseases
python tomato_training_pipeline.py --label-disease --disease bacterial_spot
```

#### Treinar Modelos | Train Models

```bash
# Treinar modelo de segmentação de tomates | Train tomato segmentation model
python tomato_training_pipeline.py --train-tomato --epochs 100

# Treinar modelo de detecção de doenças (incremental) | Train disease detection model (incremental)
python tomato_training_pipeline.py --train-disease --disease bacterial_spot --epochs 50
python tomato_training_pipeline.py --train-disease --disease early_blight --epochs 50
# ... adicione mais doenças gradualmente | ... add more diseases gradually
```

#### Verificar GPU | Check GPU

```bash
python tomato_training_pipeline.py --check-gpu
```

---

## 📊 Estrutura do Projeto | Project Structure

```
tomato-disease-detection/
├── tomato_camera_detection.py          # Detecção em tempo real | Real-time detection
├── tomato_segmentation_detection.py    # Processamento batch | Batch processing
├── tomato_training_pipeline.py         # Pipeline de treinamento | Training pipeline
├── requirements.txt                    # Dependências | Dependencies
├── README.md                           # Esta documentação | This documentation
│
├── tomato_training_project/            # Projeto de treinamento | Training project
│   ├── raw_images/                     # Imagens originais | Raw images
│   ├── labeled_data/                   # Dados rotulados | Labeled data
│   ├── datasets/                       # Datasets YOLO | YOLO datasets
│   ├── models/                         # Modelos treinados | Trained models
│   │   ├── tomato_segmentation_best.pt
│   │   └── disease_detection_best.pt
│   ├── training_results/               # Resultados | Results
│   └── disease_progress/               # Progresso incremental | Incremental progress
│
├── input_images/                       # Imagens para inferência | Images for inference
├── camera_output/                      # Saída da câmera | Camera output
├── tomato_segmentation_results/        # Resultados batch | Batch results
└── logs/                               # Logs do sistema | System logs
```

---

## 🔬 Metodologia | Methodology

### Abordagem de Dois Estágios | Two-Stage Approach

**Por que dois estágios? | Why two stages?**

1. **Precisão Melhorada | Improved Accuracy**: Focar primeiro em localizar tomates, depois analisar doenças
   Focus first on locating tomatoes, then analyze diseases

2. **Flexibilidade | Flexibility**: Treinar e melhorar cada estágio independentemente
   Train and improve each stage independently

3. **Eficiência | Efficiency**: Processar apenas regiões relevantes na segunda etapa
   Process only relevant regions in the second stage

### Treinamento Incremental | Incremental Training

O sistema suporta aprendizado incremental de doenças, permitindo:
The system supports incremental disease learning, allowing:

- Adicionar novas classes de doenças gradualmente
  Add new disease classes gradually
- Manter conhecimento de doenças previamente aprendidas
  Maintain knowledge of previously learned diseases
- Balancear dataset automaticamente
  Automatically balance dataset
- Evitar "catastrophic forgetting"
  Avoid catastrophic forgetting

---

## 📈 Resultados e Performance | Results and Performance

### Métricas de Avaliação | Evaluation Metrics

- **mAP50**: Mean Average Precision @ IoU 0.5
- **mAP50-95**: Mean Average Precision @ IoU 0.5-0.95
- **Precision**: Precisão das detecções
- **Recall**: Taxa de recuperação
- **F1-Score**: Média harmônica entre precisão e recall

### Tempo de Inferência | Inference Time

| Hardware | Tempo por Imagem | FPS (Real-Time) |
|----------|------------------|-----------------|
| RTX 4090 | ~20ms | ~50 FPS |
| RTX 3080 | ~30ms | ~33 FPS |
| RTX 3060 | ~45ms | ~22 FPS |
| CPU (i7) | ~300ms | ~3 FPS |

*Valores aproximados para imagem 640x640 | Approximate values for 640x640 image*

---

## 🎓 Contexto Acadêmico | Academic Context

Este projeto foi desenvolvido como Trabalho de Conclusão de Curso (TCC) com os seguintes objetivos:

This project was developed as a Thesis Completion Project (TCC) with the following objectives:

### Objetivos | Objectives

1. **Desenvolver sistema automatizado** para detecção precoce de doenças em tomateiros
   Develop an automated system for early detection of diseases in tomato plants

2. **Aplicar técnicas modernas** de deep learning e visão computacional
   Apply modern techniques of deep learning and computer vision

3. **Criar ferramenta prática** para auxiliar produtores rurais
   Create a practical tool to assist rural producers

4. **Validar abordagem** de detecção em dois estágios
   Validate two-stage detection approach

### Contribuições | Contributions

- 🔬 Pipeline completo de treinamento e inferência
  Complete training and inference pipeline
- 📊 Sistema de treinamento incremental de doenças
  Incremental disease training system
- ⚡ Otimização para diferentes tipos de hardware
  Optimization for different hardware types
- 📱 Suporte para detecção em tempo real
  Real-time detection support
- 🎯 Interface simplificada para uso prático
  Simplified interface for practical use

---

## 📝 Licença | License

Este projeto está sob a licença MIT. Veja o arquivo `LICENSE` para mais detalhes.
This project is under the MIT license. See the `LICENSE` file for more details.

---

## 👨‍💻 DEV Autor | DEV Author

**André Carbonieri Silva**
- TCC - Trabalho de Conclusão de Curso
- Thesis Completion Project

---

## 🙏 Agradecimentos | Acknowledgments

- **Ultralytics YOLOv8**: Framework de detecção de objetos
  Object detection framework
- **PyTorch**: Framework de deep learning
  Deep learning framework
- **OpenCV**: Biblioteca de visão computacional
  Computer vision library
- **PlantVillage Dataset**: Dataset público de doenças de plantas
  Public plant disease dataset

---

## 📚 Referências | References

1. Jocher, G., Chaurasia, A., & Qiu, J. (2023). Ultralytics YOLOv8. https://github.com/ultralytics/ultralytics

2. Hughes, D. P., & Salathé, M. (2015). An open access repository of images on plant health to enable the development of mobile disease diagnostics. arXiv preprint arXiv:1511.08060.

3. Redmon, J., Divvala, S., Girshick, R., & Farhadi, A. (2016). You only look once: Unified, real-time object detection. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 779-788).

---

## 📞 Contato | Contact

Para questões sobre este projeto:
For questions about this project:

- 📧 git: [@NDRandrew]

---

## 🗺️ Roadmap

### Implementado | Implemented ✅
- [x] Detecção de tomates em dois estágios
- [x] Classificação de maturidade
- [x] Detecção de 11 classes de doenças
- [x] Modo de câmera em tempo real
- [x] Processamento em lote
- [x] Pipeline de treinamento incremental
- [x] Suporte GPU/CPU
- [x] Logging e relatórios detalhados

### Planejado | Planned 🚀
- [ ] Interface web para upload de imagens
- [ ] Aplicativo móvel (iOS/Android)
- [ ] API REST para integração
- [ ] Dashboard de monitoramento de campo
- [ ] Recomendações de tratamento
- [ ] Suporte multilíngue expandido
- [ ] Integração com drones/IoT

---

<p align="center">
  <b>Desenvolvido com ❤️ para a agricultura sustentável</b><br>
  <b>Developed with ❤️ for sustainable agriculture</b>
</p>

<p align="center">
  <sub>Se este projeto foi útil, considere dar uma ⭐!</sub><br>
  <sub>If this project was helpful, consider giving it a ⭐!</sub>
</p>
