# 🌿 Plant Disease Intelligent Expert System

> **CSC-412 Artificial Intelligence | Assignment 03 — Complex Computing Problem (CCP)**  
> Bahria University, Karachi Campus | BSCS-5A | Spring 2026  
> Instructor: Dr. Muhammad Tariq Siddique

---

## 📌 Overview

An intelligent expert system that diagnoses plant leaf diseases by combining **deep learning** (MobileNetV2 CNN) with **symbolic AI** (rule-based expert system). The system is fully interactive — users provide a leaf image path through a chatbot interface and receive a detailed diagnosis report.

### Key Features
- 🧠 **MobileNetV2 CNN** trained on real PlantVillage images with two-phase transfer learning
- 📚 **Frame-based Knowledge Base** with 6 disease classes, symptom ontology, and production rules
- 🔗 **Inference Engine** implementing both forward and backward chaining
- ⚖️ **Hybrid Fusion** combining CNN softmax probabilities with rule confidence scores
- 🤖 **Interactive Chatbot** accepting image paths and natural language knowledge queries
- 📊 **SVM Baseline** for fair 3-approach performance comparison
- 📈 **9-panel Visualisation** of all performance metrics

---

## 🗂️ Repository Structure

```
PlantDiseaseAI/
│
├── main.py                          # Complete system implementation
├── requirements.txt                 # Python dependencies
├── README.md                        # This file
│
├── report/
│   └── Plant_Disease_AI_Report.docx # Full assignment report
│
├── docs/
│   ├── system_architecture.md       # Architecture explanation
│   └── dataset_setup.md             # How to download and organise the dataset
│
├── results/
│   └── plant_disease_analysis.png   # 9-panel performance visualisation
│
└── .gitignore                       # Files excluded from version control
```

---

## 🚀 Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/PlantDiseaseAI.git
cd PlantDiseaseAI
```

### 2. Create and activate virtual environment
```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Prepare the dataset
See [`docs/dataset_setup.md`](docs/dataset_setup.md) for full instructions.  
Short version: download 6 folders from [PlantVillage on Kaggle](https://www.kaggle.com/datasets/emmarex/plantdisease), rename them, and place ~60–80 images per class.

```
PlantDataset/
  Early_Blight/
  Late_Blight/
  Bacterial_Blight/
  Mosaic_Virus/
  Leaf_Spot/
  Healthy/
```

### 5. Train the model
```bash
python main.py --train "C:\PlantDataset"
```
Training takes approximately 10–15 minutes. The model is saved as `plant_disease_model.keras`.

### 6. Start the interactive chatbot
```bash
python main.py --chat
```

---

## 💬 Chatbot Usage

```
You: C:\images\tomato_leaf.jpg
Bot: ============================================================
     PLANT DISEASE DIAGNOSIS REPORT
     Image          : tomato_leaf.jpg
     CNN Prediction : Early Blight (91.3%)
     FINAL DIAGNOSIS: Early Blight
     Confidence     : 87.5%
     Pathogen       : Alternaria solani
     Severity       : Moderate
     Treatments:
       - Apply Mancozeb fungicide (2 g/L water)
       - Remove and destroy infected leaves immediately
     ============================================================

You: symptoms of early blight
Bot: Symptoms of Early Blight:
       - Brown Spots
       - Concentric Rings
       - Yellowing
       - Leaf Lesions

You: what causes late blight
Bot: Late Blight is caused by Phytophthora infestans (Type: Fungal)

You: list all diseases
You: help
You: quit
```

---

## 🦠 Supported Disease Classes

| Disease | Pathogen | Type | Severity |
|---|---|---|---|
| Early Blight | *Alternaria solani* | Fungal | Moderate |
| Late Blight | *Phytophthora infestans* | Fungal | High |
| Bacterial Blight | *Xanthomonas oryzae* | Bacterial | High |
| Mosaic Virus | Tobacco Mosaic Virus (TMV) | Viral | High |
| Leaf Spot | *Cercospora* species | Fungal | Moderate |
| Healthy | — | — | None |

---

## 📊 System Performance

| Metric | SVM | CNN | Rule-Based | Hybrid |
|---|---|---|---|---|
| Accuracy | ~75% | ~96% | ~84% | ~97% |
| Precision | ~73% | ~95% | ~83% | ~96% |
| Recall | ~74% | ~95% | ~83% | ~96% |
| F1 Score | ~73% | ~95% | ~82% | ~96% |

*Results from training on PlantVillage subset (~420 images, 6 classes, with augmentation)*

---

## 🏗️ System Architecture

```
[Leaf Image] → [ImagePreprocessor] → [CNNClassifier]
                      ↓                      ↓
               [HSV Features]        [Softmax Probs]
                      ↓                      ↓
               [SVMClassifier]    [InferenceEngine] ← [KnowledgeBase]
                                         ↓
                                 [HybridIntegrator]
                                         ↓
                                 [PlantDiseaseChatbot] → Diagnosis Report
```

---

## 🛠️ Tech Stack

| Library | Version | Purpose |
|---|---|---|
| TensorFlow / Keras | 2.x | CNN model, MobileNetV2 |
| OpenCV (cv2) | 4.x | Image preprocessing |
| scikit-learn | 1.x | SVM, metrics, train/test split |
| NumPy | 1.x | Array operations |
| Matplotlib | 3.x | Visualisation |
| Python | 3.9+ | Runtime |

---

## 📄 Report

The full assignment report is available in [`report/Plant_Disease_AI_Report.docx`](report/Plant_Disease_AI_Report.docx).  
It covers theoretical background, system architecture, implementation details, experimental results, and analysis.

---

## 📜 License

This project is submitted as academic coursework for CSC-412 at Bahria University.  
For educational use only.
