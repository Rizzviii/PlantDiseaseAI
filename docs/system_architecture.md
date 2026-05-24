# System Architecture

## Overview

The system is organised into 9 modules implemented as Python classes and functions in `main.py`.

```
[Leaf Image Path]
      |
      v
[ImagePreprocessor]  ──────────────────────────────────────────────────────────
      |                                                                        |
      | float32 [0,1] array                                              HSV features
      v                                                                        |
[CNNClassifier]                                                         [SVMClassifier]
      |                                                                        |
      | softmax probabilities (N classes)                              class prediction
      v                                                                        |
[InferenceEngine] ←── [KnowledgeBase]                                         |
      |                                                                        |
      | rule conclusions + inference log                                       |
      v                                                                        |
[HybridIntegrator]                                                             |
      |                                                                        |
      | fused score per disease                                                |
      v                                                                        |
[PlantDiseaseChatbot] ←────────────────────────────────────────────────────────
      |
      v
  Diagnosis Report (terminal output)
```

---

## Module Descriptions

### `KnowledgeBase`
- Stores 6 disease frames (Early Blight, Late Blight, Bacterial Blight, Mosaic Virus, Leaf Spot, Healthy)
- Each frame contains: pathogen, type, symptom weights, affected crops, treatments, severity, spread rate
- Defines `SYMPTOM_ONTOLOGY` — 17 canonical symptom terms used across KB and rules
- Contains 6 production rules (IF-THEN with confidence weights)

### `ImagePreprocessor`
- `preprocess_array()` — OpenCV pipeline: resize → BGR→RGB → float32 → ImageNet standardisation
- `augment()` — 4 augmentations per image: flip, rotate 90°, brightness +30, brightness -30
- `extract_color_features()` — HSV analysis returning 6 colour statistics
- `load_real_dataset()` — loads images from folder structure, applies augmentation
- `extract_hsv_features_batch()` — batch HSV extraction for SVM input

### `CNNClassifier`
- `build()` — MobileNetV2 (alpha=0.35, ImageNet weights) + custom head
- `train()` — two-phase training: freeze base (Phase 1) → unfreeze top 30 layers (Phase 2)
- `predict()` — returns full softmax probability vector
- `evaluate()` — sklearn metrics on test set
- `save()` / `load()` — persist model to/from `.keras` file

### `SVMClassifier`
- RBF kernel SVM (C=10, gamma=scale) on 6-dimensional HSV features
- StandardScaler normalisation before training and inference
- Serves as traditional ML baseline for comparison

### `InferenceEngine`
- `set_facts()` — loads symptom observations into working memory
- `forward_chain()` — fires all rules whose conditions are satisfied; returns `{disease: confidence}`
- `backward_chain()` — recursively verifies a hypothesis; returns `(supported, matched_symptoms, confidence)`
- `get_log()` — returns human-readable inference trace

### `HybridIntegrator`
- `fuse()` — weighted combination: `α × CNN_prob + (1-α) × rule_conf` where α=0.6
- Uses full softmax distribution (not just top-1 class)
- Returns final disease, confidence, and per-disease score breakdown

### `PlantDiseaseChatbot`
- Handles image paths → runs full diagnosis pipeline
- Handles natural language queries via regex pattern matching
- Maintains session state (last diagnosed disease for follow-up questions)
- Commands: `diagnose <path>`, `symptoms of <disease>`, `treatment for <disease>`, `what causes <disease>`, `how serious is <disease>`, `list all diseases`, `help`, `quit`

### `visualize_all()`
- Generates 9-panel matplotlib figure saved as `plant_disease_analysis.png`
- Panels: grouped bar, CNN training history, loss curve, top-3 predictions, fusion breakdown, confusion matrix, radar chart, per-class F1, accuracy comparison

### `train_pipeline()` / `run_chat()`
- `train_pipeline()` — end-to-end training, evaluation, and model saving
- `run_chat()` — loads trained model and starts interactive terminal session

---

## Data Flow for Single Image Diagnosis

1. User types image path in chatbot
2. `ImagePreprocessor` loads and preprocesses image to float32 [0,1] array
3. `CNNClassifier.predict()` returns softmax probabilities for all 6 classes
4. `ImagePreprocessor.extract_color_features()` extracts HSV features
5. `SVMClassifier.predict()` returns class index
6. CNN top-1 class is used to look up symptoms in `KnowledgeBase`
7. `InferenceEngine.forward_chain()` fires matching rules
8. `InferenceEngine.backward_chain()` verifies CNN top-1 hypothesis
9. `HybridIntegrator.fuse()` combines CNN probs and rule confidences
10. Chatbot formats and prints the diagnosis report
