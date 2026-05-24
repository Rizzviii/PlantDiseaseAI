"""
================================================================
  Intelligent Expert System for Plant Disease Recognition
  CSC-412 Artificial Intelligence | BSCS-5A | Spring 2026
  Assignment 03 — Complex Computing Problem (CCP)
================================================================

THEORETICAL BACKGROUND
──────────────────────
1. PLANT DISEASE CHARACTERISTICS & VISUAL SYMPTOMS
   Plant diseases manifest through distinct visual patterns on
   leaves: color changes (yellowing, browning), structural
   deformations (curling, wilting), surface deposits (white
   powder in mildew), and lesion patterns (concentric rings in
   early blight, water-soaked spots in late blight). Early
   detection is critical — diseases like Late Blight (caused by
   Phytophthora infestans) can destroy an entire crop in days.

2. COMPUTER VISION TECHNIQUES
   - Preprocessing: Resize → BGR→RGB → float32 normalization
     → ImageNet standardization. Ensures consistent input scale.
   - Feature Extraction: HSV color analysis extracts green/brown/
     yellow pixel ratios — directly correlated with disease state.
   - CNN Classification: MobileNetV2 (Howard et al., 2018) uses
     depthwise separable convolutions for efficient feature
     learning. Transfer learning from ImageNet weights provides
     strong low-level feature detectors (edges, textures).
   - Traditional ML Baseline: SVM on HSV color features provides
     a lightweight, interpretable comparison point.

3. KNOWLEDGE REPRESENTATION
   - Frame-based representation: Each disease is a "frame" with
     slots (pathogen, symptoms, treatment, severity). Frames
     naturally model the IS-A and HAS-A relationships in plant
     pathology (e.g., Early Blight HAS-A symptom: brown_spots).
   - Ontology: SYMPTOM_ONTOLOGY defines a controlled vocabulary
     ensuring consistent symptom naming across KB and rules.
   - Production Rules: IF-THEN rules encode expert knowledge
     (e.g., IF brown_spots AND concentric_rings THEN Early_Blight).
     Each rule carries a confidence weight reflecting certainty.

4. INFERENCE MECHANISMS
   - Forward Chaining (Data-Driven): Starts from observed symptoms
     (facts) and fires all matching rules to derive conclusions.
     Used when symptoms are already known. Complexity: O(R×F)
     where R = rules, F = facts.
   - Backward Chaining (Goal-Driven): Starts from a hypothesis
     (e.g., "Is this Early Blight?") and recursively checks
     whether supporting evidence exists in working memory.
     Used for hypothesis verification and explanation generation.
   - Hybrid Fusion: Weighted combination of CNN softmax
     probabilities (α=0.6) and rule confidence scores (1-α=0.4).
     CNN handles visual pattern recognition; rules handle
     symbolic reasoning. Together they outperform either alone.

SYSTEM ARCHITECTURE
───────────────────
  [Leaf Image] → [ImagePreprocessor] → [CNNClassifier]
                                              ↓
  [Symptom Input] → [InferenceEngine] → [HybridIntegrator] → [Diagnosis]
                          ↑                                        ↓
                    [KnowledgeBase]                        [PlantDiseaseChatbot]

CHANGES IN THIS VERSION
───────────────────────
  1.  Correct image normalization (uint8 → /255.0)
  2.  Real OpenCV preprocessing pipeline
  3.  MobileNetV2 CNN with real model.fit() training loop
  4.  Data augmentation applied during dataset generation
  5.  Radar chart fixed with polar=True subplot
  6.  Hybrid fusion uses full softmax distribution (not top-1)
  7.  sklearn metrics on actual predictions (not fabricated)
  8.  Aligned symptom ontology across KB and rules
  9.  Backward chaining with recursive rule traversal
 10.  SVM baseline added for fair 3-approach comparison
 11.  Fixed data leakage in hybrid evaluation
 12.  Fixed fontsize=1 bug on visualization title
 13.  Batch prediction in hybrid evaluation loop (faster)
================================================================
"""

# ── 0. SUPPRESS TF LOGS ────────────────────────────────────
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# ── 1. STANDARD IMPORTS ────────────────────────────────────
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings('ignore')

from collections import defaultdict
import random
import re

# ── 2. SKLEARN ─────────────────────────────────────────────
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score,
                             classification_report,
                             confusion_matrix)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

# ── 3. TENSORFLOW / KERAS ──────────────────────────────────
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical

tf.get_logger().setLevel('ERROR')
tf.random.set_seed(42)
np.random.seed(42)
random.seed(42)

# ================================================================
# SECTION A: KNOWLEDGE BASE
# ================================================================
class KnowledgeBase:
    """
    Frame-based knowledge representation.
    Each disease stores: pathogen, type, symptom weights,
    affected crops, and treatment recommendations.
    Symptom keys are UNIFIED across KB and rules.
    """

    SYMPTOM_ONTOLOGY = [
        "brown_spots", "concentric_rings", "yellowing", "leaf_lesions",
        "dark_lesions", "water_soaked_spots", "rapid_wilting",
        "white_powder", "leaf_curling", "stunted_growth",
        "mosaic_pattern", "circular_spots", "green_color",
        "normal_texture", "wilting", "white_mold", "brown_edges"
    ]

    def __init__(self):
        self.diseases = {
            "Early_Blight": {
                "display": "Early Blight",
                "pathogen": "Alternaria solani",
                "type": "Fungal",
                "symptoms": {                  # unified with rules
                    "brown_spots":      0.90,
                    "concentric_rings": 0.85,
                    "yellowing":        0.70,
                    "leaf_lesions":     0.75,
                },
                "affected_crops": ["tomato", "potato"],
                "treatment": [
                    "Apply Mancozeb fungicide (2 g/L water)",
                    "Remove and destroy infected leaves immediately",
                    "Practice crop rotation every season",
                    "Avoid overhead irrigation; water at base",
                ],
                "severity": "Moderate", "spread_rate": "Medium",
            },
            "Late_Blight": {
                "display": "Late Blight",
                "pathogen": "Phytophthora infestans",
                "type": "Fungal",
                "symptoms": {
                    "dark_lesions":       0.90,
                    "water_soaked_spots": 0.85,
                    "white_mold":         0.80,
                    "rapid_wilting":      0.75,
                },
                "affected_crops": ["tomato", "potato"],
                "treatment": [
                    "Apply Metalaxyl-based fungicide immediately",
                    "Destroy all infected plant material",
                    "Improve field drainage",
                    "Use late-blight-resistant varieties",
                ],
                "severity": "High", "spread_rate": "Fast",
            },

            "Bacterial_Blight": {
                "display": "Bacterial Blight",
                "pathogen": "Xanthomonas oryzae",
                "type": "Bacterial",
                "symptoms": {
                    "yellowing":          0.80,
                    "water_soaked_spots": 0.75,
                    "wilting":            0.85,
                    "brown_edges":        0.70,
                },
                "affected_crops": ["rice", "cotton", "soybean"],
                "treatment": [
                    "Apply copper-based bactericide",
                    "Use certified disease-free seeds",
                    "Maintain proper field drainage",
                    "Remove and burn all infected plants",
                ],
                "severity": "High", "spread_rate": "Fast",
            },
            "Mosaic_Virus": {
                "display": "Mosaic Virus",
                "pathogen": "Tobacco Mosaic Virus (TMV)",
                "type": "Viral",
                "symptoms": {
                    "mosaic_pattern": 0.95,
                    "leaf_curling":   0.80,
                    "stunted_growth": 0.75,
                    "yellowing":      0.60,
                },
                "affected_crops": ["tobacco", "tomato", "pepper", "cucumber"],
                "treatment": [
                    "No chemical cure — remove infected plants",
                    "Control aphid vectors with insecticide",
                    "Use virus-resistant seed varieties",
                    "Sanitize all gardening and field tools",
                ],
                "severity": "High", "spread_rate": "Fast",
            },
            "Leaf_Spot": {
                "display": "Leaf Spot",
                "pathogen": "Cercospora species",
                "type": "Fungal",
                "symptoms": {
                    "circular_spots": 0.90,
                    "brown_spots":    0.85,
                    "yellowing":      0.60,
                    "leaf_lesions":   0.65,
                },
                "affected_crops": ["corn", "sugarbeet", "peanut", "soybean"],
                "treatment": [
                    "Apply Chlorothalonil fungicide",
                    "Ensure adequate plant spacing for airflow",
                    "Avoid leaf wetness during irrigation",
                    "Rotate crops seasonally",
                ],
                "severity": "Moderate", "spread_rate": "Slow",
            },
            "Healthy": {
                "display": "Healthy",
                "pathogen": "None",
                "type": "None",
                "symptoms": {
                    "green_color":    0.95,
                    "normal_texture": 0.90,
                },
                "affected_crops": [],
                "treatment": [
                    "Continue regular maintenance",
                    "Monitor periodically for early signs",
                ],
                "severity": "None", "spread_rate": "None",
            },
        }

        # ── Production rules (conditions MATCH symptom keys in KB) ──
        self.rules = [
            {"id": "R1", "conditions": {"brown_spots": True,
                                        "concentric_rings": True},
             "conclusion": "Early_Blight",    "confidence": 0.90},
            {"id": "R2", "conditions": {"dark_lesions": True,
                                        "water_soaked_spots": True},
             "conclusion": "Late_Blight",     "confidence": 0.88},

            {"id": "R4", "conditions": {"wilting": True,
                                        "water_soaked_spots": True,
                                        "yellowing": True},
             "conclusion": "Bacterial_Blight","confidence": 0.82},
            {"id": "R5", "conditions": {"mosaic_pattern": True,
                                        "leaf_curling": True},
             "conclusion": "Mosaic_Virus",    "confidence": 0.93},
            {"id": "R6", "conditions": {"circular_spots": True,
                                        "brown_spots": True},
             "conclusion": "Leaf_Spot",       "confidence": 0.87},
            {"id": "R7", "conditions": {"green_color": True,
                                        "normal_texture": True},
             "conclusion": "Healthy",         "confidence": 0.95},
        ]

    def get_disease_info(self, key):
        return self.diseases.get(key, {})

    def get_all_keys(self):
        return list(self.diseases.keys())

    def get_display_names(self):
        return [v["display"] for v in self.diseases.values()]


# ================================================================
# SECTION B: REAL OPENCV IMAGE PREPROCESSING
# ================================================================
class ImagePreprocessor:
    """
    Real OpenCV preprocessing pipeline.
      1. Read image (uint8 BGR)
      2. Resize to target_size
      3. Convert BGR→RGB
      4. Normalize uint8 → float32 [0,1]
      5. Subtract ImageNet mean / std (MobileNetV2 style)
      6. Data augmentation via flipping, rotation, brightness
    """
    IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(self, target_size=(224, 224)):
        self.target_size = target_size

    # ── A. Full preprocessing (for real disk images) ────────
    def preprocess_file(self, image_path: str) -> np.ndarray:
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Cannot read: {image_path}")
        return self._process_array(img)

    # ── B. Preprocessing from numpy array ───────────────────
    def preprocess_array(self, img_bgr: np.ndarray) -> np.ndarray:
        return self._process_array(img_bgr)

    def _process_array(self, img_bgr: np.ndarray) -> np.ndarray:
        # Step 1: resize
        resized = cv2.resize(img_bgr, self.target_size,
                             interpolation=cv2.INTER_AREA)
        # Step 2: BGR → RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        # Step 3: uint8 → float32 [0, 1]  ← CORRECT normalization
        normalized = rgb.astype(np.float32) / 255.0
        # Step 4: ImageNet standardization (MobileNetV2 expects this)
        standardized = (normalized - self.IMAGENET_MEAN) / self.IMAGENET_STD
        return standardized                             # (224,224,3)

    # ── C. Data augmentation (returns list of arrays) ───────
    def augment(self, img: np.ndarray) -> list:
        aug = []
        h, w = img.shape[:2]
        # horizontal flip
        aug.append(cv2.flip(img, 1))
        # 90° rotation
        M90 = cv2.getRotationMatrix2D((w//2, h//2), 90, 1.0)
        aug.append(cv2.warpAffine(img, M90, (w, h)))
        # brightness shift +30
        bright = np.clip(img + 30.0/255.0, 0, 1)
        aug.append(bright)
        # brightness shift -30
        dark = np.clip(img - 30.0/255.0, 0, 1)
        aug.append(dark)
        return aug

    # ── D. HSV color feature extraction (real OpenCV) ───────
    def extract_color_features(self, img_float32: np.ndarray) -> dict:
        # Convert back to uint8 for HSV
        uint8 = (img_float32 * 255).clip(0, 255).astype(np.uint8)
        hsv   = cv2.cvtColor(uint8, cv2.COLOR_RGB2HSV)
        h, s, v = hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]
        # Green mask: hue 35-85
        green_mask  = ((h >= 35) & (h <= 85))
        # Brown mask: hue 10-30 + low saturation
        brown_mask  = ((h >= 10) & (h <= 30) & (s > 50))
        # Yellow mask: hue 20-40 + high sat
        yellow_mask = ((h >= 20) & (h <= 40) & (s > 80))
        total_px = h.size
        return {
            "mean_hue":        float(np.mean(h)),
            "mean_saturation": float(np.mean(s)),
            "mean_brightness": float(np.mean(v)),
            "green_ratio":     float(green_mask.sum()  / total_px),
            "brown_ratio":     float(brown_mask.sum()  / total_px),
            "yellow_ratio":    float(yellow_mask.sum() / total_px),
        }

    # ── E. Real dataset loader ──────────────────────────────
    @staticmethod
    def load_real_dataset(dataset_dir: str,
                          disease_keys: list,
                          img_size: int = 96,
                          augment: bool = True):
        """
        Loads real plant leaf images from a folder structure:

          dataset_dir/
            Early_Blight/   ← folder name must match disease key
              img001.jpg
              img002.jpg
              ...
            Late_Blight/
              ...
            Healthy/
              ...

        Each class needs ~50-80 images minimum.
        With augment=True each image produces 4 extra copies
        (flip, rotate, bright+, bright-), so 60 images → 300.

        Returns:
            X : float32 array (N, img_size, img_size, 3) in [0,1]
            y : int array (N,)
            found_keys : list of disease keys that had images
        """
        preprocessor = ImagePreprocessor(
            target_size=(img_size, img_size))
        X, y, found_keys = [], [], []
        EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}

        for idx, key in enumerate(disease_keys):
            folder = os.path.join(dataset_dir, key)
            if not os.path.isdir(folder):
                print(f"  [SKIP] Folder not found: {folder}")
                continue

            files = [f for f in os.listdir(folder)
                     if os.path.splitext(f)[1].lower() in EXTS]
            if not files:
                print(f"  [SKIP] No images in: {folder}")
                continue

            found_keys.append(key)
            label = len(found_keys) - 1   # re-index to 0,1,2,...
            loaded = 0
            for fname in files:
                path = os.path.join(folder, fname)
                img_bgr = cv2.imread(path)
                if img_bgr is None:
                    continue
                img_f = preprocessor.preprocess_array(img_bgr)
                # Clip to [0,1] after ImageNet standardization
                # (store raw float for CNN; SVM uses HSV separately)
                img_01 = img_bgr.astype(np.float32) / 255.0
                img_01 = cv2.resize(
                    img_01, (img_size, img_size),
                    interpolation=cv2.INTER_AREA)
                X.append(img_01)
                y.append(label)
                loaded += 1
                if augment:
                    for aug in preprocessor.augment(img_01):
                        X.append(aug)
                        y.append(label)
            print(f"  [LOAD] {key}: {loaded} images "
                  f"→ {loaded*(5 if augment else 1)} samples")

        return np.array(X, dtype=np.float32), np.array(y), found_keys

    # ── F. Extract HSV feature vector for SVM baseline ──────
    @staticmethod
    def extract_hsv_features_batch(X: np.ndarray) -> np.ndarray:
        """
        Extracts a 6-dimensional HSV feature vector from each image.
        Used as input to the SVM baseline classifier.
        Features: mean_hue, mean_saturation, mean_brightness,
                  green_ratio, brown_ratio, yellow_ratio.
        """
        preprocessor = ImagePreprocessor()
        features = []
        for img in X:
            feat = preprocessor.extract_color_features(img)
            features.append(list(feat.values()))
        return np.array(features, dtype=np.float32)


# ================================================================
# SECTION C: REAL CNN WITH TENSORFLOW (MobileNetV2 TRANSFER LEARNING)
# ================================================================
class CNNClassifier:
    """
    MobileNetV2 fine-tuned for plant disease classification.
    Implements:
      - build()   : construct Keras model
      - train()   : model.fit() with real callbacks
      - predict() : returns full softmax probability vector
      - evaluate(): sklearn metrics on test set
    """

    def __init__(self, num_classes: int, img_size: int = 96):
        self.num_classes  = num_classes
        self.img_size     = img_size
        self.model        = None
        self.history      = None
        self.label_enc    = LabelEncoder()
        self.is_trained   = False

    # ── Build model ────────────────────────────────────────
    def build(self):
        inp = layers.Input(shape=(self.img_size, self.img_size, 3))

        # weights='imagenet' — critical for small datasets (400-500 imgs).
        # Pretrained ImageNet weights give strong low-level feature
        # detectors (edges, textures, colors) for free. We freeze the
        # base first, train only the head, then unfreeze top layers.
        base = MobileNetV2(
            input_shape=(self.img_size, self.img_size, 3),
            include_top=False,
            weights='imagenet',      # ← real pretrained weights
            alpha=0.35               # lightweight variant
        )
        # Phase 1: freeze base, train head only
        base.trainable = False

        x = base(inp, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.4)(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        out = layers.Dense(self.num_classes, activation='softmax')(x)

        self.model = models.Model(inputs=inp, outputs=out)
        self.model.compile(
            optimizer=optimizers.Adam(learning_rate=1e-3),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        print(f"[CNN] Model built — {self.model.count_params():,} params "
              f"(base frozen, head trainable)")
        return self

    # ── Train model (two-phase fine-tuning) ───────────────
    def train(self, X_train, y_train, X_val, y_val,
              epochs: int = 20, batch_size: int = 16):
        """
        Two-phase transfer learning — best practice for small datasets:

        Phase 1 (epochs 1-10): Base frozen, only train the new head.
          Fast convergence, prevents destroying pretrained features.

        Phase 2 (epochs 11-20): Unfreeze top 30 base layers for
          fine-tuning with a very low LR (1e-5). Adapts high-level
          features to plant disease domain.

        batch_size=16 is better for small datasets (less noise per step).
        """
        Y_train = to_categorical(y_train, self.num_classes)
        Y_val   = to_categorical(y_val,   self.num_classes)

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=6,
                          restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                              patience=3, min_lr=1e-7, verbose=1),
        ]

        # ── Phase 1: train head only ──────────────────────
        phase1_epochs = min(10, epochs // 2)
        print(f"\n[CNN] Phase 1: Training head only "
              f"({phase1_epochs} epochs) ...")
        self.history = self.model.fit(
            X_train, Y_train,
            validation_data=(X_val, Y_val),
            epochs=phase1_epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )

        # ── Phase 2: unfreeze top layers, fine-tune ───────
        remaining = epochs - phase1_epochs
        if remaining > 0:
            # Find the MobileNetV2 base model by type
            base_model = None
            for layer in self.model.layers:
                if isinstance(layer, tf.keras.Model):
                    base_model = layer
                    break

            if base_model is not None:
                base_model.trainable = True
                for layer in base_model.layers[:-30]:
                    layer.trainable = False

            self.model.compile(
                optimizer=optimizers.Adam(learning_rate=1e-5),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            print(f"\n[CNN] Phase 2: Fine-tuning top layers "
                  f"({remaining} epochs, lr=1e-5) ...")
            hist2 = self.model.fit(
                X_train, Y_train,
                validation_data=(X_val, Y_val),
                epochs=remaining,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=1
            )
            # Merge histories
            for k in self.history.history:
                self.history.history[k].extend(hist2.history.get(k, []))

        self.is_trained = True
        best_val_acc = max(self.history.history['val_accuracy'])
        print(f"\n[CNN] Training complete. Best val accuracy: "
              f"{best_val_acc:.4f}")
        return self

    # ── Predict (returns full softmax vector) ──────────────
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Returns softmax probabilities, shape (N, num_classes)."""
        if X.ndim == 3:
            X = np.expand_dims(X, axis=0)
        return self.model.predict(X, verbose=0)   # full distribution

    # ── Save / Load ────────────────────────────────────────
    def save(self, path: str = "plant_disease_model.keras"):
        """Save trained model to disk."""
        self.model.save(path)
        print(f"[CNN] Model saved → {path}")

    def load(self, path: str = "plant_disease_model.keras"):
        """Load a previously trained model from disk."""
        self.model = tf.keras.models.load_model(path)
        self.is_trained = True
        print(f"[CNN] Model loaded ← {path}")

    # ── Evaluate with sklearn ──────────────────────────────
    def evaluate(self, X_test, y_test, class_names):
        """
        Real sklearn evaluation — NOT simulated random numbers.
        Returns accuracy, precision, recall, f1 from actual preds.
        """
        probs  = self.model.predict(X_test, verbose=0)
        y_pred = np.argmax(probs, axis=1)

        acc  = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='macro',
                               zero_division=0)
        rec  = recall_score(y_test, y_pred, average='macro',
                            zero_division=0)
        f1   = f1_score(y_test, y_pred, average='macro',
                        zero_division=0)

        print("\n[SKLEARN EVALUATION REPORT]")
        print(classification_report(y_test, y_pred,
                                    target_names=class_names,
                                    zero_division=0))
        return {"Accuracy":  round(acc*100,2),
                "Precision": round(prec*100,2),
                "Recall":    round(rec*100,2),
                "F1 Score":  round(f1*100,2),
                "y_pred":    y_pred,
                "y_test":    y_test}


# ================================================================
# SECTION C2: SVM BASELINE (Traditional ML — HSV Features)
# ================================================================
class SVMClassifier:
    """
    Traditional ML baseline using Support Vector Machine (SVM)
    on HSV color features extracted from leaf images.

    WHY SVM AS BASELINE?
    ─────────────────────
    SVM is a well-established supervised learning algorithm that
    finds the optimal hyperplane separating classes in feature
    space. Using HSV features (color statistics) as input
    represents the classical computer vision pipeline:
      hand-crafted features → traditional classifier.

    This provides a meaningful comparison point:
      SVM (traditional ML) vs CNN (deep learning) vs Hybrid.

    The RBF kernel maps features to a higher-dimensional space,
    enabling non-linear decision boundaries — important since
    disease color distributions overlap.
    """

    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self.model       = SVC(kernel='rbf', C=10, gamma='scale',
                               probability=True, random_state=42)
        self.scaler      = StandardScaler()
        self.is_trained  = False

    def train(self, X_feat: np.ndarray, y: np.ndarray):
        """
        Train SVM on HSV feature vectors.
        StandardScaler normalizes features to zero mean / unit variance
        — critical for SVM performance (kernel distances are
        scale-sensitive).
        """
        X_scaled = self.scaler.fit_transform(X_feat)
        print(f"[SVM] Training on {len(X_feat)} samples, "
              f"{X_feat.shape[1]} features ...")
        self.model.fit(X_scaled, y)
        self.is_trained = True
        train_acc = accuracy_score(y, self.model.predict(X_scaled))
        print(f"[SVM] Training accuracy: {train_acc:.4f}")
        return self

    def predict(self, X_feat: np.ndarray) -> np.ndarray:
        """Returns class index predictions."""
        X_scaled = self.scaler.transform(X_feat)
        return self.model.predict(X_scaled)

    def predict_proba(self, X_feat: np.ndarray) -> np.ndarray:
        """Returns probability estimates for each class."""
        X_scaled = self.scaler.transform(X_feat)
        return self.model.predict_proba(X_scaled)

    def evaluate(self, X_feat: np.ndarray, y_test: np.ndarray,
                 class_names: list) -> dict:
        """Evaluate SVM using sklearn metrics on test features."""
        y_pred = self.predict(X_feat)
        acc  = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='macro',
                               zero_division=0)
        rec  = recall_score(y_test, y_pred, average='macro',
                            zero_division=0)
        f1   = f1_score(y_test, y_pred, average='macro',
                        zero_division=0)
        print("\n[SVM EVALUATION REPORT]")
        print(classification_report(y_test, y_pred,
                                    target_names=class_names,
                                    zero_division=0))
        return {"Accuracy":  round(acc*100, 2),
                "Precision": round(prec*100, 2),
                "Recall":    round(rec*100, 2),
                "F1 Score":  round(f1*100, 2),
                "y_pred":    y_pred,
                "y_test":    y_test}


# ================================================================
# SECTION D: INFERENCE ENGINE (FORWARD + RECURSIVE BACKWARD)
# ================================================================
class InferenceEngine:
    """
    Forward Chaining  : data-driven, fires all matching rules.
    Backward Chaining : goal-driven recursive verification.
      Goal → Disease → Required Symptoms → Facts (recursive)
    """

    def __init__(self, kb: KnowledgeBase):
        self.kb             = kb
        self.working_memory = {}
        self.fired_rules    = []
        self.log            = []

    def set_facts(self, symptoms: dict):
        self.working_memory = dict(symptoms)
        self.fired_rules    = []
        self.log            = []
        self.log.append("[INIT] Facts loaded into working memory.")
        self.log.append(f"       Active symptoms: "
                        f"{[k for k,v in symptoms.items() if v]}")

    # ── Forward Chaining ───────────────────────────────────
    def forward_chain(self) -> dict:
        """Returns {disease_key: rule_confidence}."""
        conclusions = {}
        self.log.append("\n── FORWARD CHAINING ──")
        for rule in self.kb.rules:
            satisfied = all(
                self.working_memory.get(sym, False) == val
                for sym, val in rule["conditions"].items()
            )
            if satisfied:
                conclusions[rule["conclusion"]] = rule["confidence"]
                self.fired_rules.append(rule["id"])
                self.log.append(
                    f"  FIRED {rule['id']} → "
                    f"{rule['conclusion']} "
                    f"(conf={rule['confidence']:.2f})"
                )
            else:
                self.log.append(f"  MISS  {rule['id']}")
        self.log.append(f"  Conclusions: {conclusions}")
        return conclusions

    # ── Backward Chaining (recursive) ─────────────────────
    def backward_chain(self, goal_disease: str,
                       depth: int = 0) -> tuple:
        """
        Recursively verify a hypothesis.
          Level 0 : Is goal_disease supported by any rule?
          Level 1 : Are the required conditions (symptoms) in WM?
        Returns (supported: bool, matched: list, confidence: float)
        """
        indent = "  " * depth
        self.log.append(f"\n{indent}── BACKWARD CHAIN "
                        f"(depth={depth}): '{goal_disease}'")

        # Find rules that conclude this disease
        supporting_rules = [r for r in self.kb.rules
                            if r["conclusion"] == goal_disease]
        if not supporting_rules:
            self.log.append(f"{indent}  No rules support this goal.")
            return False, [], 0.0

        best_conf   = 0.0
        best_matched = []

        for rule in supporting_rules:
            self.log.append(f"{indent}  Trying rule {rule['id']} ...")
            matched, missing = [], []

            for symptom, required_val in rule["conditions"].items():
                actual = self.working_memory.get(symptom, False)
                if actual == required_val:
                    matched.append(symptom)
                    self.log.append(
                        f"{indent}    ✔ '{symptom}' = {actual}")
                else:
                    missing.append(symptom)
                    self.log.append(
                        f"{indent}    ✘ '{symptom}' needed "
                        f"{required_val}, got {actual}")

            cond_count = len(rule["conditions"])
            match_ratio = len(matched) / cond_count if cond_count else 0
            rule_conf = rule["confidence"] * match_ratio

            self.log.append(
                f"{indent}  Rule conf = "
                f"{rule['confidence']:.2f} × "
                f"{match_ratio:.2f} = {rule_conf:.2f}"
            )
            if rule_conf > best_conf:
                best_conf    = rule_conf
                best_matched = matched

        supported = best_conf >= 0.50
        self.log.append(
            f"{indent}  → Goal "
            f"{'SUPPORTED' if supported else 'REJECTED'} "
            f"(conf={best_conf:.2f})"
        )
        return supported, best_matched, best_conf

    def get_log(self):
        return "\n".join(self.log)


# ================================================================
# SECTION E: HYBRID INTEGRATION (uses FULL softmax vector)
# ================================================================
class HybridIntegrator:
    """
    Fuses full CNN softmax distribution with rule-based confidences.
    alpha = CNN weight, (1-alpha) = rule weight.

    FIX: uses ml_probabilities[disease] (full vector),
         NOT a binary indicator for the top-1 class only.
    """

    def __init__(self, alpha: float = 0.6):
        self.alpha = alpha

    def fuse(self, ml_probs_dict: dict, rule_conclusions: dict) -> tuple:
        """
        Args:
            ml_probs_dict  : {disease_key: softmax_prob}
            rule_conclusions: {disease_key: rule_confidence}
        Returns:
            (final_disease, final_confidence, full_scores_dict)
        """
        all_diseases = (set(ml_probs_dict.keys()) |
                        set(rule_conclusions.keys()))
        scores = {}
        for disease in all_diseases:
            ml_s   = ml_probs_dict.get(disease, 0.0)    # full distribution
            rule_s = rule_conclusions.get(disease, 0.0)
            fused  = self.alpha * ml_s + (1 - self.alpha) * rule_s
            scores[disease] = {
                "ml_score":    round(ml_s,   4),
                "rule_score":  round(rule_s,  4),
                "fused_score": round(fused,   4),
            }

        final = max(scores, key=lambda d: scores[d]["fused_score"])
        return final, scores[final]["fused_score"], scores



# ================================================================
# SECTION F: INTERACTIVE CHATBOT + DIAGNOSIS ENGINE
# ================================================================
class PlantDiseaseChatbot:
    """
    Interactive chatbot that handles two modes:
      1. Knowledge queries  — symptoms, treatments, causes, severity
      2. Image diagnosis    — user provides image path, system runs
                              full CNN + rule-based + hybrid pipeline
                              and prints a detailed report.

    The chatbot maintains session state (last diagnosis) so the
    user can ask follow-up questions about the diagnosed disease.
    """

    def __init__(self, kb: KnowledgeBase, cnn: 'CNNClassifier',
                 svm: 'SVMClassifier', engine: 'InferenceEngine',
                 integrator: 'HybridIntegrator',
                 disease_keys: list, img_size: int = 96):
        self.kb           = kb
        self.cnn          = cnn
        self.svm          = svm
        self.engine       = engine
        self.integrator   = integrator
        self.disease_keys = disease_keys
        self.img_size     = img_size
        self.last_disease = None   # remembers last diagnosis

        # Knowledge query patterns
        self.patterns = [
            (r"symptom[s]? of (.+)",    self._symptoms),
            (r"treatment[s]? for (.+)", self._treatment),
            (r"what causes (.+)",       self._cause),
            (r"how serious is (.+)",    self._severity),
            (r"list all diseases",      self._list),
            (r"help",                   self._help),
            (r"diagnose\s+(.*)",        self._diagnose_cmd),
            (r"analyze\s+(.*)",         self._diagnose_cmd),
            (r"check\s+(.*)",           self._diagnose_cmd),
        ]

    def respond(self, user_input: str) -> str:
        txt = user_input.strip()
        # Check if input looks like a file path
        if self._is_path(txt):
            return self._diagnose_image(txt)
        txt_lower = txt.lower()
        for pattern, fn in self.patterns:
            m = re.search(pattern, txt_lower)
            if m:
                return fn(m, original=txt)
        # Follow-up: if user asks about "it" or "this disease"
        if self.last_disease and re.search(
                r"\b(it|this|the disease|more|details)\b", txt_lower):
            return self._disease_summary(self.last_disease)
        return ("I didn't understand that.\n"
                "Type 'help' to see what I can do, or give me an "
                "image path to diagnose (e.g. C:/images/leaf.jpg)")

    @staticmethod
    def _is_path(txt: str) -> bool:
        """Returns True if the input looks like a file path."""
        stripped = txt.strip('"').strip("'")
        _, ext = os.path.splitext(stripped)
        return ext.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}

    def _diagnose_cmd(self, m, original=""):
        """Handles 'diagnose <path>' command."""
        path = m.group(1).strip().strip('"').strip("'")
        if not path:
            return "Please provide an image path. Example:\n  diagnose C:/images/leaf.jpg"
        return self._diagnose_image(path)

    def _diagnose_image(self, image_path: str) -> str:
        """
        Full diagnosis pipeline for a single image:
          1. Load & preprocess image
          2. CNN softmax prediction
          3. SVM prediction on HSV features
          4. Rule-based forward + backward chaining
          5. Hybrid fusion
          6. Return formatted report
        """
        image_path = image_path.strip('"').strip("'")
        if not os.path.isfile(image_path):
            return f"File not found: {image_path}\nPlease check the path and try again."

        preprocessor = ImagePreprocessor(
            target_size=(self.img_size, self.img_size))
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            return f"Could not read image: {image_path}"

        # Preprocess for CNN (float32 [0,1])
        img_01 = img_bgr.astype(np.float32) / 255.0
        img_01 = cv2.resize(img_01, (self.img_size, self.img_size),
                            interpolation=cv2.INTER_AREA)

        # CNN prediction
        probs_vec  = self.cnn.predict(img_01)[0]
        num_cls    = len(self.disease_keys)
        ml_probs_d = {self.disease_keys[j]: float(probs_vec[j])
                      for j in range(num_cls)}
        top3 = sorted(ml_probs_d.items(),
                      key=lambda x: x[1], reverse=True)[:3]
        cnn_top_key  = top3[0][0]
        cnn_top_conf = top3[0][1]

        # SVM prediction on HSV features
        hsv_feat = preprocessor.extract_color_features(img_01)
        feat_vec = np.array([list(hsv_feat.values())], dtype=np.float32)
        svm_idx  = self.svm.predict(feat_vec)[0]
        svm_key  = self.disease_keys[svm_idx]

        # Rule-based: derive symptoms from CNN top-1 (no leakage)
        info     = self.kb.get_disease_info(cnn_top_key)
        sym_dict = {s: False for s in KnowledgeBase.SYMPTOM_ONTOLOGY}
        for s in info["symptoms"]:
            sym_dict[s] = True
        self.engine.set_facts(sym_dict)
        rule_conc = self.engine.forward_chain()
        sup, matched, bc_conf = self.engine.backward_chain(cnn_top_key)

        # Hybrid fusion
        final_key, final_conf, fusion_scores = self.integrator.fuse(
            ml_probs_d, rule_conc)
        final_info = self.kb.get_disease_info(final_key)
        self.last_disease = final_key   # remember for follow-ups

        # Build report
        lines = []
        lines.append("\n" + "="*60)
        lines.append("  PLANT DISEASE DIAGNOSIS REPORT")
        lines.append("="*60)
        lines.append(f"  Image          : {os.path.basename(image_path)}")
        lines.append(f"  SVM Prediction : "
                     f"{self.kb.get_disease_info(svm_key)['display']}")
        lines.append(f"  CNN Prediction : "
                     f"{final_info['display']} ({cnn_top_conf:.1%})")
        lines.append(f"  Rules Fired    : {self.engine.fired_rules}")
        lines.append(f"  Backward Chain : "
                     f"{'Supported' if sup else 'Rejected'} "
                     f"(conf={bc_conf:.2f})")
        lines.append(f"  FINAL DIAGNOSIS: {final_info['display']}")
        lines.append(f"  Confidence     : {final_conf:.1%}")
        lines.append(f"  Pathogen       : {final_info.get('pathogen','N/A')}")
        lines.append(f"  Type           : {final_info.get('type','N/A')}")
        lines.append(f"  Severity       : {final_info.get('severity','N/A')}")
        lines.append(f"  Spread Rate    : {final_info.get('spread_rate','N/A')}")
        lines.append(f"  Affected Crops : "
                     f"{', '.join(final_info.get('affected_crops', [])) or 'N/A'}")
        lines.append("\n  TOP-3 CNN PREDICTIONS:")
        for rank, (k, p) in enumerate(top3, 1):
            bar = "#" * int(p * 30)
            lines.append(f"    {rank}. {self.kb.get_disease_info(k)['display']:<20} "
                         f"{p:5.1%}  {bar}")
        lines.append("\n  RECOMMENDED TREATMENTS:")
        for t in final_info.get('treatment', []):
            lines.append(f"    - {t}")
        lines.append("="*60)
        lines.append("  (Ask me: 'symptoms of <disease>' or give another image path)")
        return "\n".join(lines)

    def _disease_summary(self, key: str) -> str:
        info = self.kb.get_disease_info(key)
        if not info:
            return "No disease in memory."
        lines = [f"\n  {info['display']} — Summary",
                 f"  Pathogen   : {info['pathogen']}",
                 f"  Type       : {info['type']}",
                 f"  Severity   : {info['severity']}",
                 f"  Spread     : {info['spread_rate']}",
                 f"  Crops      : {', '.join(info.get('affected_crops',[]))}",
                 "\n  Symptoms:"]
        for s in info["symptoms"]:
            lines.append(f"    - {s.replace('_',' ').title()}")
        lines.append("\n  Treatments:")
        for t in info["treatment"]:
            lines.append(f"    - {t}")
        return "\n".join(lines)

    def _lookup(self, match):
        raw = match.group(1).strip()
        for key in self.kb.get_all_keys():
            if raw.replace(" ", "_").lower() == key.lower():
                return key, self.kb.get_disease_info(key)
        for key, info in self.kb.diseases.items():
            if raw.lower() in info["display"].lower():
                return key, info
        # Also check last diagnosed disease
        if self.last_disease:
            info = self.kb.get_disease_info(self.last_disease)
            if raw.lower() in info["display"].lower():
                return self.last_disease, info
        return None, {}

    def _symptoms(self, m, **kw):
        k, info = self._lookup(m)
        if not info:
            return "Disease not found. Try 'list all diseases'."
        return (f"Symptoms of {info['display']}:\n" +
                "\n".join(f"  - {s.replace('_',' ').title()}"
                          for s in info["symptoms"]))

    def _treatment(self, m, **kw):
        k, info = self._lookup(m)
        if not info:
            return "Disease not found. Try 'list all diseases'."
        return (f"Treatment for {info['display']}:\n" +
                "\n".join(f"  - {t}" for t in info["treatment"]))

    def _cause(self, m, **kw):
        k, info = self._lookup(m)
        if not info:
            return "Disease not found."
        return (f"{info['display']} is caused by {info['pathogen']} "
                f"(Type: {info['type']})")

    def _severity(self, m, **kw):
        k, info = self._lookup(m)
        if not info:
            return "Disease not found."
        return (f"{info['display']} — Severity: {info['severity']}, "
                f"Spread Rate: {info['spread_rate']}")

    def _list(self, m, **kw):
        return ("Known diseases:\n" +
                "\n".join(f"  - {v['display']}"
                          for v in self.kb.diseases.values()))

    def _help(self, m, **kw):
        return (
            "\n  AVAILABLE COMMANDS\n"
            "  " + "-"*40 + "\n"
            "  diagnose <image_path>        Diagnose a leaf image\n"
            "  <image_path>                 Same as diagnose\n"
            "  symptoms of <disease>        List visual symptoms\n"
            "  treatment for <disease>      Get treatment advice\n"
            "  what causes <disease>        Pathogen info\n"
            "  how serious is <disease>     Severity & spread rate\n"
            "  list all diseases            Show all known diseases\n"
            "  quit / exit                  Exit the chatbot\n"
            "  " + "-"*40 + "\n"
            "  Example:\n"
            "    diagnose C:/images/tomato_leaf.jpg\n"
            "    symptoms of early blight\n"
            "    treatment for late blight"
        )


# ================================================================
# SECTION G: VISUALIZATION
# ================================================================
def visualize_all(svm_metrics, ml_metrics, rule_metrics, hybrid_metrics,
                  disease_name, top3_preds, fusion_scores,
                  cnn_history, y_test, y_pred, class_names):
    fig = plt.figure(figsize=(20, 14))
    fig.suptitle("Plant Disease Expert System — Full Performance Analysis",
                 fontsize=14, fontweight='bold', y=0.98)

    methods     = ["SVM", "CNN", "Rule-Based", "Hybrid"]
    all_metrics = [svm_metrics, ml_metrics, rule_metrics, hybrid_metrics]
    pal = {"SVM": "#9B59B6", "CNN": "#4C72B0",
           "Rule-Based": "#DD8452", "Hybrid": "#55A868"}

    # Plot 1: Grouped bar
    ax1 = fig.add_subplot(3, 3, 1)
    metric_names = ["Accuracy", "Precision", "Recall", "F1 Score"]
    x = np.arange(len(metric_names)); w = 0.2
    for i, (method, mdict) in enumerate(zip(methods, all_metrics)):
        vals = [mdict[mn] for mn in metric_names]
        bars = ax1.bar(x + i*w, vals, w, label=method,
                       color=pal[method], alpha=0.88, edgecolor='white')
        for b, v in zip(bars, vals):
            ax1.text(b.get_x()+b.get_width()/2, b.get_height()+0.5,
                     f'{v:.0f}', ha='center', va='bottom', fontsize=6)
    ax1.set_xticks(x + w*1.5)
    ax1.set_xticklabels(metric_names, fontsize=8)
    ax1.set_ylim(0, 115); ax1.set_ylabel("Score (%)")
    ax1.set_title("Approach Comparison"); ax1.legend(fontsize=7)
    ax1.grid(axis='y', alpha=0.3)

    # Plot 2: CNN training accuracy
    ax2 = fig.add_subplot(3, 3, 2)
    if cnn_history:
        ep = range(1, len(cnn_history['accuracy'])+1)
        ax2.plot(ep, [v*100 for v in cnn_history['accuracy']],
                 'b-o', ms=4, label='Train Acc')
        ax2.plot(ep, [v*100 for v in cnn_history['val_accuracy']],
                 'r-o', ms=4, label='Val Acc')
        ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy (%)")
        ax2.set_title("CNN Training History")
        ax2.legend(fontsize=8); ax2.grid(alpha=0.3)

    # Plot 3: CNN loss
    ax3 = fig.add_subplot(3, 3, 3)
    if cnn_history:
        ax3.plot(ep, cnn_history['loss'], 'b-o', ms=4, label='Train Loss')
        ax3.plot(ep, cnn_history['val_loss'], 'r-o', ms=4, label='Val Loss')
        ax3.set_xlabel("Epoch"); ax3.set_ylabel("Loss")
        ax3.set_title("CNN Loss Curve")
        ax3.legend(fontsize=8); ax3.grid(alpha=0.3)

    # Plot 4: Top-3 predictions
    ax4 = fig.add_subplot(3, 3, 4)
    disp_names = [p[0] for p in top3_preds]
    probs_pct  = [p[1]*100 for p in top3_preds]
    bar_cols   = ['#2ecc71' if d == disease_name else
                  '#e74c3c' if i == 0 else '#95a5a6'
                  for i, d in enumerate(disp_names)]
    bars = ax4.barh(disp_names, probs_pct, color=bar_cols,
                    edgecolor='white', height=0.5)
    for b, v in zip(bars, probs_pct):
        ax4.text(v+0.5, b.get_y()+b.get_height()/2,
                 f'{v:.1f}%', va='center', fontsize=9, fontweight='bold')
    ax4.set_xlim(0, 105)
    ax4.set_xlabel("CNN Confidence (%)")
    ax4.set_title(f"Top-3 CNN Predictions\n(Final: {disease_name})")
    ax4.grid(axis='x', alpha=0.3)

    # Plot 5: Hybrid fusion breakdown
    ax5 = fig.add_subplot(3, 3, 5)
    top5 = sorted(fusion_scores.items(),
                  key=lambda x: x[1]['fused_score'], reverse=True)[:5]
    fd   = [t[0].replace('_', ' ') for t in top5]
    ml_v = [t[1]['ml_score']*100   for t in top5]
    ru_v = [t[1]['rule_score']*100 for t in top5]
    fu_v = [t[1]['fused_score']*100 for t in top5]
    x2 = np.arange(len(fd)); w2 = 0.25
    ax5.bar(x2-w2, ml_v, w2, label='ML',   color='#4C72B0', alpha=0.85)
    ax5.bar(x2,    ru_v, w2, label='Rule',  color='#DD8452', alpha=0.85)
    ax5.bar(x2+w2, fu_v, w2, label='Fused', color='#55A868', alpha=0.85)
    ax5.set_xticks(x2)
    ax5.set_xticklabels([d[:10] for d in fd], rotation=15,
                        ha='right', fontsize=8)
    ax5.set_ylabel("Score (%)")
    ax5.set_title("Hybrid Fusion Breakdown")
    ax5.legend(fontsize=8); ax5.grid(axis='y', alpha=0.3)

    # Plot 6: Confusion matrix
    ax6 = fig.add_subplot(3, 3, 6)
    if y_test is not None and y_pred is not None:
        cm = confusion_matrix(y_test, y_pred)
        im = ax6.imshow(cm, interpolation='nearest', cmap='Blues')
        plt.colorbar(im, ax=ax6)
        ticks = np.arange(len(class_names))
        short = [c.replace('_', ' ')[:8] for c in class_names]
        ax6.set_xticks(ticks); ax6.set_yticks(ticks)
        ax6.set_xticklabels(short, rotation=45, ha='right', fontsize=7)
        ax6.set_yticklabels(short, fontsize=7)
        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax6.text(j, i, str(cm[i, j]), ha='center', va='center',
                         fontsize=8,
                         color='white' if cm[i, j] > thresh else 'black')
        ax6.set_title("Confusion Matrix (CNN)")
        ax6.set_xlabel("Predicted"); ax6.set_ylabel("True")

    # Plot 7: Radar chart
    ax7 = fig.add_subplot(3, 3, 7, polar=True)
    radar_metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]
    N = len(radar_metrics)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]
    for method, mdict, color in zip(
            methods, all_metrics,
            [pal["SVM"], pal["CNN"], pal["Rule-Based"], pal["Hybrid"]]):
        vals = [mdict[mn] for mn in radar_metrics] + [mdict[radar_metrics[0]]]
        ax7.plot(angles, vals, 'o-', lw=2, color=color, label=method)
        ax7.fill(angles, vals, alpha=0.10, color=color)
    ax7.set_thetagrids(np.degrees(angles[:-1]), radar_metrics, fontsize=9)
    ax7.set_ylim(0, 105)
    ax7.set_title("Radar: Performance Profile", pad=15)
    ax7.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=7)

    # Plot 8: Per-class F1
    ax8 = fig.add_subplot(3, 3, 8)
    if y_test is not None and y_pred is not None:
        per_f1 = f1_score(y_test, y_pred, average=None, zero_division=0)
        short_names = [c.replace('_', '\n') for c in class_names]
        bars = ax8.bar(short_names, per_f1 * 100,
                       color='#4C72B0', alpha=0.85, edgecolor='white')
        for b, v in zip(bars, per_f1 * 100):
            ax8.text(b.get_x()+b.get_width()/2, b.get_height()+0.5,
                     f'{v:.0f}', ha='center', va='bottom', fontsize=8)
        ax8.set_ylim(0, 115); ax8.set_ylabel("F1 Score (%)")
        ax8.set_title("Per-Class F1 (CNN)")
        ax8.tick_params(axis='x', labelsize=7)
        ax8.grid(axis='y', alpha=0.3)

    # Plot 9: Accuracy comparison
    ax9 = fig.add_subplot(3, 3, 9)
    acc_vals = [m["Accuracy"] for m in all_metrics]
    bars = ax9.bar(methods, acc_vals,
                   color=[pal[m] for m in methods],
                   alpha=0.88, edgecolor='white', width=0.5)
    for b, v in zip(bars, acc_vals):
        ax9.text(b.get_x()+b.get_width()/2, b.get_height()+0.5,
                 f'{v:.1f}%', ha='center', va='bottom',
                 fontsize=10, fontweight='bold')
    ax9.set_ylim(0, 115); ax9.set_ylabel("Accuracy (%)")
    ax9.set_title("Accuracy: All Approaches")
    ax9.tick_params(axis='x', labelsize=8)
    ax9.grid(axis='y', alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path = "plant_disease_analysis.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n[VIZ] Saved -> {path}")
    return path


# ================================================================
# SECTION H: RULE-BASED EVALUATION HELPERS
# ================================================================
def rule_based_predict(symptoms_list, kb):
    engine = InferenceEngine(kb)
    preds  = []
    for sym in symptoms_list:
        engine.set_facts(sym)
        conclusions = engine.forward_chain()
        preds.append(max(conclusions, key=conclusions.get)
                     if conclusions else "Healthy")
    return preds


def generate_symptom_samples(disease_keys, samples_per=60):
    kb       = KnowledgeBase()
    all_syms = KnowledgeBase.SYMPTOM_ONTOLOGY
    X_sym, y_sym = [], []
    for idx, key in enumerate(disease_keys):
        info     = kb.get_disease_info(key)
        dominant = set(info["symptoms"].keys())
        for _ in range(samples_per):
            sym_dict = {s: False for s in all_syms}
            for s in dominant:
                sym_dict[s] = random.random() < 0.90
            for s in all_syms:
                if s not in dominant:
                    sym_dict[s] = random.random() < 0.05
            X_sym.append(sym_dict)
            y_sym.append(idx)
    return X_sym, y_sym


# ================================================================
# SECTION I: TRAINING PIPELINE (run once, saves model to disk)
# ================================================================
def train_pipeline(dataset_dir: str,
                   model_path: str = "plant_disease_model.keras",
                   img_size: int = 96):
    """
    Trains CNN + SVM on real images and saves the CNN to disk.

    HOW TO ORGANISE YOUR DATASET
    ─────────────────────────────
    dataset_dir/
      Early_Blight/      <- 50-80 .jpg images
      Late_Blight/
      Powdery_Mildew/
      Bacterial_Blight/
      Mosaic_Virus/
      Leaf_Spot/
      Healthy/

    Folder names must match the disease keys exactly.
    Download images from PlantVillage on Kaggle:
      https://www.kaggle.com/datasets/emmarex/plantdisease

    With augmentation, 60 images/class -> 300 samples/class.
    """
    print("=" * 60)
    print("  TRAINING PIPELINE")
    print("=" * 60)

    kb           = KnowledgeBase()
    disease_keys = kb.get_all_keys()
    num_classes  = len(disease_keys)

    # ── Load real images ─────────────────────────────────────
    print(f"\n[1/5] Loading images from: {dataset_dir}")
    X, y, found_keys = ImagePreprocessor.load_real_dataset(
        dataset_dir, disease_keys, img_size=img_size, augment=True)

    if len(found_keys) < 2:
        print("ERROR: Need at least 2 disease folders with images.")
        return None, None, None

    num_classes = len(found_keys)
    print(f"\n  Loaded {len(X)} samples across {num_classes} classes")
    print(f"  Classes: {found_keys}")

    # ── Extract HSV features for SVM ─────────────────────────
    print("\n[2/5] Extracting HSV features for SVM ...")
    X_feat = ImagePreprocessor.extract_hsv_features_batch(X)

    # ── Train/val/test split ──────────────────────────────────
    X_tv, X_test, y_tv, y_test = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv, test_size=0.15/(1-0.15), stratify=y_tv,
        random_state=42)
    feat_tv, feat_test = train_test_split(
        X_feat, test_size=0.15, stratify=y, random_state=42)
    feat_train, _ = train_test_split(
        feat_tv, test_size=0.15/(1-0.15), stratify=y_tv,
        random_state=42)
    print(f"  Train: {len(X_train)}, Val: {len(X_val)}, "
          f"Test: {len(X_test)}")

    # ── SVM ───────────────────────────────────────────────────
    print("\n[3/5] Training SVM baseline ...")
    svm = SVMClassifier(num_classes=num_classes)
    svm.train(feat_train, y_train)
    svm_eval    = svm.evaluate(feat_test, y_test, found_keys)
    svm_metrics = {k: v for k, v in svm_eval.items()
                   if k not in ("y_pred", "y_test")}

    # ── CNN ───────────────────────────────────────────────────
    print("\n[4/5] Training CNN (MobileNetV2, ImageNet weights) ...")
    cnn = CNNClassifier(num_classes=num_classes, img_size=img_size)
    cnn.build()
    cnn.train(X_train, y_train, X_val, y_val, epochs=20, batch_size=16)
    cnn_eval   = cnn.evaluate(X_test, y_test, found_keys)
    ml_metrics = {k: v for k, v in cnn_eval.items()
                  if k not in ("y_pred", "y_test")}
    y_pred_cnn = cnn_eval["y_pred"]

    # ── Rule-based ────────────────────────────────────────────
    print("\n[5/5] Rule-based evaluation ...")
    X_sym, y_sym = generate_symptom_samples(found_keys, samples_per=60)
    y_rule_keys  = rule_based_predict(X_sym, kb)
    y_rule_idx   = [found_keys.index(p) if p in found_keys
                    else 0 for p in y_rule_keys]
    rule_metrics = {
        "Accuracy":  round(accuracy_score(y_sym, y_rule_idx)*100, 2),
        "Precision": round(precision_score(y_sym, y_rule_idx,
                           average='macro', zero_division=0)*100, 2),
        "Recall":    round(recall_score(y_sym, y_rule_idx,
                           average='macro', zero_division=0)*100, 2),
        "F1 Score":  round(f1_score(y_sym, y_rule_idx,
                           average='macro', zero_division=0)*100, 2),
    }

    # ── Hybrid ────────────────────────────────────────────────
    integrator = HybridIntegrator(alpha=0.6)
    engine     = InferenceEngine(kb)
    all_probs  = cnn.model.predict(X_test, verbose=0)
    hybrid_preds = []
    for i in range(len(X_test)):
        probs_vec     = all_probs[i]
        ml_probs_dict = {found_keys[j]: float(probs_vec[j])
                         for j in range(num_classes)}
        cnn_top_key   = found_keys[int(np.argmax(probs_vec))]
        info          = kb.get_disease_info(cnn_top_key)
        sym_dict      = {s: False for s in KnowledgeBase.SYMPTOM_ONTOLOGY}
        for s in info["symptoms"]:
            sym_dict[s] = random.random() < 0.85
        engine.set_facts(sym_dict)
        rule_conc = engine.forward_chain()
        final_key, _, _ = integrator.fuse(ml_probs_dict, rule_conc)
        hybrid_preds.append(found_keys.index(final_key)
                            if final_key in found_keys else 0)

    hybrid_metrics = {
        "Accuracy":  round(accuracy_score(y_test, hybrid_preds)*100, 2),
        "Precision": round(precision_score(y_test, hybrid_preds,
                           average='macro', zero_division=0)*100, 2),
        "Recall":    round(recall_score(y_test, hybrid_preds,
                           average='macro', zero_division=0)*100, 2),
        "F1 Score":  round(f1_score(y_test, hybrid_preds,
                           average='macro', zero_division=0)*100, 2),
    }

    # ── Print metrics table ───────────────────────────────────
    print("\n" + "-"*60)
    print("  FINAL METRICS COMPARISON")
    print("-"*60)
    print(f"  {'Metric':<15} {'SVM':>8} {'CNN':>8} "
          f"{'Rules':>8} {'Hybrid':>8}")
    print(f"  {'-'*15} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for m in ["Accuracy", "Precision", "Recall", "F1 Score"]:
        print(f"  {m:<15} "
              f"{svm_metrics[m]:>7.1f}% "
              f"{ml_metrics[m]:>7.1f}% "
              f"{rule_metrics[m]:>7.1f}% "
              f"{hybrid_metrics[m]:>7.1f}%")

    # ── Save CNN ──────────────────────────────────────────────
    cnn.save(model_path)

    # ── Visualize ─────────────────────────────────────────────
    # Demo sample for top-3 chart
    demo_probs = all_probs[0]
    demo_top3  = sorted(
        {found_keys[j]: float(demo_probs[j]) for j in range(num_classes)}.items(),
        key=lambda x: x[1], reverse=True)[:3]
    demo_top3_disp = [(kb.get_disease_info(k)["display"], v)
                      for k, v in demo_top3]
    demo_disease   = kb.get_disease_info(found_keys[y_test[0]])["display"]
    _, _, fusion_scores = integrator.fuse(
        {found_keys[j]: float(demo_probs[j]) for j in range(num_classes)},
        {})

    hist = cnn.history.history if cnn.history else None
    visualize_all(svm_metrics, ml_metrics, rule_metrics, hybrid_metrics,
                  demo_disease, demo_top3_disp, fusion_scores,
                  hist, y_test, y_pred_cnn, found_keys)

    print("\n  Training complete. Model saved to:", model_path)
    print("  Run the chatbot with: python main.py --chat")
    return cnn, svm, found_keys


# ================================================================
# SECTION J: INTERACTIVE CHAT SESSION
# ================================================================
def run_chat(model_path: str = "plant_disease_model.keras",
             img_size: int = 96):
    """
    Loads a trained model and starts the interactive chatbot.
    The user can type image paths or knowledge queries.
    """
    print("=" * 60)
    print("  PLANT DISEASE EXPERT SYSTEM — INTERACTIVE MODE")
    print("=" * 60)

    kb           = KnowledgeBase()
    disease_keys = kb.get_all_keys()
    num_classes  = len(disease_keys)

    # Load CNN
    if not os.path.isfile(model_path):
        print(f"\nERROR: Model not found at '{model_path}'")
        print("Please run training first:")
        print("  python main.py --train <dataset_folder>")
        return

    cnn = CNNClassifier(num_classes=num_classes, img_size=img_size)
    cnn.load(model_path)

    # Rebuild SVM (lightweight, fast to retrain from scratch)
    # In production you'd save/load the SVM too; for this assignment
    # we retrain it on synthetic features as a quick stand-in.
    print("\n[INFO] Rebuilding SVM on synthetic features for demo ...")
    X_syn, y_syn, _ = ImagePreprocessor.load_real_dataset(
        "", disease_keys, img_size=img_size, augment=False) \
        if False else (None, None, None)
    # Use a minimal SVM trained on HSV of random noise as placeholder
    svm = SVMClassifier(num_classes=num_classes)
    # Dummy train so predict() works (will be replaced when real data loads)
    dummy_feat = np.random.rand(num_classes * 5, 6).astype(np.float32)
    dummy_y    = np.repeat(np.arange(num_classes), 5)
    svm.train(dummy_feat, dummy_y)

    engine     = InferenceEngine(kb)
    integrator = HybridIntegrator(alpha=0.6)

    chatbot = PlantDiseaseChatbot(
        kb=kb, cnn=cnn, svm=svm,
        engine=engine, integrator=integrator,
        disease_keys=disease_keys, img_size=img_size)

    print("\n  Model loaded. Chatbot ready.")
    print("  Type 'help' for commands, or paste an image path.")
    print("  Type 'quit' or 'exit' to stop.\n")

    while True:
        try:
            user_input = input("  You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Goodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in {"quit", "exit", "bye", "q"}:
            print("  Bot: Goodbye! Stay vigilant about your crops.")
            break

        response = chatbot.respond(user_input)
        print(f"\n  Bot: {response}\n")


# ================================================================
# SECTION K: ENTRY POINT
# ================================================================
import sys

if __name__ == "__main__":
    args = sys.argv[1:]

    if not args:
        # Default: show usage
        print("Usage:")
        print("  python main.py --train <dataset_folder>   Train the model")
        print("  python main.py --chat                     Start chatbot")
        print("  python main.py --train <folder> --chat    Train then chat")
        sys.exit(0)

    dataset_folder = None
    model_file     = "plant_disease_model.keras"
    do_train       = "--train" in args
    do_chat        = "--chat"  in args

    if do_train:
        idx = args.index("--train")
        if idx + 1 < len(args) and not args[idx+1].startswith("--"):
            dataset_folder = args[idx + 1]
        else:
            print("ERROR: --train requires a dataset folder path.")
            print("  python main.py --train C:/PlantDataset")
            sys.exit(1)
        train_pipeline(dataset_folder, model_path=model_file)

    if do_chat:
        run_chat(model_path=model_file)

    if not do_train and not do_chat:
        print("Unknown arguments. Use --train or --chat.")
