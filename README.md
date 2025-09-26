# Facial Expression Recognition

This repository contains  **categorical facial expression recognition** and **continuous valence–arousal regression** using transfer learning with CNN baselines.

---

## Task Description
- **Dataset**: Provided face images with 8 categorical expression labels and continuous values for **Valence** and **Arousal**.  
- **Objectives**:
  1. Classify facial expressions into 8 categories (Neutral, Happy, Sad, Surprise, Fear, Disgust, Anger, Contempt).
  2. Predict continuous **Valence** (positive/negative) and **Arousal** (active/calm) values.  
- **Approach**: Use transfer learning with at least two CNN baselines and compare their performance.  

---

## Network Architectures
We used two pre-trained CNN baselines:
1. **MobileNetV2** (pre-trained on ImageNet)  
2. **DenseNet121** (pre-trained on ImageNet)  

**Model Adaptation**:
- Convolutional base frozen (non-trainable).  
- Added layers:
  - `GlobalAveragePooling2D`  
  - `Dense(256, ReLU)` + `Dropout(0.5)`  
  - Output layer:
    - Classification → `Dense(8, softmax)`  
    - Regression → `Dense(2, linear)`  

**Training Settings**:
- Optimizer: Adam (`lr=1e-4`)  
- Batch size: 16  
- Epochs: 10  
- Loss:
  - Classification → Sparse Categorical Crossentropy  
  - Regression → Mean Squared Error (MSE)  

---

## Performance Metrics

### Classification
- Accuracy  
- F1-Score  
- Cohen’s Kappa  
- Krippendorff’s Alpha  
- AUC-ROC, AUC-PR  

### Regression (Continuous Domain)
- RMSE  
- Correlation (CORR)  
- Sign Agreement (SAGR)  
- Concordance Correlation Coefficient (CCC)  

---

