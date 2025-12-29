# Flower Species Classification - Technical Assessment

## Overview
This project implements an end-to-end Computer Vision pipeline to classify 102 flower species. It addresses the challenge of fine-grained classification where inter-class similarity is high.

**Model:** EfficientNet-B0 (Transfer Learning)
**Accuracy:** ~97% (Test Set)
**F1-Score:** 0.97 (Weighted)

## 🛠️ Approach & Methodology

### 1. Data Engineering
- **Imbalance Handling:** Utilized `WeightedRandomSampler` to address the variance in class counts (40 to 258 images).
- **Augmentation:** Applied RandomRotation and HorizontalFlip to simulate field conditions.
- **Analysis:** EDA revealed high intra-class variation, justifying the need for a pre-trained backbone.

### 2. Model Development
- Selected **EfficientNet-B0** over ResNet50 for its parameter efficiency (4M vs 25M params) and comparable accuracy.
- Employed **Transfer Learning** with a fine-tuning strategy (frozen backbone -> unfrozen fine-tuning).
- Implemented **Early Stopping** to prevent overfitting.

### 3. Explainability
- Used **Grad-CAM** to verify the model focuses on relevant features (flower center/pistil) rather than background noise.
- **Insight:** The model correctly ignores background foliage, indicating robust generalization.

## 🚀 How to Run
1. Open `notebooks/02_Training.ipynb` in Google Colab (Enable GPU).
2. Run all cells to download data and train.

## 🔮 Future Work (If I had more time)
- **Test Time Augmentation (TTA):** To further improve confidence on ambiguous samples.
- **Deployment:** Wrap the inference function in a FastAPI container for real-time serving.
- **Hyperparameter Search:** Use Optuna to fine-tune the learning rate and weight decay dynamically.
