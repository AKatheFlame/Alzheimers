# Explainable Hybrid Vision-Swin Transformer for Alzheimer's Disease Classification

**Authors:** Akarsh Katiyar & Deepak Kumar Yadav  
**Institution:** Sharda University 

This repository contains the official implementation for the paper: *"An Explainable Hybrid Vision-Swin Transformer Framework for Alzheimer's Disease Classification using Automated Skull-Stripping and MRI dataset."*

## 📖 Overview
Alzheimer's Disease (AD) and its transitional stage, Mild Cognitive Impairment (MCI), are challenging to diagnose from structural MRIs due to subtle neuroanatomical changes and severe clinical class imbalances. 

This project introduces a unified, end-to-end deep learning pipeline that prioritizes **clinical realism, computational efficiency, and diagnostic transparency**. By combining deterministic computer vision preprocessing, a dual-stream transformer architecture, and advanced Explainable AI (XAI), this framework bridges the trust gap between AI models and medical professionals.

### ✨ Key Features
1. **Automated Morphological Skull-Stripping:** A highly efficient OpenCV-based pipeline using Otsu's thresholding and morphological operations to perfectly isolate brain parenchyma without the massive computational overhead of deep learning segmenters (like U-Net or BET).
2. **Dual-Branch Transformer Architecture:** Simultaneously captures macro-level cortical asymmetry (Vision Transformer) and micro-structural texture anomalies (Swin Transformer).
3. **Multi-Head Attention Fusion (MHAF):** A dynamic gated fusion mechanism that intelligently combines global and local feature vectors.
4. **Clinical Imbalance Handling:** Implements Weighted Cross-Entropy Loss to heavily penalize misclassifications of the minority healthy class, ensuring a "fail-safe" clinical triage model.
5. **Integrated Explainability:** Uses Grad-CAM to generate anatomically grounded heatmaps, providing visual proof of the model's decision-making process (highlighting the medial temporal lobe).

---

## 📊 Dataset
The model was trained and evaluated on raw, uncompressed **DICOM structural MRI (sMRI) scans** sourced from Firoozgar Hospital via the Mendeley Data repository.

* **Total Scans:** 478
* **Classes:**
  * Mild Cognitive Impairment (MCI): 205 scans (~43%)
  * Alzheimer's Disease (AD): 199 scans (~42%)
  * Normal Cognition (NC): 74 scans (~15%)

*Note: The dataset exhibits a natural, severe clinical class imbalance. The model strictly evaluates on unaugmented, real-world data splits to prevent data leakage.*

---

## 📁 Source Code Structure

```text
├── batch_skull_strip.py              # Script to preprocess the entire dataset
├── best_hybrid_model.pth             # Trained model weights (Generated after training)
├── confusion_matrix.png              # Auto-generated evaluation confusion matrix
├── gradcam_output.png                # Sample Grad-CAM++ explainability heatmap
├── live_demo.py                      # Script to run inference on a single MRI scan
├── requirements.txt                  # Stating all the required dependencies for the code
├── skull_stripping_vis.py            # Script to visualize skull-stripping on a single image
├── train_hybrid_model.py             # Main training script (Dual-transformer architecture)
└── training_curves_clean.png         # Auto-generated loss and accuracy graphs

---

## ⚙️ Prerequisites & Installation

**Environment:** Python 3.8+ and PyTorch (CUDA recommended).

Install the required dependencies:
```bash
pip install -r requirements.txt
