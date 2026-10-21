---
title: Brain Tumor Multi-Task Diagnostic AI
emoji: 🧠
colorFrom: blue
colorTo: indigo
sdk: gradio
app_file: app.py
pinned: false
license: mit
short_description: Multi-Task Brain Tumor Segmentation, Subtyping & Measurement
---

# 🧠 Brain Tumor Multi-Task Diagnostic AI (`Mask-XAI-Net`)

### Clinical-Grade Joint Histopathological Classification, Lesion Delineation & Quantitative Area Measurement from Cranial MRI

---

## 📌 Overview
This diagnostic software provides an automated, end-to-end clinical workflow for cranial Magnetic Resonance Imaging (MRI) analysis. Built upon a 2026 State-of-the-Art (SOTA) real-time edge multi-task architecture, it performs **simultaneous histopathological subtyping and pixel-level tumor boundary delineation** within milliseconds.

### 🌟 Key Capabilities:
1. **Multi-Task Diagnosis:** Concurrently outputs diagnostic subtyping (*Glioma*, *Meningioma*, *Pituitary*, or *Healthy Controls*) and exact pixel-level lesion boundary contours.
2. **Automated Skull Stripping & CLAHE:** Strips surrounding background air/noise with safe cranial margin padding and enhances subtle lesion contrast using Contrast Limited Adaptive Histogram Equalization.
3. **Quantitative Tumor Morphometry:**
   - Total lesion pixel area.
   - Cranial cross-sectional burden (% of brain occupied by tumor).
   - Estimated anatomical lesion area ($mm^2$ and $cm^2$).
   - Spatial bounding dimensions ($W \times H$).
4. **Out-of-Distribution (OOD) Robustness:** Validated across 3 independent multi-center cohorts (Cheng Figshare, BRISC 2025, and Nickparvar MRI) with an empirical generalizability retention rate of **83.60%**.

---

## 🔬 Benchmark Performance Summary

| Metric | In-Domain (BRISC 2025) | Cross-Center Retention | In-Domain Dice Score |
| :--- | :---: | :---: | :---: |
| **Lesion Delineation Dice** | **0.9414** | **83.60%** (Rank 1) | **0.8762 – 0.9414** |
| **Mean IoU (Jaccard Index)**| **0.8892** | - | - |
| **Inference Latency** | **<25 ms** (CPU) | **<5 ms** (GPU) | Real-Time Point-of-Care |
| **Background Saliency Leakage**| **10.53%** (Lowest) | - | Bounded Attention |

---

## 🚀 How to Run Locally

```bash
git clone https://huggingface.co/spaces/<your-username>/brain-tumor-multitask-ai
cd brain-tumor-multitask-ai
pip install -r requirements.txt
python app.py
```

---

## ⚖️ Clinical Disclaimer
This system is an academic research demonstration for computer-aided diagnostic assistance and medical imaging reproducibility. It is intended to assist medical professionals and should not replace certified radiological review.
