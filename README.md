# 🧠 Mask-XAI-Net: Clinical-Grade Brain Tumor Multi-Task Diagnostic AI

<div align="center">

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Live%20Diagnostic%20Demo-blue?style=for-the-badge)](https://huggingface.co/spaces/mahfuj735/Brain-Tumor-Diagnostic-AI)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-brightgreen.svg?style=for-the-badge&logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg?style=for-the-badge&logo=pytorch)](https://pytorch.org/)
[![Dice Score](https://img.shields.io/badge/Dice%20Score-0.9414-success.svg?style=for-the-badge)](https://github.com/mahfuj735/Brain_Tumor_Diagonistic_AI)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)

**A Mask-Supervised, Attention-Aligned, and Ultra-Lightweight Deep Learning Framework for Clinically Faithful Brain Tumor MRI Classification, Lesion Delineation & Quantitative Area Morphometry Across Multi-Center Cohorts**

*Developed at AIR Lab (AI Innovation & Research Lab), Department of Computer Science & Engineering*  
*Target Publication: IEEE Journal of Biomedical and Health Informatics (J-BHI) / Computers in Biology and Medicine (Elsevier)*

</div>

---

## 📌 Executive Summary

Accurate diagnosis of primary brain tumors from cranial Magnetic Resonance Imaging (MRI) is essential for neurosurgical trajectory planning, stereotactic biopsy, and oncology prognostication. Standard computer-aided diagnostic (CAD) systems frequently suffer from **spatial shortcut learning** (relying on skull geometry or background acquisition noise rather than authentic neoplastic tissue) and **domain fragility** across disparate scanner vendors.

**`Mask-XAI-Net`** overcomes these clinical hurdles through an automated, end-to-end multi-task architecture that simultaneously performs:
1. **Histopathological Subtyping:** 4-class classification (*Glioma*, *Meningioma*, *Pituitary Adenoma*, or *Healthy Normal Controls*).
2. **Pixel-Level Lesion Delineation:** High-fidelity tumor boundary segmentation ($Dice = 0.9414$, $mIoU = 0.8892$).
3. **Quantitative Tumor Morphometry:** Real-time computation of intracranial lesion area ($mm^2, cm^2$) and volumetric cross-sectional intracranial burden.
4. **Faithful Explainability (Quantitative XAI):** Eliminates cherry-picked qualitative saliency maps by providing mathematically rigorous spatial attention metrics ($\text{mIoU}_{\text{CAM}}$, $\text{Dice}_{\text{CAM}}$, and Pointing Game Hit Rate).

---

## 🚀 Live Interactive Diagnostic Web App

A clinical-grade, zero-setup interactive demonstration is hosted live on **Hugging Face Spaces**:

👉 **[Launch Live Diagnostic Software Demo](https://huggingface.co/spaces/mahfuj735/Brain-Tumor-Diagnostic-AI)**

```
             ┌─────────────────────────┐
             │ Input Cranial MRI Slice │
             └────────────┬────────────┘
                          │
             ┌────────────▼────────────┐
             │ Anatomical Sanity & OOD │
             │  Cranial Guardrail QA   │
             └────────────┬────────────┘
                          │ (Verified)
             ┌────────────▼────────────┐
             │ Automated Skull Strip   │
             │   & 5% Margin CLAHE     │
             └────────────┬────────────┘
                          │
        ┌─────────────────┴─────────────────┐
        │                                   │
┌───────▼────────────────┐         ┌────────▼────────────────┐
│   Histopathological    │         │  Pixel-Level Boundary   │
│   Tumor Classification │         │  Contour Segmentation   │
│ (Glioma/Mening/Pit/Norm)│        │     (Dice = 0.9414)     │
└───────┬────────────────┘         └────────┬────────────────┘
        │                                   │
        └─────────────────┬─────────────────┘
                          │
             ┌────────────▼────────────┐
             │  Morphometric Analytics │
             │  • Estimated Area (cm²) │
             │  • % Cranial Burden     │
             │  • Single Primary Focus │
             └─────────────────────────┘
```

---

## 🔬 Key Scientific Innovations

| Research Gap in Prior Literature | Our Solution & Novelty (`Mask-XAI-Net`) |
| :--- | :--- |
| **Cherry-Picked Saliency (Qualitative XAI):** Prior studies showcase 3–5 handpicked Grad-CAM images without mathematical validation. | **Quantitative XAI Metric Suite:** Exhaustive audit across every test slice measuring $\text{mIoU}_{\text{CAM}}$, $\text{Dice}_{\text{CAM}}$, and Pointing Game Hit Rate against radiologist-delineated masks. |
| **Spatial Shortcut Learning:** Models trained solely on cross-entropy learn background acquisition artifacts and cranial bone contours. | **Mask-Supervised Alignment Loss ($\mathcal{L}_{\text{Mask}}$):** Constrains backward gradient flow directly to true pathological tissue boundaries. |
| **Lack of Normal Controls & Private Cohorts:** Over-reliance on single-center or unreleased private datasets lacking healthy control MRIs. | **Three-Tier Multi-Center Public Benchmark (16,086 MRIs):** Multi-cohort validation across BRISC 2025, Cheng Figshare, and Nickparvar MRI. |
| **Multi-Scanner Fragility:** Performance degrades drastically when evaluated on unseen scanner hardware and field strengths. | **$3 \times 3$ Cross-Domain Generalization Matrix:** Demonstrates state-of-the-art generalizability with an **83.60%** retention rate. |
| **Theoretical Edge Claims:** Complex heavy networks unusable on point-of-care mobile or embedded hardware. | **Edge-Deployable Ultra-Lightweight Backbone:** Parameter footprint of only **0.58M**, model size of **1.8 MB**, and latency **<25 ms on standard CPU**. |

---

## 📊 Experimental Results & Benchmark Performance

### 1. In-Domain & Cross-Domain Delineation Metrics
Evaluated rigorously across independent multi-center MRI cohorts:

| Metric | In-Domain (BRISC 2025) | Cross-Center Retention | Cheng Figshare | Nickparvar MRI |
| :--- | :---: | :---: | :---: | :---: |
| **Lesion Dice Score** | **0.9414** | **83.60% (Rank 1)** | 0.8924 | 0.9150 |
| **Mean IoU (Jaccard Index)** | **0.8892** | - | 0.8241 | 0.8522 |
| **Precision** | **0.9520** | - | 0.9110 | 0.9340 |
| **Recall (Sensitivity)** | **0.9312** | - | 0.8750 | 0.8970 |
| **Background Saliency Leakage**| **10.53% (Lowest)** | - | Bounded Attention | Bounded Attention |
| **Inference Latency (CPU)** | **<25 ms** | Real-time POC | - | - |
| **Inference Latency (GPU)** | **<5 ms** | Real-time POC | - | - |

---

## 🗂️ Repository Organization

```text
Brain_Tumor_Diagonistic_AI/
├── software/                            # Diagnostic software application suite
│   ├── app.py                           # Gradio multi-task clinical inference application
│   ├── best_model.pt                    # Pretrained lightweight weights
│   ├── requirements.txt                 # Software-specific dependencies
│   ├── run_locally.bat                  # One-click Windows local launcher
│   ├── deploy_to_huggingface.py         # Automated deployment script for HF Spaces
│   └── examples/                        # Clinical sample scans for instant testing
│       ├── sample_glioma.png
│       ├── sample_meningioma.png
│       ├── sample_pituitary.png
│       └── sample_healthy_normal.png
├── scripts/                             # Reproducible pipeline & evaluation scripts
│   ├── dataset.py                       # Universal multi-cohort dataset loader & transforms
│   ├── sanity_check.py                  # Full integrity & zero-leakage mathematical audit
│   ├── evaluate_deeplabv3plus_cross_domain.py # Multi-center cross-domain evaluator
│   ├── split_cheng_dataset.py           # Patient-level leakage-free split generator
│   ├── split_cheng_figshare.py          # Cross-validation fold builder
│   ├── convert_brisc_to_png.py          # Standardized preprocessor for BRISC 2025
│   ├── convert_nickparvar_to_png.py     # Standardized preprocessor for Nickparvar cohort
│   └── verify_preprocessing.py          # Skull-stripping & CLAHE QA auditor
├── notebooks/                           # Comprehensive experimental notebooks
│   ├── multi_model_benchmark_brisc2025.ipynb      # SOTA architecture comparison on BRISC
│   ├── multi_model_benchmark_cheng_figshare.ipynb # SOTA architecture comparison on Figshare
│   ├── cross_domain_multi_model_benchmark.ipynb   # 3x3 multi-center cross-scanner validation
│   ├── quantitative_xai_fidelity_benchmark.ipynb  # Mathematical explainability auditor
│   ├── ablation_study_framework.ipynb             # Loss & architecture ablation analyses
│   ├── cheng_figshare_pipeline.ipynb              # Baseline end-to-end pipeline
│   └── nickparvar_ensemble_pseudolabeling.ipynb   # Semi-supervised pseudo-labeling pipeline
├── LICENSE                              # MIT Open-Source License
├── .gitignore                           # Excludes large datasets and training checkpoints
└── README.md                            # Primary documentation (this file)
```

---

## 💻 Running the Diagnostic Software Locally

### Step 1: Clone the Repository
```bash
git clone https://github.com/mahfuj735/Brain_Tumor_Diagonistic_AI.git
cd Brain_Tumor_Diagonistic_AI/software
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Launch the Diagnostic Web App
```bash
python app.py
```
Or on Windows, simply double-click **`run_locally.bat`**.  
The interactive interface will immediately launch at `http://127.0.0.1:7860`.

---

## 🔬 Reproducing Research Experiments

### 1. Data Integrity and Leakage Sanity Audit
To verify zero-leakage across patient folds, consistent binary mask bounds, and file checksums:
```bash
python scripts/sanity_check.py
```

### 2. Multi-Model Benchmarks
Open and run the notebooks in the `notebooks/` directory to reproduce all tables and figures presented in the manuscript:
* **`notebooks/cross_domain_multi_model_benchmark.ipynb`**: Re-computes the $3 \times 3$ cross-domain transferability matrix.
* **`notebooks/quantitative_xai_fidelity_benchmark.ipynb`**: Generates IoU and Dice distributions for CAM saliency fidelity.
* **`notebooks/ablation_study_framework.ipynb`**: Evaluates individual contributions of $\mathcal{L}_{\text{Mask}}$, CLAHE, and safety-margin parenchyma isolation.

---

## 📖 Citation

If you use this software, code, or methodology in your research, please cite our manuscript:

```bibtex
@article{mahfuj2026maskxainet,
  title={Mask-XAI-Net: A Mask-Supervised, Attention-Aligned, and Ultra-Lightweight Deep Learning Framework for Clinically Faithful Brain Tumor MRI Classification Across Multi-Center Benchmarks},
  author={Mahfuj, et al.},
  journal={IEEE Journal of Biomedical and Health Informatics (Under Review)},
  year={2026},
  publisher={IEEE}
}
```

---

## ⚖️ Clinical Disclaimer

This software and research artifacts are developed for academic research, peer reproducibility, and computer-aided diagnostic research. It is intended to assist qualified medical practitioners and should not be used as the sole basis for clinical diagnosis or treatment decisions without certified radiological oversight.

---

<div align="center">
<sub>AIR Lab • Open-Source Medical AI • Designed for Clinical Reproducibility</sub>
</div>