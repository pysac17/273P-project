# Image Clustering and 3D Reconstruction for Unstructured Image Collections

**Authors:** Sachi Shah, Natasha Jha, Hitha Shri Nagaruru  
**University of California, Irvine, USA**

---

## 📋 Project Overview

This project addresses the challenge of reconstructing 3D scenes from unstructured image collections. Real-world image datasets often contain noisy, unordered, or unrelated photos, making traditional Structure-from-Motion (SfM) pipelines inefficient or prone to errors.

Our approach combines global and local feature extraction, density-based image clustering, and per-cluster 3D reconstruction to robustly estimate camera poses and reconstruct scenes.

### 🔧 Key Components

- **Global Features:** DINOv2 ViT-S/14 embeddings for semantic similarity
- **Clustering:** HDBSCAN for scene separation and noise handling  
- **Local Features:** ALIKED descriptors for geometric matching
- **Feature Matching:** LightGlue for fast and robust correspondences
- **3D Reconstruction:** COLMAP via HLOC

The pipeline efficiently handles large datasets by avoiding exhaustive pairwise matching through semantically meaningful image clusters.

---

## 🚀 Setup Instructions

### 1. Clone Repository
```bash
git clone https://github.com/pysac17/273P-project/
cd 273P-project
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

**Key Dependencies:**
- Python 3.10+
- PyTorch + CUDA (optional for GPU acceleration)
- DINOv2 (global embeddings)
- ALIKED + LightGlue (local features)
- HDBSCAN (density-based clustering)
- FAISS (efficient similarity search)
- COLMAP (via HLOC)
- PIL, NumPy, SciPy, tqdm

### 3. Dataset Preparation
The code was tested on the Kaggle Image Matching Challenge 2025 benchmark. Since full datasets are large, this repository contains sample images for testing pipeline execution.

**Expected Directory Structure:**
```
DATA_DIR/
├── dataset1/
│   ├── img1.jpg
│   ├── img2.jpg
│   └── ...
├── dataset2/
│   └── ...
```

---

## 🎯 How to Train the Model

This pipeline uses pre-trained models rather than requiring traditional training:

### 1. Configure Environment
```python
# Set up paths
DATA_DIR = Path("/path/to/datasets")
OUTPUT_DIR = Path("./output")
FEATURES_DIR = OUTPUT_DIR / "features"
MATCHES_DIR = OUTPUT_DIR / "matches"
SFM_DIR = OUTPUT_DIR / "sfm"
PAIRS_DIR = OUTPUT_DIR / "pairs"

# Configure hyperparameters
TOP_K = 50              # FAISS neighbors per image
RESIZE_MAX = 1600       # Local feature resolution
MAX_PAIRS_PER_DS = 8000 # Cap total pairs per dataset
EXHAUSTIVE_THRESH = 80  # Use exhaustive matching for small clusters
```

### 2. Run the Pipeline
```bash
jupyter notebook ml-project.ipynb
```

Execute all cells sequentially. The pipeline will:

1. **Dataset Discovery & Image Loading**
2. **Global Feature Extraction** (DINOv2)
3. **Image Clustering** (HDBSCAN)
4. **Pair Selection** via FAISS
5. **Local Feature Extraction** (ALIKED)
6. **Feature Matching** (LightGlue)
7. **Per-Cluster SfM Reconstruction** (COLMAP/HLOC)
8. **Final Pose Consolidation** & submission table generation

---

## 📊 How to Evaluate the Model

### Primary Metrics

**Registration Score:** X/Y images successfully registered  
**Clustering Effectiveness:** Number of clusters formed & noise reassignment  
**Reconstruction Completeness:** Visual inspection of generated 3D models

### Performance Results

Based on the experimental evaluation:

| Dataset | Clusters | Registered | Rate |
|----------|-----------|------------|-------|
| ETs | 3 | 19/22 | 86.4% |
| amy_gardens | 4 | 165/200 | 82.5% |
| fbk_vineyard | 2 | 155/163 | 95.1% |
| imc2023_haiper | 2 | 54/54 | 100.0% |
| imc2023_heritage | 6 | 110/209 | 52.6% |
| imc2023_theater_church | 3 | 75/76 | 98.7% |
| imc2024_dioscuri_baalshamin | 5 | 115/138 | 83.3% |
| imc2024_lizard_pond | 3 | 144/214 | 67.3% |
| pt_brandenburg_british_buckingham | 3 | 224/225 | 99.6% |
| pt_piazzasanmarco_grandplace | 2 | 168/168 | 100.0% |
| pt_sacrecoeur_trevi_tajmahal | 3 | 225/225 | 100.0% |
| pt_stpeters_stpauls | 2 | 200/200 | 100.0% |
| stairs | 3 | 38/51 | 74.5% |
| **OVERALL** | - | **1692/1945** | **87.0%** |

### Evaluation Code Example
```python
# Check reconstruction results
import pandas as pd
submission = pd.read_csv('submission.csv')
print(f"Total poses: {len(submission)}")
print(f"Reconstruction rate: {len(submission) / total_images:.2%}")
```

---

## 📁 Expected Outputs

The pipeline generates the following directory structure:

```
output/
├── features/          # ALIKED features per dataset
│   ├── ETs/
│   │   └── feats-aliked-n16.h5
│   └── amy_gardens/
│       └── feats-aliked-n16.h5
├── matches/           # LightGlue matches
│   ├── ETs/
│   │   └── matches-aliked-lightglue.h5
│   └── amy_gardens/
│       └── matches-aliked-lightglue.h5
├── sfm/              # COLMAP reconstructions
│   ├── ETs/
│   │   ├── cluster_0/
│   │   ├── cluster_1/
│   │   └── cluster_2/
│   └── amy_gardens/
│       └── ...
├── pairs/            # Generated image pairs
│   ├── ETs_c0.txt
│   ├── ETs_c1.txt
│   └── ETs_c2.txt
└── submission.csv     # Final poses in competition format
```

**⚠️ Important Note:** Due to large file sizes, the `output/` directory is **not included** in this repository. You need to run the pipeline to generate these files yourself.

### Output File Sizes
- **Features:** ~100-500MB per dataset
- **Matches:** ~50-200MB per dataset  
- **SfM Models:** ~10-100MB per cluster
- **Submission:** ~1-5MB

### Submission File Format
```csv
dataset,image,scene,rotation_matrix,translation_vector
ETs,et_et000.png,scene_2,"r11;r12;r13;r21;r22;r23;r31;r32;r33","t1;t2;t3"
```

---

## 🔄 How to Reproduce Results

### Step-by-Step Reproduction

1. **Prepare datasets** under `DATA_DIR`
2. **Run** `ml-project.ipynb` notebook
3. **Monitor** feature extraction, clustering, and per-cluster SfM progress
4. **Wait** for completion (approximately 115 minutes for full dataset)

### Expected Processing Time
- **Sample Dataset:** ~2-3 minutes
- **Full Competition Dataset:** ~115 minutes
- **Memory Usage:** 8-16GB depending on dataset size

### Key Observations from Experiments

**✅ What Worked Well:**
- Perfect reconstruction (100%) on 4 datasets: imc2023_haiper, pt_piazzasanmarco_grandplace, pt_sacrecoeur_trevi_tajmahal, pt_stpeters_stpauls
- Strong performance (>95%) on large-scale datasets like pt_brandenburg_british_buckingham (224/225)
- Consistent clustering producing 2-5 meaningful clusters per dataset

**⚠️ Failure Cases:**
- Lower performance on datasets with repetitive patterns: imc2023_heritage (52.6%), imc2024_lizard_pond (67.3%)
- Performance degrades with limited image overlap and ambiguous viewpoints

---

## 🎯 Project Highlights

### 🏆 Key Strengths
- **Efficient Hierarchical Matching:** Combines global semantic embeddings and local geometric features
- **Robust Clustering:** Handles noisy images and multi-scene datasets automatically
- **Scalable SfM:** Matches only relevant image pairs per cluster to reduce runtime
- **Hybrid Feature Pipeline:** DINOv2 + ALIKED + LightGlue achieves high registration rates
- **Code Reusability:** Modular functions for global/local features, clustering, pair generation, and pose extraction

### 💡 Innovation
- **Noise-aware clustering** with automatic reassignment
- **Cross-cluster bridge pairs** for connectivity
- **Adaptive parameter selection** based on dataset size
- **Multi-format pose extraction** supporting different COLMAP versions

---

## 📦 Sample Data & Demo

### Sample Dataset
`sample_data/` — Small set of images to test pipeline execution without downloading full datasets

### Demo Notebook
`ml-project.ipynb` — Complete notebook to run the pipeline and inspect intermediate outputs

**Note:** Full pipeline may take multiple hours depending on dataset size and GPU availability.

---

## 🔧 Technical Implementation

### Core Functions
- `extract_global_features()`: DINOv2 embedding extraction with L2 normalization
- `cluster_images()`: HDBSCAN clustering with noise reassignment
- `generate_pairs()`: Intelligent pair generation (FAISS + exhaustive)
- `extract_all_poses()`: Multi-format COLMAP pose extraction

### Error Handling
- Automatic skipping of corrupted images
- Multiple fallback methods for pose extraction
- Memory management with cache clearing
- Resume capability to avoid redundant computations

---

## 📚 References

- **DINOv2:** https://arxiv.org/abs/2304.07193
- **ALIKED:** https://github.com/shuaizhao/aliked
- **LightGlue:** https://arxiv.org/abs/2305.09682
- **COLMAP / HLOC:** https://github.com/cvg/Hierarchical-Localization
- **Kaggle Image Matching Challenge 2025:** https://www.kaggle.com/competitions/image-matching-challenge-2025

---

## 📈 Performance Summary

**Overall Achievement:** 87.0% reconstruction rate across 13 diverse datasets  
**Total Processing Time:** 115 minutes for complete benchmark  
**Perfect Reconstructions:** 4 out of 13 datasets achieved 100% registration  
**Scalability:** Successfully processed datasets ranging from 22 to 225 images  

This demonstrates the pipeline's robustness across varying scene types, scales, and complexity levels.
