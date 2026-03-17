Image Clustering and 3D Reconstruction for Unstructured Image Collections
Authors:
Sachi Shah, Natasha Jha, Hitha Shri Nagaruru
University of California, Irvine, USA
Project Overview
This project addresses the problem of reconstructing 3D scenes from unstructured image collections. Real-world image datasets often contain noisy, unordered, or unrelated photos, making traditional Structure-from-Motion (SfM) pipelines inefficient or prone to errors.
Our approach combines global and local feature extraction, density-based image clustering, and per-cluster 3D reconstruction to robustly estimate camera poses and reconstruct scenes.
Key components:
Global Features: DINOv2 ViT-S/14 embeddings for semantic similarity
Clustering: HDBSCAN for scene separation and noise handling
Local Features: ALIKED descriptors for geometric matching
Feature Matching: LightGlue for fast and robust correspondences
3D Reconstruction: COLMAP via HLOC
The pipeline is capable of handling large datasets efficiently, avoiding exhaustive pairwise matching by focusing on semantically meaningful image clusters.
Setup Instructions
1. Clone the Repository
git clone https://github.com/<your-username>/image-clustering-sfm.git
cd image-clustering-sfm
2. Install Dependencies
Use conda or pip to install required packages:
pip install -r requirements.txt
Key dependencies:
Python 3.10+
PyTorch + CUDA (optional for GPU)
DINOv2 (global embeddings)
ALIKED + LightGlue (local features)
HDBSCAN
FAISS
COLMAP (via HLOC)
PIL, NumPy, SciPy, tqdm
3. Dataset
The code was tested on the Kaggle Image Matching Challenge 2025 benchmark.
Since full datasets are large and not included here, the repository contains sample images for testing pipeline execution.
Directory structure:
DATA_DIR/
├── dataset1/
│   ├── img1.jpg
│   ├── img2.jpg
│   └── ...
├── dataset2/
│   └── ...
Running the Pipeline
Set the paths:
DATA_DIR = Path("/path/to/datasets")
OUTPUT_DIR = Path("./output")
FEATURES_DIR = OUTPUT_DIR / "features"
MATCHES_DIR = OUTPUT_DIR / "matches"
SFM_DIR = OUTPUT_DIR / "sfm"
PAIRS_DIR = OUTPUT_DIR / "pairs"
Configure hyperparameters (optional):
TOP_K = 50              # FAISS neighbors
RESIZE_MAX = 1600       # local feature resolution
MAX_PAIRS_PER_DS = 8000 # cap total pairs per dataset
EXHAUSTIVE_THRESH = 80  # use exhaustive matching if cluster ≤ this size
Run the pipeline:
jupyter notebook
Open the run_pipeline.ipynb notebook and execute all cells sequentially.
Pipeline Steps:
Dataset discovery & image loading
Global feature extraction (DINOv2)
Image clustering (HDBSCAN)
Pair selection via FAISS
Local feature extraction (ALIKED)
Feature matching (LightGlue)
Per-cluster SfM reconstruction (COLMAP/HLOC)
Final pose consolidation & submission table generation
Evaluation Metrics
Registration Score: X/Y images successfully registered
Clustering Effectiveness: Number of clusters formed & noise reassignment
Reconstruction Completeness: Visual inspection of generated 3D models
Example from logs:
Dataset	Clusters	Registered
ETs	3	19/22
amy_gardens	4	165/200
fbk_vineyard	2	155/163
Project Highlights
Efficient hierarchical matching: Combines global semantic embeddings and local geometric features
Robust clustering: Handles noisy images and multi-scene datasets
Scalable SfM: Matches only relevant image pairs per cluster to reduce runtime
Hybrid feature pipeline: DINOv2 + ALIKED + LightGlue achieves high registration rates
Code reusability: Modular functions for global/local features, clustering, pair generation, and pose extraction
How to Reproduce Results
Prepare datasets under DATA_DIR
Run the run_pipeline.ipynb notebook
Wait for feature extraction, clustering, and per-cluster SfM to complete
Output will include:
features/
matches/
sfm/ (3D reconstructions per cluster)
pairs/ (image pairs used for matching)
submission.csv (poses for evaluation)
Note: Full pipeline may take multiple hours depending on dataset size and GPU availability.
Sample Data & Notebook
sample_data/ — Small set of images to test pipeline execution
run_pipeline.ipynb — Notebook to run the pipeline and inspect intermediate outputs
References
DINOv2: https://arxiv.org/abs/2304.07193
ALIKED: https://github.com/shuaizhao/aliked
LightGlue: https://arxiv.org/abs/2305.09682
COLMAP / HLOC: https://github.com/cvg/Hierarchical-Localization
Kaggle Image Matching Challenge 2025: https://www.kaggle.com/competitions/image-matching-challenge-2025
