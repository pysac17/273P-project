# Image Matching Challenge 2025 - Structure from Motion Pipeline

A comprehensive Structure from Motion (SfM) pipeline for the Image Matching Challenge 2025, leveraging DINOv2 global features, HDBSCAN clustering, and COLMAP reconstruction for robust camera pose estimation.

## Project Overview

This project implements an end-to-end pipeline for estimating camera poses (rotation matrices and translation vectors) from unordered image collections. The solution combines modern deep learning approaches with traditional computer vision techniques to achieve high reconstruction rates across diverse datasets.

The pipeline processes multiple datasets, automatically clusters images into coherent scenes, generates optimal image pairs, and performs 3D reconstruction to estimate camera poses for each image.

## Setup Instructions

### Prerequisites

- Python 3.8 or higher
- PyTorch with MPS/CUDA support recommended
- 16GB+ RAM for large datasets
- 50GB+ storage for features and intermediate results

### Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd image-matching-challenge-2025
```

2. Create and activate virtual environment:
```bash
python -m venv ml-env
source ml-env/bin/activate  # On Windows: ml-env\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. For Mac users, ensure proper environment setup:
```python
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
```

## Required Dependencies

### Core Dependencies
- **torch**: PyTorch for deep learning operations
- **torchvision**: Computer vision utilities
- **numpy**: Numerical computations
- **pandas**: Data manipulation and CSV handling
- **pillow**: Image processing
- **tqdm**: Progress bars
- **scipy**: Scientific computing
- **scikit-learn**: Machine learning utilities

### Computer Vision
- **hdbscan**: Density-based clustering
- **pycolmap**: Structure from Motion reconstruction
- **hloc**: Hierarchical localization framework
- **transformers**: Alternative DINOv2 loading

## Dataset

### Full Dataset
The complete Image Matching Challenge 2025 dataset can be downloaded from:
[Competition Website](https://www.kaggle.com/c/image-matching-challenge-2025)

### Sample Dataset
A small sample dataset is provided in `sample_data/ETs_sample/` containing 9 images from the ETs dataset. This sample is sufficient for testing the pipeline functionality.

### Data Organization
The pipeline expects the following directory structure:

```
image-matching-challenge-2025/
├── test/
│   ├── dataset1/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   ├── dataset2/
│   └── ...
├── train/
│   ├── dataset1/
│   └── ...
├── sample_submission.csv
└── train_labels.csv
```

## How to Train the Model

This pipeline doesn't require traditional training. The "model" consists of pre-trained neural networks (DINOv2, ALiED) used for feature extraction. To run the complete pipeline:

1. Open the main notebook:
```bash
jupyter notebook ml-project.ipynb
```

2. Execute cells sequentially. The pipeline will:
   - Load pre-trained DINOv2 model
   - Process each dataset
   - Extract global and local features
   - Generate image pairs
   - Perform 3D reconstruction

3. Monitor progress through the verbose output showing reconstruction statistics.

## How to Evaluate the Model

The pipeline automatically evaluates reconstruction quality by reporting:

- **Reconstruction Rate**: Percentage of images with valid poses
- **Scene Clustering**: Number of coherent scenes detected
- **Registration Statistics**: COLMAP reconstruction metrics

To manually evaluate results:

```python
# Check reconstruction results
import pandas as pd
submission = pd.read_csv('submission.csv')
print(f"Total poses: {len(submission)}")
print(f"Reconstruction rate: {len(submission) / total_images:.2%}")
```

## Expected Outputs

The pipeline generates several outputs:

### Directory Structure
```
output/
├── features/          # ALiED features per dataset
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
└── pairs/            # Generated image pairs
    ├── ETs_c0.txt
    ├── ETs_c1.txt
    └── ETs_c2.txt
```

### Submission File
`submission.csv` in competition format:
```csv
dataset,image,scene,rotation_matrix,translation_vector
ETs,et_et000.png,scene_0,"r11;r12;r13;r21;r22;r23;r31;r32;r33","t1;t2;t3"
ETs,et_et001.png,scene_0,"r11;r12;r13;r21;r22;r23;r31;r32;r33","t1;t2;t3"
...
```

## Data Preprocessing

### Automatic Preprocessing
The pipeline handles preprocessing automatically:

1. **Image Discovery**: Recursive search for valid image formats
2. **EXIF Correction**: Automatic rotation based on EXIF data
3. **Resizing**: Images resized to 224x224 for DINOv2, 1024px max for ALiED
4. **Normalization**: L2 normalization for feature vectors

### Manual Preprocessing (Optional)
For custom preprocessing, modify these functions:
- `fix_rotation()`: EXIF-based rotation correction
- `extract_global_features()`: DINOv2 preprocessing
- Feature extraction configuration in hloc calls

## How to Reproduce Results

### Using Sample Data
1. Use the provided sample dataset in `sample_data/ETs_sample/`
2. Run the pipeline with reduced dataset size
3. Expected results: ~85% reconstruction rate on 9 images

### Using Full Dataset
1. Download complete competition dataset
2. Place in `image-matching-challenge-2025/` directory
3. Run complete pipeline (8+ hours)
4. Expected results: ~85% average reconstruction rate across all datasets

### Key Parameters
Modify these parameters in the notebook:
```python
TOP_K = 50              # Nearest neighbors for pair generation
RESIZE_MAX = 1024       # Local feature resolution
MAX_PAIRS_PER_DS = 8000 # Maximum pairs per dataset
EXHAUSTIVE_THRESH = 80  # Use exhaustive matching for small clusters
TIME_BUDGET = 8.0 * 3600 # Total processing time budget
```

## Demo Notebook

The main notebook `ml-project.ipynb` serves as a comprehensive demo that allows you to:

### Run the Code
- Execute cells sequentially to run the complete pipeline
- Monitor progress through detailed logging
- Pause and resume at any stage

### Test the Model
- Test on sample data with reduced computation time
- Verify feature extraction quality
- Validate clustering and reconstruction steps

### Reproduce Key Results
- Reproduce competition-level performance
- Benchmark different parameter settings
- Analyze reconstruction statistics

### Key Demo Sections
1. **Setup and Configuration**: Environment setup and hyperparameters
2. **Feature Extraction**: DINOv2 global and ALiED local features
3. **Clustering**: HDBSCAN scene segmentation
4. **Pair Generation**: Intelligent image pair selection
5. **Reconstruction**: COLMAP 3D reconstruction
6. **Results Analysis**: Performance metrics and statistics

## Performance Benchmarks

### Sample Dataset (ETs, 9 images)
- **Processing Time**: ~2 minutes
- **Reconstruction Rate**: ~85%
- **Memory Usage**: ~2GB
- **Scenes Detected**: 2-3

### Full Competition Dataset (13 datasets)
- **Processing Time**: ~8 hours
- **Average Reconstruction Rate**: ~85%
- **Memory Usage**: 8-16GB
- **Total Images**: 2000+

## Troubleshooting

### Common Issues

1. **Kernel Crashes on Mac**
   - Ensure all thread environment variables are set
   - Monitor memory usage with `torch.mps.empty_cache()`
   - Use reduced batch sizes for large datasets

2. **Low Reconstruction Rates**
   - Check image quality and variety
   - Adjust HDBSCAN parameters (min_cluster_size, min_samples)
   - Verify pair generation counts

3. **COLMAP Failures**
   - Ensure sufficient image pairs per cluster (minimum 3)
   - Check feature extraction quality
   - Verify image EXIF data

4. **Memory Issues**
   - Reduce RESIZE_MAX parameter
   - Limit concurrent operations
   - Clear cache between datasets

### Debug Mode
Enable verbose logging by commenting out:
```python
# warnings.filterwarnings("ignore")  # Comment out for debug mode
```

## Architecture Details

### Pipeline Components

1. **Global Feature Extraction**
   - DINOv2 ViT-S/14 for 384-dimensional embeddings
   - Image preprocessing with EXIF rotation correction
   - L2 normalization for cosine similarity

2. **Scene Clustering**
   - HDBSCAN with adaptive parameters based on dataset size
   - Noise point reassignment to nearest cluster centroids
   - Automatic fallback to single cluster for small datasets

3. **Pair Generation**
   - Exhaustive matching for small clusters (≤80 images)
   - Top-K similarity search for larger clusters using PyTorch
   - Cross-cluster bridge pairs for connectivity

4. **Local Feature Processing**
   - ALiED-N16 feature extraction
   - LightGlue matching for robust correspondences
   - Configurable image resolution (default: 1024px)

5. **Structure from Motion**
   - Per-cluster COLMAP reconstruction
   - Multi-format pose extraction (binary/text)
   - Comprehensive error handling and fallbacks

## Key Functions

### Core Utilities
- `extract_global_features()`: DINOv2 embedding extraction
- `cluster_images()`: HDBSCAN clustering with noise handling
- `generate_pairs()`: Intelligent pair generation
- `read_poses_from_colmap()`: Multi-format pose extraction
- `extract_all_poses()`: Comprehensive pose reading

### Error Handling
The pipeline includes robust error handling for:
- Corrupted images (automatic skipping)
- COLMAP reconstruction failures
- Memory management (automatic cleanup)
- Multiple pycolmap API versions

## Optimization Features

### Mac-Specific Optimizations
- Thread limiting to prevent kernel crashes
- MPS backend utilization for Apple Silicon
- Memory cleanup after each dataset
- SSL verification handling for model downloads

### Resume Capability
- Automatic detection of existing features/matches
- Skip completed clusters in SfM reconstruction
- Incremental pose accumulation
- Time budget management

## License

This project is developed for the Image Matching Challenge 2025. Please refer to the competition guidelines for usage restrictions.

## Acknowledgments

- Facebook AI Research for DINOv2
- COLMAP developers for SfM pipeline
- HLOC contributors for feature extraction framework
- Image Matching Challenge organizers
