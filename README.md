# Image Matching Challenge 2025 - Structure from Motion Pipeline

A comprehensive Structure from Motion (SfM) pipeline for the Image Matching Challenge 2025, leveraging DINOv2 global features, HDBSCAN clustering, and COLMAP reconstruction for robust camera pose estimation.

## Overview

This project implements an end-to-end pipeline for estimating camera poses (rotation matrices and translation vectors) from unordered image collections. The solution combines modern deep learning approaches with traditional computer vision techniques to achieve high reconstruction rates across diverse datasets.

## Key Features

- **DINOv2 Global Feature Extraction**: Uses Facebook's DINOv2 ViT-S/14 model for robust image embeddings
- **HDBSCAN Clustering**: Intelligent scene segmentation with noise reassignment
- **PyTorch-based Similarity Search**: Efficient pair generation using matrix operations
- **COLMAP Integration**: Robust 3D reconstruction using state-of-the-art SfM
- **Resume Capability**: Intelligent caching to avoid redundant computations
- **Mac-Optimized**: Special configurations for stability on macOS systems

## Architecture

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
   - Configurable image resolution (default: 1600px)

5. **Structure from Motion**
   - Per-cluster COLMAP reconstruction
   - Multi-format pose extraction (binary/text)
   - Comprehensive error handling and fallbacks

## Installation

### Dependencies

```bash
# Core dependencies
pip install torch torchvision
pip install numpy pandas pillow tqdm
pip install scipy scikit-learn
pip install hdbscan pycolmap

# Computer vision
pip install hloc extract_features match_features reconstruction

# Optional: For alternative DINOv2 loading
pip install transformers
```

### System Requirements

- **Python**: 3.8+
- **PyTorch**: 1.12+ (with MPS/CUDA support recommended)
- **Memory**: 16GB+ RAM for large datasets
- **Storage**: 50GB+ for features and intermediate results

## Usage

### Quick Start

```python
# Set up environment variables for Mac stability
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# Run the main pipeline
python ml-project.ipynb
```

### Configuration

Key hyperparameters can be adjusted at the top of the notebook:

```python
TOP_K = 50              # Nearest neighbors for similarity search
RESIZE_MAX = 1600       # Local feature resolution
MAX_PAIRS_PER_DS = 8000 # Maximum pairs per dataset
EXHAUSTIVE_THRESH = 80  # Use exhaustive matching for small clusters
TIME_BUDGET = 8.0 * 3600 # Total processing time budget
```

### Data Organization

The pipeline expects the following directory structure:

```
image-matching-challenge-2025/
├── test/
│   ├── dataset1/
│   ├── dataset2/
│   └── ...
├── train/
│   ├── dataset1/
│   └── ...
└── sample_submission.csv
```

## Output Structure

```
output/
├── features/          # ALiED features per dataset
├── matches/           # LightGlue matches
├── sfm/              # COLMAP reconstructions
│   └── dataset/
│       ├── cluster_0/
│       ├── cluster_1/
│       └── ...
└── pairs/            # Generated image pairs
```

## Performance

### Benchmark Results

- **Average Reconstruction Rate**: ~85% across 13 datasets
- **Processing Time**: ~8 hours for full competition dataset
- **Memory Usage**: Optimized for 16GB systems
- **Scene Diversity**: Automatic detection of 3-8 scenes per dataset

### Dataset Performance

| Dataset | Images | Reconstruction Rate | Scenes |
|---------|--------|-------------------|--------|
| ETs | 22 | 86.4% | 3 |
| amy_gardens | 200 | ~85% | 4 |
| ... | ... | ... | ... |

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

## Troubleshooting

### Common Issues

1. **Kernel Crashes on Mac**
   - Ensure all thread environment variables are set
   - Monitor memory usage with `torch.mps.empty_cache()`

2. **Low Reconstruction Rates**
   - Check image quality and variety
   - Adjust HDBSCAN parameters
   - Verify pair generation counts

3. **COLMAP Failures**
   - Ensure sufficient image pairs per cluster
   - Check feature extraction quality
   - Verify image EXIF data

### Debug Mode

Enable verbose logging by modifying the warning filter:

```python
# warnings.filterwarnings("ignore")  # Comment out for debug mode
```

## Competition Integration

### Submission Format

The pipeline generates submissions in the required format:

```csv
dataset,image,scene,rotation_matrix,translation_vector
dataset1,image1.jpg,scene_0,"r11;r12;r13;r21;r22;r23;r31;r32;r33","t1;t2;t3"
```

### Evaluation Metrics

- **Reconstruction Rate**: Percentage of images with valid poses
- **Scene Accuracy**: Correct scene clustering
- **Pose Precision**: Accuracy of rotation and translation estimates

## Future Improvements

- **Multi-Scale Features**: Integration of SuperPoint + DINOv2
- **Graph Neural Networks**: Learned pair selection
- **Adaptive Clustering**: Dynamic parameter tuning
- **Ensemble Methods**: Multiple reconstruction approaches

## License

This project is developed for the Image Matching Challenge 2025. Please refer to the competition guidelines for usage restrictions.

## Acknowledgments

- Facebook AI Research for DINOv2
- COLMAP developers for SfM pipeline
- HLOC contributors for feature extraction framework
- Image Matching Challenge organizers
