# Sample Dataset - Image Matching Challenge 2025

This directory contains a small sample dataset from the Image Matching Challenge 2025, specifically from the ETs dataset.

## Sample Dataset Contents

The `ETs_sample/` directory contains 9 images:

- `et_et000.png` through `et_et004.png` (5 images)
- `another_et_another_et001.png` through `another_et_another_et004.png` (4 images)

## Purpose

This sample dataset is provided to:
- Test the pipeline functionality without downloading the full dataset
- Verify installation and setup
- Demonstrate the complete workflow on a small scale
- Allow quick experimentation with different parameters

## Usage

### Quick Test
1. Ensure the main pipeline is installed and configured
2. Run the main notebook `ml-project.ipynb`
3. The pipeline will automatically detect and process this sample data

### Expected Results
With this sample dataset, you should expect:
- **Processing Time**: 2-3 minutes
- **Reconstruction Rate**: ~85% (7-8 out of 9 images)
- **Scenes Detected**: 2-3 clusters
- **Memory Usage**: ~2GB

### Modifying for Testing
To use this sample with the main pipeline, you may need to:
1. Update the data path in the notebook to point to this directory
2. Adjust hyperparameters for the smaller dataset size
3. Monitor the output to ensure proper processing

## Data Characteristics

- **Image Format**: PNG
- **Image Size**: Variable (approximately 400-500KB each)
- **Scene Content**: Extraterrestrial-themed images
- **Image Quality**: High resolution with good texture

## Integration with Main Pipeline

To integrate this sample with the main pipeline:

1. Copy the `ETs_sample` directory to the appropriate location
2. Update the dataset discovery code to include this path
3. Run the pipeline with reduced parameters for faster processing

## Limitations

- This is a very small sample and may not represent full dataset complexity
- Reconstruction rates may vary compared to the full dataset
- Some advanced features may not be fully testable with this limited data

## Full Dataset

For complete results and competition submission, download the full dataset from:
[Image Matching Challenge 2025](https://www.kaggle.com/c/image-matching-challenge-2025)
