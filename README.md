# Pneumonia Detection from Chest X-Rays using ResNeXt50

This project applies deep learning techniques to classify chest X-ray images as Pneumonia or Normal using the ResNeXt50 architecture.

## Overview

- **Dataset**: Publicly available chest X-ray images (Normal & Pneumonia)
- **Challenge**: Dataset imbalance and noisy labels
- **Solution**: Data resampling, augmentation, and regularization

## Methodology

- **Data Augmentation**: Rotation, shift, brightness, zoom, etc. using `ImageDataGenerator`
- **Model**: ResNeXt50 CNN
  - Grouped convolutions with increasing filter depth
  - Regularized using L2 and Dropout
- **Metrics**:
  - Precision, Recall, F1-Score, Accuracy
  - Confusion matrix for detailed error analysis

## Results

- **Model Parameters**: ~4.49M
- **Performance**:
  - High recall and precision for Pneumonia cases
  - Lower performance on Normal class due to class imbalance
- **Training Graphs**:
  - Consistent drop in training/validation loss
  - Accuracy improved over epochs

## Key Takeaways

- Effective in detecting pneumonia from X-rays
- Needs improvement for detecting normal cases
- Future work: Analyze misclassifications and improve class balance
