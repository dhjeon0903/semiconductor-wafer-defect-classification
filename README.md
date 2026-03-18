# Semiconductor Wafer Defect Classification

AI-based semiconductor wafer defect classification using the WM-811K wafer map dataset.

## Overview
This project explores machine learning and deep learning approaches for semiconductor wafer defect pattern classification from semiconductor wafer maps.

The main goal was to compare a simple baseline model with convolutional neural network (CNN) approaches and analyze the challenges of defect classification, especially class imbalance and overfitting.

## Project Workflow
The project was developed in several stages:

1. **Dataset loading and preprocessing**
   - Loaded the WM-811K wafer map dataset from pickle format
   - Cleaned nested labels such as training/test labels and failure types
   - Visualized wafer map samples

2. **Baseline machine learning model**
   - Built a Random Forest classifier using handcrafted wafer-level features
   - Used features such as valid cell count, defect cell count, and defect ratio
   - This baseline showed limited performance because it could not capture spatial defect patterns well

3. **Multiclass CNN model**
   - Built a CNN to classify wafer maps into multiple defect classes
   - The model learned spatial structure better than the baseline
   - However, the model showed strong overfitting:
     - training accuracy became very high
     - validation accuracy remained very low
     - test generalization was limited

4. **Improved lightweight CNN experiment**
   - To reduce overfitting and shorten training time, an additional experiment was performed
   - Changes included:
     - balanced sampling across classes
     - smaller training and test subsets
     - a lighter CNN architecture
     - additional dropout and regularization
   - This version trained faster, but overly aggressive downsampling caused the model to collapse toward predicting a single class

## Methods
### Baseline
- Random Forest classifier
- Feature-based approach using summary statistics from wafer maps

### Deep Learning
- Multiclass CNN for defect classification
- Additional lightweight CNN experiment with:
  - balanced sampling
  - dropout
  - data augmentation
  - L2 regularization

## Dataset
- **Dataset:** WM-811K Wafer Map Dataset
- **Main classes used:**
  - none
  - Center
  - Donut
  - Edge-Loc
  - Edge-Ring
  - Loc
  - Random
  - Scratch
  - Near-full

## Key Findings
- CNN-based models were more appropriate than the baseline model because wafer maps contain important spatial patterns
- Severe class imbalance made training difficult
- The original multiclass CNN showed overfitting
- A smaller balanced version reduced training time but also reduced class separability too much
- This project shows that practical wafer defect classification requires careful handling of data balance, model complexity, and regularization

## Outputs
The project generates:
- training history plots
- confusion matrix plots
- sample prediction visualizations

## Note
Dataset files are not included in this repository due to size.
Please download the WM-811K dataset separately and place it in the project directory before running the scripts.
