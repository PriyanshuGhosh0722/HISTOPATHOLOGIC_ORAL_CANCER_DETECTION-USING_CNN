# Histopathological Oral Cancer Classification - README

This repository provides a complete pipeline for the classification of oral cancer from histopathological images using advanced deep learning techniques. The project is structured as a Jupyter Notebook (`Histopathological_Oral_Cancer.ipynb`) and covers dataset preparation, model training, evaluation, and ensembling.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Setup](#setup)
- [Step-by-Step Workflow](#step-by-step-workflow)
  - [1. Environment Setup](#1-environment-setup)
  - [2. Data Preparation](#2-data-preparation)
  - [3. Data Exploration and Visualization](#3-data-exploration-and-visualization)
  - [4. Data Preprocessing and Generators](#4-data-preprocessing-and-generators)
  - [5. Model Architectures](#5-model-architectures)
    - [EfficientNetB3 Model](#efficientnetb3-model)
    - [ResNet-50 Model](#resnet-50-model)
    - [DenseNet-121 Model](#densenet-121-model)
    - [Inception Model](#inception-model)
    - [Ensemble Models](#ensemble-models)
  - [6. Training and Evaluation](#6-training-and-evaluation)
  - [7. Results and Performance](#7-results-and-performance)
- [Usage](#usage)
- [Notes](#notes)
- [Acknowledgements](#acknowledgements)

## Project Overview

This notebook demonstrates a full workflow for training deep learning models to classify histopathological images into OSCC (Oral Squamous Cell Carcinoma) and Normal tissue. The process includes preprocessing, data augmentation, model building (EfficientNet, ResNet, DenseNet, Inception), ensembling, and thorough evaluation (accuracy, confusion matrices).

## Dataset

- The dataset is downloaded from Kaggle using [`kagglehub`](https://www.kaggle.com/datasets/ashenafifasilkebede/dataset).
- Classes: **OSCC** (cancer) and **Normal**
- Upon extraction, files are organized into `train`, `val`, and `test` directories. Each folder contains subfolders for each class.

## Setup

You must have Python 3.x, TensorFlow 2.15, and all necessary libraries:
```bash
pip uninstall -y tensorflow
pip install tensorflow==2.15.0
# Install additional packages as required by the notebook:
pip install pandas seaborn matplotlib scikit-learn kagglehub opencv-python plotly
```

## Step-by-Step Workflow

### 1. Environment Setup
- Uninstall current TensorFlow version and install 2.15.0 for compatibility.
- Import required libraries: TensorFlow, Keras, pandas, numpy, matplotlib, seaborn, scikit-learn, cv2, etc.

### 2. Data Preparation

- Download the dataset and extract its content to the working directory.
- Organize folders: Ensure that `train`, `val`, and `test` directories exist, each with `OSCC` and `Normal`.
- Move image files for each class and split to a unified `/data` directory for easier DataFrame creation and model feeding.
- Mirror dataset structure to your Google Drive for persistence and use in cloud environments.

### 3. Data Exploration and Visualization

- Use pandas to create a DataFrame mapping image paths and their labels.
- Output dataset shape, null/missing values, and unique values for sanity check.
- Visualize class distribution using `seaborn` bar plots to check for class balance/imbalance.

### 4. Data Preprocessing and Generators

- **Functions are defined for:**
  - Collecting image filepaths and labels.
  - Splitting data into train/validation/test splits (stratified sampling for balanced sets).
  - Creating image data generators for augmentation (with horizontal flip, scaling, etc.).
  - Functions to display image samples for rapid data inspection.

### 5. Model Architectures

#### EfficientNetB3 Model

- Built using `tf.keras.applications.efficientnet.EfficientNetB3` as the base.
- BatchNorm, Dense, Dropout, and final Dense layer with softmax activation (2 classes).
- Compiled with `Adamax` optimizer and `categorical_crossentropy` loss.

#### ResNet-50 Model

- Similar structure: `ResNet50` base, BatchNorm, Dense, Dropout, Dense-softmax.
- Uses pre-trained ImageNet weights, with base layers frozen during training.

#### DenseNet-121 Model

- Uses `DenseNet121` as base (without top), followed by usual Dense/Dropout head.

#### Inception Model

- Similar approach with Inception architecture (details shown in code cells).

#### Ensemble Models

- Simple averaging (weighted voting) of model outputs (e.g., DenseNet & Inception or EfficientNet & ResNet).
- Ensemble models usually boost accuracy by combining strengths of multiple networks.

### 6. Training and Evaluation

- Models are trained using predefined callbacks:
  - **Early Stopping** (patience=5, monitor `val_loss`)
  - **ReduceLROnPlateau** (monitor `val_loss`, reduce LR by factor if no improvement)
- Output: Epoch-wise loss and accuracy for both train and validation sets.
- Evaluation generates loss and accuracy on train, validation, and test datasets.
- Confusion matrices (using `sklearn.metrics.confusion_matrix` and `seaborn.heatmap`) for detailed model performance on true vs. predicted labels.
- Training and validation loss/accuracy curves visualized for each model.

### 7. Results and Performance

- **EfficientNetB3**: High accuracy; overfitting is monitored and controlled via regularization and dropout.
- **ResNet-50**: Slightly lower test accuracy than EfficientNet, but strong validation accuracy.
- **DenseNet-121 & Inception**: Slightly lower accuracy; DenseNet shows potential underfitting, possibly due to limited train samples or batch size.
- **Ensembles**: Generally outperform individual models. For example:
  - Ensemble test accuracy: up to **97%**
  - Validation accuracy: up to **97%**
- Confusion matrices indicate most misclassifications occur when normal tissue is classified as OSCC or vice versa—critical for clinical relevance.

## Usage

- Clone this repository and follow the notebook step-by-step in Google Colab.
- Ensure Kaggle API credentials are set up for dataset download.
- Adjust batch size, number of epochs, or optimizer params as per your GPU/memory constraints and desired accuracy.
- Run all code cells sequentially for a working demo.

## Notes

- All custom functions are documented inline.
- Model selection (EfficientNet, ResNet, DenseNet, Inception) can be adjusted by commenting/uncommenting relevant sections.
- Ensemble approach can be tailored by changing the model weighting scheme.
- For deployment, consider freezing the best model and exporting for inference on unseen data.

## Acknowledgements

- Dataset: [Kaggle - ashenafifasilkebede/dataset]
- Model architectures: Keras Applications
- Support libraries: pandas, numpy, seaborn, matplotlib, scikit-learn

**For more details, model summaries, and exact cell-by-cell explanations, refer to the Jupyter notebook included in this repository, where each cell is commented thoroughly.**
