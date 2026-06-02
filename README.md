# Face Anti-Spoofing Detection using Manual PCA and Hybrid PCA-CNN Model

## Overview

This project presents a complete facial anti-spoofing system designed to distinguish between real (live) faces and spoofed facial attacks. The work combines a fully manual implementation of Principal Component Analysis (PCA) with classical machine learning classifiers and a deep learning hybrid architecture.

The proposed framework was developed as part of a Data Analysis course and evaluated using the CelebA-Spoof dataset, one of the largest publicly available facial anti-spoofing datasets.

The project explores three complementary approaches:

* **Manual Principal Component Analysis (PCA)** implemented from scratch using NumPy.
* **Classical Machine Learning Classification** using K-Nearest Neighbors (KNN) and Support Vector Machines (SVM).
* **Hybrid PCA + Convolutional Neural Network (CNN)** architecture combining global and local facial features.

---

## Objectives

The main goals of this project are:

* Understand and implement PCA without automated machine learning libraries.
* Reduce the dimensionality of facial images while preserving essential information.
* Visualize and analyze facial data using principal components and eigenfaces.
* Compare traditional machine learning classifiers for face anti-spoofing.
* Develop a hybrid PCA-CNN architecture for improved detection performance.
* Evaluate model performance using multiple statistical and classification metrics.

---

## Dataset

### CelebA-Spoof Dataset

The project uses the CelebA-Spoof dataset, a large-scale facial anti-spoofing benchmark containing more than 600,000 images from 10,177 subjects.

#### Dataset Characteristics

* More than 600,000 facial images.
* 10,177 unique individuals.
* Real and spoofed face samples.
* Multiple attack scenarios.
* Various lighting conditions and environments.
* Train, Validation, and Test partitions.

#### Supported Spoofing Attacks

* Printed photo attacks.
* Replay attacks using mobile or computer screens.
* Cut-photo attacks.
* Partial 3D mask attacks.

---

## Methodology

### 1. Image Preprocessing

Each image undergoes several preprocessing operations:

* Grayscale conversion.
* Image resizing to 64×64 pixels.
* Flattening into a 4096-dimensional vector.
* Pixel normalization into the range [0,1].

For the CNN branch:

* RGB image loading.
* Resizing to 128×128 pixels.
* Tensor normalization.

---

### 2. Manual Principal Component Analysis (PCA)

Unlike standard implementations using Scikit-Learn, PCA was implemented manually using NumPy through the following steps:

#### Step 1 – Mean Face Computation

Calculation of the average facial image across the dataset.

#### Step 2 – Data Centering

Subtraction of the mean face from every sample.

#### Step 3 – Standardization

Variance normalization for each pixel.

#### Step 4 – Correlation Matrix Construction

Computation of the pixel correlation matrix.

#### Step 5 – Eigenvalue Decomposition

Extraction of eigenvalues and eigenvectors.

#### Step 6 – Eigenfaces Generation

Visualization of principal components as facial patterns.

#### Step 7 – Explained Variance Analysis

Selection of the optimal number of principal components.

#### Step 8 – Projection into PCA Space

Transformation of images into a lower-dimensional feature space.

---

### 3. KNN Classification

The PCA-reduced feature vectors are classified using K-Nearest Neighbors.

Characteristics:

* Euclidean distance metric.
* k = 5 nearest neighbors.
* Simple and interpretable baseline classifier.

---

### 4. SVM Classification

Support Vector Machines with RBF kernel are used to model nonlinear decision boundaries.

Characteristics:

* RBF kernel.
* Balanced class weighting.
* Probabilistic outputs.
* Improved robustness compared to KNN.

---

### 5. Hybrid PCA + CNN Architecture

The proposed hybrid model combines global information extracted by PCA with local texture features learned by a Convolutional Neural Network.

#### PCA Branch

```
64×64 Grayscale Image
        ↓
Manual PCA
        ↓
200 Principal Components
        ↓
Dense Layer
```

#### CNN Branch

```
128×128 RGB Image
        ↓
Conv2D + MaxPooling
        ↓
Conv2D + MaxPooling
        ↓
Conv2D
        ↓
Global Average Pooling
        ↓
Dense Layer
```

#### Feature Fusion

```
PCA Features
        +
CNN Features
        ↓
Concatenation
        ↓
Dense Layers
        ↓
Dropout
        ↓
Sigmoid Output
```

---

## Experimental Evaluation

The models were evaluated using:

* Accuracy
* ROC Curve
* Area Under Curve (AUC)
* Confusion Matrix
* Explained Variance Ratio
* Principal Component Visualization
* Eigenface Analysis

Additional visualizations include:

* PCA Projection Spaces
* Correlation Circle
* 3D Correlation Sphere
* Scree Plot
* Image Reconstruction Analysis

---

## Results

### Classification Performance

| Model            | Accuracy | AUC   |
| ---------------- | -------- | ----- |
| KNN + PCA        | 84.33%   | -     |
| SVM + PCA        | 85.67%   | 0.908 |
| Hybrid PCA + CNN | 94.67%   | 0.992 |

### Key Findings

* PCA reduced image dimensionality from **4096 features to 200 components**.
* The selected PCA representation preserved **91.4% of total variance**.
* Eigenfaces successfully captured major facial variations.
* SVM outperformed KNN due to nonlinear decision boundaries.
* The hybrid PCA-CNN architecture achieved the best overall performance.
* Combining global PCA features and local CNN features significantly improved spoof detection.

---

## Visual Analysis

The project includes several visualization modules:

* Mean Face Representation.
* Centered and Standardized Images.
* Eigenfaces Visualization.
* Scree Plot Analysis.
* Cumulative Explained Variance.
* Real vs Fake PCA Projection.
* Correlation Circle.
* 3D Correlation Sphere.
* Confusion Matrix.
* ROC Curves.
* PCA Reconstruction Analysis.

---

## Technologies Used

* Python
* NumPy
* OpenCV
* Matplotlib
* Seaborn
* Scikit-Learn
* TensorFlow
* Keras
* PCA
* CNN
* SVM
* KNN

---



---

## Key Contributions

* Complete manual implementation of PCA using NumPy.
* Mathematical derivation of all PCA stages.
* Eigenface-based facial representation analysis.
* Comparative evaluation of KNN and SVM classifiers.
* Development of a hybrid PCA-CNN anti-spoofing framework.
* Extensive visualization and interpretability study.
* High-performance facial liveness detection system.

---

## Future Work

Potential improvements include:

* Vision Transformer (ViT) architectures.
* Attention-based CNN models.
* Deepfake detection integration.
* Real-time deployment on edge devices.
* Multi-modal biometric authentication systems.
* Explainable AI techniques for spoof detection.

---

## Authors

**CHALLAL Saloua**
**REHAL Amel**
**GHEMMOUR Lehna**

Master 1 – Artificial Intelligence and Virtual Modeling (M1 IV)

University of Science and Technology Houari Boumediene (USTHB)

Academic Year 2025–2026
