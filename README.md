# Statistical Feature-Based EEG Pattern Analysis and Anomaly Detection

## Overview

This project focuses on building a lightweight and interpretable system for analyzing EEG signals using **statistical feature engineering and unsupervised learning**.

Instead of relying on labeled data or deep learning models, this approach extracts key statistical features from time-series EEG signals and explores their **underlying structure and abnormal patterns** without predefined labels.

The goal is not to classify predefined brain activity types, but to **discover latent patterns and detect anomalous EEG signals**.

---

## Objectives

* Represent EEG signals efficiently using statistical features
* Discover latent structures and patterns in EEG data
* Detect abnormal EEG signals without relying on labels
* Enable fast inference in CPU-based environments
* Provide interpretable insights into EEG signal behavior

---

## Dataset

* Competition: Harmful Brain Activity Classification
* Source: Harvard Medical School
* Size: ~106,800 samples

### Download via Kaggle API

```bash
kaggle competitions download -c hms-harmful-brain-activity-classification
```

## Data Description

### Input Features (EEG Signals)

20-channel EEG signals:

- **Frontal**: Fp1, Fp2, F3, F4  
- **Central**: C3, C4, Cz  
- **Parietal**: P3, P4, Pz  
- **Occipital**: O1, O2  
- **Temporal**: T3, T4, T5, T6  
- **Midline**: Fz  
- **EKG**

---

### Reference Labels (Used for Post-hoc Analysis Only)

- Seizure  
- LPD (Lateralized Periodic Discharges)  
- GPD (Generalized Periodic Discharges)  
- LRDA (Lateralized Rhythmic Delta Activity)  
- GRDA (Generalized Rhythmic Delta Activity)  
- Other  

> **Note:** These labels are **NOT used for training**, but only for interpreting and validating discovered patterns.

---

## Methodology

### 1. Statistical Feature Engineering

From each EEG channel, extract:

- Mean  
- Standard deviation  
- Minimum / Maximum  
- Range  
- Skewness  
- (Optional) Energy, kurtosis, entropy  

These features summarize the temporal characteristics of EEG signals.

---

### 2. Tabular Data Construction

- Combine features from all 20 channels into a structured table  
- Each row represents a single EEG segment  

---

### 3. Unsupervised Learning

#### Clustering

- K-means  
- Gaussian Mixture Model (GMM)  

**Goal:**  
- Discover natural groupings of EEG patterns  

---

#### Anomaly Detection

- Isolation Forest  
- One-Class SVM  

**Goal:**  
- Detect EEG signals that deviate from normal patterns  

---

### 4. Dimensionality Reduction & Visualization

- PCA  
- t-SNE / UMAP  

**Goal:**  
- Visualize high-dimensional EEG feature space  
- Understand cluster structure and data distribution  

---

### 5. Evaluation

Since this is an unsupervised approach, evaluation is performed using:

#### Internal Metrics

- Silhouette Score  
- Cluster separation  

#### External Analysis

- Compare clusters with expert labels  
- Analyze relationship between anomaly scores and harmful brain activity  

---

## Project Structure

```bash
.
├── data/
│ ├── raw/
│ ├── processed/
├── notebooks/
├── src/
│ ├── preprocessing.py
│ ├── feature_engineering.py
│ ├── clustering.py
│ ├── anomaly_detection.py
│ └── evaluation.py
├── results/
└── README.md
```

---

## Pipeline

1. Data loading and preprocessing  
2. Statistical feature extraction per channel  
3. Feature aggregation into tabular format  
4. Clustering and anomaly detection  
5. Visualization and structure analysis  
6. Post-hoc comparison with expert labels  

---

## Expected Results

- Identification of latent EEG pattern clusters  
- Detection of abnormal EEG signals  
- Fast inference without GPU  
- Interpretable statistical insights into brain activity  

---

## Key Contributions

- Unsupervised learning framework for EEG analysis  
- Statistical feature-based anomaly detection  
- Interpretable and lightweight pipeline  
- Ability to discover previously undefined EEG patterns  

---

## Analysis and Interpretability

- Statistical feature analysis per cluster  
- Identification of abnormal signal characteristics  
- Explanation of anomalies based on signal variability and distribution  
- Comparison with known clinical labels for validation  

---

## Future Work

- Real-time EEG anomaly monitoring system  
- Integration with portable EEG devices  
- Semi-supervised learning using limited labels  
- Hybrid approach combining statistical and deep learning features  

---

## Conclusion

This project demonstrates that EEG signals can be effectively analyzed without relying on predefined labels, by leveraging statistical features and unsupervised learning techniques.

Rather than focusing on classification, the system aims to **understand the structure of EEG data and detect abnormal patterns**, providing a flexible and interpretable approach for real-world monitoring scenarios.
