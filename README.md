# Statistical Feature-Based EEG Classification for Harmful Brain Activity

## Overview

This project focuses on building a lightweight and interpretable machine learning system for detecting harmful brain activity from EEG signals using **statistical feature engineering**.

Instead of converting EEG signals into images or using deep learning, this approach extracts key statistical features from time-series data and applies classical machine learning models such as Random Forest and XGBoost.

---

## Objectives

* Efficiently represent EEG signals using statistical features
* Build an interpretable classification model
* Enable fast training and inference on CPU environments
* Provide a strong baseline for EEG classification tasks

---

## Dataset

* Competition: Harmful Brain Activity Classification
* Source: Harvard Medical School
* Size: ~106,800 samples

### Download via Kaggle API

```bash
kaggle competitions download -c hms-harmful-brain-activity-classification
```

---

## Data Description

### Input Features (EEG Signals)

20-channel EEG signals:

* Frontal: Fp1, Fp2, F3, F4
* Central: C3, C4, Cz
* Parietal: P3, P4, Pz
* Occipital: O1, O2
* Temporal: T3, T4, T5, T6
* Midline: Fz
* EKG

### Target Labels (6 Classes)

* Seizure
* LPD (Lateralized Periodic Discharges)
* GPD (Generalized Periodic Discharges)
* LRDA (Lateralized Rhythmic Delta Activity)
* GRDA (Generalized Rhythmic Delta Activity)
* Other

Labels are based on expert voting distributions.

---

## Methodology

### 1. Statistical Feature Engineering

From each EEG channel, extract:

* Mean
* Standard deviation
* Minimum / Maximum
* Range
* Skewness
* (Optional) Energy, kurtosis, entropy

These features summarize the temporal characteristics of EEG signals.

### 2. Tabular Data Construction

* Combine features from all 20 channels into a structured table
* Each row represents a single EEG segment

### 3. Model Training

* Random Forest (baseline)
* XGBoost (advanced boosting model)

### 4. Evaluation

* Cross-validation
* Metrics:

  * Accuracy
  * F1-score
  * Recall

---

## Project Structure

```
.
├── data/
│   ├── raw/
│   ├── processed/
├── notebooks/
├── src/
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── train_rf.py
│   ├── train_xgb.py
│   └── evaluate.py
├── results/
└── README.md
```

---

## Training Pipeline

1. Data loading and preprocessing
2. Statistical feature extraction per channel
3. Feature aggregation into tabular format
4. Model training (Random Forest / XGBoost)
5. Cross-validation and evaluation

---

## Expected Results

* Fast and stable training without GPU
* Competitive baseline performance
* Improved interpretability compared to deep learning models

---

## Key Contributions

* Efficient EEG representation using statistical features
* Interpretable machine learning pipeline
* Practical baseline for medical time-series classification

---

## Analysis and Interpretability

* Feature importance analysis to identify key EEG channels
* Insights into which brain regions contribute most to classification
* Transparent decision-making process for medical collaboration

---

## Future Work

* Real-time EEG monitoring system
* Integration with portable EEG devices
* Hybrid approach combining statistical features with deep learning

---

## Conclusion

This project demonstrates that EEG signals can be effectively classified using simple statistical features and classical machine learning models, providing a fast, interpretable, and practical alternative to deep learning approaches.
