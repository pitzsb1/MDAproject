# EEG Unsupervised Analysis & Anomaly Detection

## Overview

EEG (Electroencephalogram) signals are high-dimensional time-series data with complex and continuous patterns.
This project explores **unsupervised learning approaches** to:

* Discover latent structure in EEG data
* Detect abnormal brain activity patterns
* Build an interpretable and lightweight analysis pipeline

---

## Objectives

1. **Latent Structure Discovery**
   Identify hidden patterns and cluster structures in EEG signals.

2. **Anomaly Detection**
   Automatically detect abnormal EEG signals without labels.

3. **Interpretability**
   Analyze statistical characteristics of clusters and anomalies.

4. **Lightweight System**
   Build a CPU-efficient pipeline without deep learning.

---

## Dataset

* Source: HMS Harmful Brain Activity Classification (Kaggle)
* EEG shape: `(10000 timesteps × 20 channels)`
* Total EEG files: ~17,300
* Labels (for post-hoc analysis):

  * Seizure, GPD, LPD, GRDA, LRDA, Other

### Key Observations

* EEG channels: similar scale (std ~50)
* EKG channel: extremely large scale (std ~3000)
* Multiple segments per EEG ID

---

## Feature Engineering

Statistical features were extracted from each EEG channel:

* Mean, Std
* Min, Max, Range
* Skewness, Kurtosis
* Absolute Mean
* Flat Signal Flag

> Initially included autocorrelation features, but removed due to noise sensitivity and computational cost.

---

## Critical Issue: Artifact Detection

### Initial Problem

The anomaly detection model identified **sensor artifacts instead of meaningful EEG patterns**.

Examples:

* Sudden spikes (~30,000 amplitude)
* Flat or corrupted signals

---

### Solution: Data Cleaning

Artifact filtering applied:

* Extreme amplitude threshold
* Abnormal global variance
* High NaN ratio

Result:

```
17300 → 8739 EEG samples (cleaned)
```

 Key Insight:
> **Data quality had a larger impact than model choice**

---

## Dimensionality Reduction (PCA)

* PCA applied after scaling and imputation
* 2 components explained ~45–54% variance

### Insight

EEG data does **not form discrete clusters**, but rather:

> -> Continuous structure of brain activity patterns

---

## Clustering (K-Means)

* Tested k = 2 ~ 10 (Elbow method)
* Optimal k ≈ 3
* Silhouette Score ≈ **0.77**

### Insight

* Clusters formed clearly in feature space
* But did **not perfectly align with labels**

-> EEG patterns are **continuous, not categorical**

---

## Anomaly Detection (Isolation Forest)

* Contamination: 5%
* Output:

  * `-1`: anomaly
  * `1`: normal
  * continuous anomaly score

### Key Findings

| Before Cleaning   | After Cleaning            |
| ----------------- | ------------------------- |
| Detects artifacts | Detects real EEG patterns |
| Noisy results     | Structured anomalies      |

---

## Results

### Major Findings

* EEG data exhibits **continuous latent structure**
* Unsupervised models detect **degree of abnormality**, not strict classes
* Artifact removal is **critical for meaningful results**

---

## Interpretation

Anomalous EEG signals showed:

* Higher variance (std)
* More extreme values (range)
* Increased spikiness (kurtosis)
* Non-flat dynamic patterns

---

## Key Takeaways

> “Unsupervised learning does not classify EEG —
> it reveals the structure of brain activity.”

* Not a classification problem
* But a **pattern discovery problem**

---

## Tech Stack

* Python
* Pandas / NumPy
* Scikit-learn
* Matplotlib / Seaborn
* Joblib (parallel processing)

---

## Future Work

* Frequency domain features (FFT)
* Time-frequency analysis (STFT, Wavelet)
* Deep learning comparison (CNN / Transformer)
* Real-time EEG anomaly detection system

---

## Example Visualization

* PCA projection
* K-means clustering
* Anomaly score heatmap

---

## Conclusion

This project demonstrates that:

* EEG data is inherently continuous
* Unsupervised learning is effective for structure discovery
* Data preprocessing is more important than model complexity

---

## Final Thought

> “Once noise is removed, the model starts to see the brain.”

---
