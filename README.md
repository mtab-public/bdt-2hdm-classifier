# BDT Classifier for 2HDM Event Analysis

This repository implements a Boosted Decision Tree (BDT) framework to discriminate signal and background events in a Two-Higgs-Doublet Model (2HDM) analysis. It includes training, evaluation, feature studies, and basic data exploration tools.


## Overview

Main components:

* `train_bdt.py` → Train BDT model and save it
* `analysis.py` → Evaluate model, compute optimal BDT cut, produce plots
* `bdt_sort_features.py` → Study feature importance via removal tests
* `plot_correlations.py` → Explore feature distributions and correlations


## Requirements

* Python ≥ 3.8
* `numpy`
* `matplotlib`
* `scikit-learn`

Install with:

```bash
pip install numpy matplotlib scikit-learn
```


## Data Format

* Input data: `.txt` (raw) or `.dat` (processed binary via NumPy)
* Each event has **22 features**
* Feature `0` = event weight
* Training uses selected features:

  ```
  [1,2,3,4,5,9–21]
  ```


## Event Selection

Applied in all scripts:

* ( E_T^{miss} > 35 )
* ( H_T > 350 )


## Usage

### 1. Train BDT

```bash
python train_bdt.py <signal_process>
```

* Trains an **AdaBoost (BDT)** classifier
* Saves model to:

  ```
  models/<region>/<signal>/mhc_<signal>.sav
  ```


### 2. Run Analysis

```bash
python analysis.py <signal_process>
```

Outputs:

* Accuracy (train/test)
* Confusion matrix
* BDT score distribution
* ROC curve
* **Optimal BDT cut (via significance maximization)**

Plots saved in:

```
plots/<region>/<signal>/
```


### 3. Feature Importance Study

```bash
python bdt_sort_features.py <signal_process>
```

* Removes features (single + pairs)
* Retrains BDT
* Evaluates impact on **signal significance**


### 4. Feature Exploration

```bash
cd subfolder/
python plot_correlations.py <signal_process>
```

* Visualizes feature distributions (e.g. jet multiplicity)
* Contains optional (commented) 2D feature scatter plots


## BDT Optimization

The optimal cut is computed by maximizing:

```
S / sqrt(S + B + (ε_sys · B)^2)
```

with:

* ( ε_{sys} = 0.05 ) (or 0.10 in correlation study)


## Background Processes

* `ttbarjs_5FS`
* `wt_jets`
* `tj_plus_1jet`
* `ttbarh`
* `ttbarZ`


## Notes

* Training uses **one background sample** (`ttbarjs_5FS`); others are included in testing
* Event weights are computed but **not used in training**
* Feature definitions are not documented here
* Code is script-based (limited modularization)


## Structure

```
.
├── train_bdt.py
├── analysis.py
├── bdt_sort_features.py
├── data/
├── models/
├── plots/
├── correlations/
│   └── plot_correlations.py
```


## Summary

This repository provides a complete pipeline for:

* BDT-based signal/background classification
* Physics-motivated cut optimization
* Feature sensitivity studies

It is intended for fast iteration and exploratory analysis in 2HDM studies.
