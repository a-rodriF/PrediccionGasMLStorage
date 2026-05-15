# Predicting gas consumption in Ethereum transactions using Machine Learning on storage-aware data.
Final Degree Computer Sciences Proyect for Double Degree in Mathematics and Computer Science UCM
Ana Rodríguez de la Fuente

Machine learning framework for Ethereum transaction gas prediction using ABI decoding, transaction metadata and smart contract storage analysis.

## Overview

This project focuses on predicting the `receipt_gas_used` value of Ethereum transactions by training machine learning models for each `(contract address, function signature)` pair.

The framework combines:

- ABI decoding
- Transaction input feature extraction
- Storage state analysis
- Storage slot mapping
- KNN-based storage imputation
- Multiple regression models

The objective is to evaluate how storage information affects gas prediction accuracy in Ethereum smart contracts.

---

## Features

- Ethereum ABI input decoding
- Automatic feature engineering from transaction inputs
- Smart contract storage extraction from JSON traces
- Slot mapping system for storage normalization
- KNN-based missing storage reconstruction
- Parallelized preprocessing and training pipeline
- Per-signature gas prediction models
- Multiple ML regressors:
  - Linear Regression
  - Ridge Regression
  - SVR (linear)
  - SVR (RBF)
  - Gradient Boosting
  - Random Forest
  - XGBoost
- Storage-aware vs no-storage model comparison
- Stratified sampling by gas usage
- SHAP explainability support
- Export/import system for trained models

---

## Project Structure

```text
model/
│
├── config/                 # Configuration and execution settings
├── data_processing/        # Data loading and storage extraction
├── export/                 # Export utilities
├── modeling/               # ML model training
├── preprocessing/          # Feature engineering and storage processing
├── utils/                  # Parallel execution utilities
│
evaluacion/                 # Experiments and evaluation notebooks
data/                       # Input/output datasets
```

---

## Pipeline

### 1. Transaction Loading

Transactions are loaded from CSV files and grouped by:

- Contract address
- Function signature
Only example files have been provided in the repository.

### 2. ABI Decoding

Transaction inputs are decoded using contract ABI definitions.

Generated features include:

- Integer parameters
- Array sizes
- Magnitude proxies for large integers
- Zero indicators

### 3. Storage Processing

Storage traces are parsed and transformed into structured numerical features.

Features include:

- Storage slot identifiers
- Storage values
- Large-value indicators
- Storage length

### 4. Missing Storage Handling

The framework supports multiple strategies:

| Mode | Description |
|---|---|
| 0 | Remove rows without storage |
| 1 | Replace missing storage with empty dictionaries |
| 2 | Predict missing storage using KNN |

### 5. Model Training

Models are trained independently for each contract and signature.

Two model groups are generated:

- With storage information
- Without storage information

---

## Installation

### Clone repository

```bash
git clone https://github.com/YOUR_USERNAME/storage-aware-gas-prediction.git
cd storage-aware-gas-prediction
```

### Install dependencies

Using Poetry:

```bash
poetry install
```

Or using pip:

```bash
pip install -r requirements.txt
```

---

## Main Dependencies

- Python 3.12+
- pandas
- numpy
- scikit-learn
- xgboost
- tqdm
- pyarrow
- shap
- web3-input-decoder

---

## Configuration

Configuration is managed through:

```text
model/config.env
```

Main parameters include:

| Parameter | Description |
|---|---|
| `MIN_LEN_TSX` | Minimum samples required per signature |
| `MAX_LEN_TSX` | Maximum samples per signature |
| `STRG_MANAGEMENT` | Missing storage handling strategy |
| `DO_TRAIN_NOT_KNN` | Train gas models or KNN storage models |
| `MAX_LEN_KNN` | Maximum dataset size for KNN |

---

## Usage

### Run complete pipeline

```bash
poetry run python -m model.main
```

### Train only gas models

Set:

```env
DO_TRAIN_NOT_KNN=True
```

### Train only KNN storage models

Set:

```env
DO_TRAIN_NOT_KNN=False
```

## Evaluation

Evaluation notebooks are available in:

```text
evaluacion/
```

These notebooks include:

- MAE analysis
- R² comparison
- Storage impact analysis
- SHAP explainability
- Feature importance visualization
- Outlier filtering experiments

---

## Research Objective

This project investigates whether Ethereum gas consumption can be more accurately predicted when incorporating storage-state information alongside traditional transaction and ABI-derived features.

---

## Future Work

- Graph neural networks for smart contract interactions
- Dynamic gas estimation APIs
- Feature selection optimization
- Distributed training
- Adaptation to classification problem
- Use of NLP for interrelation extraction 

---

## Author

Ana Rodríguez de la Fuente

Double Degree in Mathematics and Computer Science


---

## License

This project is intended for academic and research purposes.
