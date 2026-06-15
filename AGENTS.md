# AGENTS.md

## Project Overview

This project studies Learnable Linear Canonical Transform (LCT) and Fractional Riesz Transform for Financial Time Series Forecasting.

The goal is to build a frequency-domain enhanced deep learning framework that improves forecasting accuracy and interpretability for financial market data.

Target applications include:

* Stock price forecasting
* Bitcoin price forecasting
* Financial trend prediction
* Volatility analysis
* Frequency-domain feature extraction

---

## Research Objectives

The research should focus on:

1. Frequency-domain representation of financial time series.
2. Learnable LCT parameter optimization.
3. Fractional-order Riesz feature enhancement.
4. Integration with deep learning forecasting models.
5. Model interpretability through learned transform parameters.

The project is intended for academic research and thesis publication.

---

## Preferred Model Architecture

Input Financial Series

↓

Normalization

↓

1D FFT

↓

Learnable LCT

↓

Fractional Riesz Transform

↓

Inverse Transform

↓

Feature Fusion

↓

Forecasting Backbone

↓

Prediction Output

---

## Backbone Models

Preferred order:

1. LSTM
2. TCN
3. Transformer

All new experiments should support easy replacement of forecasting backbones.

---

## Coding Standards

* Use Python 3.12+
* Use PyTorch
* Use type hints whenever possible
* Keep functions modular
* Write clear comments for mathematical operations
* Avoid hard-coded paths
* Use configuration files instead of magic numbers

Every function should have a concise docstring.

---

## Reproducibility Requirements

Every experiment must be reproducible.

Required:

* Fixed random seed
* Save configuration
* Save model checkpoint
* Save training logs
* Save evaluation metrics

Record:

* Learning rate
* Batch size
* Epoch number
* Dataset name
* Random seed
* Device information

---

## Dataset Management

Raw datasets must never be modified.

Directory structure:

data/
├── raw/
├── processed/
└── external/

Rules:

* Preserve original files
* Save processed datasets separately
* Document preprocessing steps

---

## Experiment Output Requirements

Every training run must automatically save:

outputs/
├── training_log.csv
├── metrics.json
├── experiment_summary.txt
└── prediction_results.csv

Required metrics:

* MAE
* MSE
* RMSE
* MAPE
* R²

---

## Model Checkpoint Requirements

Save:

checkpoints/
├── best_model.pth
├── latest_model.pth
└── checkpoint_epoch_x.pth

The best model should be selected according to validation performance.

---

## Learnable LCT Requirements

For every experiment using learnable LCT:

Export learned parameters.

Required output:

outputs/learned_lct_parameters.txt

Include:

* alpha_x
* alpha_y
* m_x
* m_y
* q_x
* q_y

and corresponding

* A
* B
* C
* D

matrix entries.

---

## Visualization Requirements

Every experiment must generate publication-quality figures.

Save all figures in:

figures/

Required visualizations:

1. Training Loss Curve
2. Validation Loss Curve
3. Prediction vs Ground Truth
4. Residual Distribution
5. Error Histogram
6. Frequency-domain Feature Visualization
7. Learned LCT Parameter Visualization

Image format:

* PNG
* PDF

Minimum resolution:

300 DPI

Suitable for thesis and journal publication.

---

## Thesis-Oriented Outputs

Every completed experiment should generate:

outputs/

* experiment_summary.txt
* training_log.csv
* metrics.json
* learned_lct_parameters.txt

figures/

* loss_curve.png
* prediction_curve.png
* residual_distribution.png
* frequency_feature.png
* lct_parameter_analysis.png

These files should be directly usable in a master's thesis.

---

## Current Research Direction

Current topic:

Learnable LCT-Riesz Transform Enhanced Financial Time Series Forecasting

Research focus:

* Learnable ABCD matrix optimization
* Frequency-domain enhancement
* Financial sequence representation
* Deep forecasting networks
* Explainable spectral learning

Future extension:

* Multi-scale frequency learning
* Adaptive fractional order learning
* Transformer integration
* Cross-market forecasting
* Financial risk prediction

---

## Important Rule

Whenever code is modified:

1. Preserve existing functionality.
2. Do not remove experiment outputs.
3. Do not remove visualization generation.
4. Keep research reproducibility.
5. Prioritize academic-quality implementations.
6. Prioritize outputs suitable for publication and thesis writing.
