# TinyGS QoS

A machine learning tool for predicting satellite transmission quality of service (QoS) probabilities in the TinyGS network. This project provides a Streamlit web application and utilities for analyzing LoRa satellite transmission success rates based on orbital parameters and radio configuration.
https://tinygsqos.streamlit.app/

## Overview

TinyGS QoS uses machine learning (Positive-Unlabeled learning) to predict the probability of successful satellite transmissions based on:

- Satellite position (latitude, longitude, altitude)
- LoRa radio parameters (spreading factor, bandwidth, coding rate)
- Antenna configuration (gain)
- Geometric factors (elevation angle, distance to nearest ground station)

## Features

- Interactive Streamlit web interface for transmission probability visualization
- Global coverage heatmaps showing transmission success probability
- Support for custom satellite orbital parameters
- Batch prediction capabilities
- Sun-Synchronous Orbit (SSO) simulation tools
- Pre-trained ML model using scikit-learn

## Installation

### Requirements

- Python 3.12 or higher
- Poetry (recommended) or pip

### Using Poetry

```bash
poetry install
```

### Using pip

```bash
pip install .
```

Note: This installs dependencies from `pyproject.toml`. For development, you may use `pip install -e .` for editable mode.

## Usage

### Running the Web Application

Launch the Streamlit interface:

```bash
streamlit run main.py
```

The application provides:
- Interactive parameter controls for SF, BW, CR, antenna gain, and altitude
- Validation for valid LoRa parameter combinations
- Global transmission probability heatmaps
- Overlay of actual transmission data

### Using the TransmissionPredictor Class

```python
from main import TransmissionPredictor

# Initialize predictor
predictor = TransmissionPredictor()

# Single prediction
prob = predictor.predict(
    sat_alt=600.0,      # Satellite altitude in km
    sf=10,              # Spreading factor
    bw=125.0,           # Bandwidth in kHz
    cr=5,               # Coding rate
    min_gain=5.0,       # Antenna gain in dB
    el=45.0,            # Elevation angle in degrees
    distance_to_station=500.0  # Distance to nearest station in km
)

# Batch prediction
import pandas as pd
predictions = predictor.predict_batch(dataframe)
```

### Generating Test Samples

```python
from utils.tiny_utils import TestSample, SSOOrbitTestSample

# Random global samples
X_test = TestSample(
    n_samples=10000,
    rand_lat=True,
    sf=[10],
    bw=[125.0],
    cr=[5],
    gain=[5.0],
    alt=600.0
)

# Sun-Synchronous Orbit samples
X_orbit = SSOOrbitTestSample(
    altitude=600.0,    # km
    sf=10,
    bw=125.0,
    gain=5.0,
    raan=0.0,         # Right Ascension of Ascending Node
    time_step=1.0     # seconds
)
```

## Data Files

The project requires the following data files in the `data/` directory:

- `PU_optuna_SDG_log_loss_12_7_v2.joblib` - Pre-trained ML model
- `packet_features.parquet` - Historical packet transmission data
- `station_locations.parquet` - TinyGS ground station locations
- `kdtree_stations.joblib` - KDTree for fast nearest station lookup
- `satellite_min_gain.parquet` - Satellite-specific minimum gain data
- `kde_satPosAlt.joblib` - Kernel Density Estimator for altitude distribution

## Valid LoRa Parameter Combinations

The model supports the following (Bandwidth, Spreading Factor) pairs based on actual TinyGS data:

| Bandwidth (kHz) | Spreading Factor |
|----------------|------------------|
| 125.0          | 7                |
| 62.5           | 8                |
| 125.0          | 8                |
| 125.0          | 9                |
| 500.0          | 9                |
| 125.0          | 10               |
| 250.0          | 10               |
| 125.0          | 11               |

## Model Details

The prediction model uses:
- Positive-Unlabeled (PU) learning framework
- Modified F1 score metric optimized for PU learning
- Features: satellite altitude, spreading factor, bandwidth, coding rate, elevation angle, distance to station, antenna gain
- Training via Optuna hyperparameter optimization