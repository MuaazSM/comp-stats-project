# Mumbai Ward Clustering Analysis

clustering analysis of mumbai wards based on economic activity and workforce structure using economic census 2013 data.

## Overview

this project uses k-means clustering to identify patterns in mumbai's economic landscape across different wards. it analyzes establishment distribution, workforce composition, sector participation, and business characteristics.

## Features

- **data aggregation**: ward-level aggregation of economic census data
- **feature engineering**: compute 14+ features including sector distribution, ownership patterns, workforce demographics
- **train/test/validation split**: proper data splitting for model evaluation
- **model persistence**: save and load trained models with metadata
- **clustering analysis**: k-means clustering with automatic k selection support
- **evaluation metrics**: elbow method, silhouette score, davies-bouldin index
- **interactive visualizations**: 
  - elbow curve for optimal k selection
  - silhouette score comparison
  - 2D PCA scatter plot with cluster assignments
  - cluster profile bar charts
  - radar charts for multi-feature comparison
- **export functionality**: download clustered results as csv

## Project structure

```
comp-stats-project/
├── app.py                      # main streamlit dashboard
├── train.py                    # model training script
├── validate.py                 # model validation script
├── data/
│   └── mumbai-suburban.csv     # economic census data
├── models/                     # saved models directory
├── src/
│   ├── features.py             # feature aggregation and preprocessing
│   ├── modeling.py             # clustering models and metrics
│   └── viz.py                  # visualization helpers
├── requirements.txt            # python dependencies
└── README.md                   # project documentation
```

## Getting started

### prerequisites

- python 3.8 or higher
- pip package manager

### Installation

1. clone or download the repository:
```bash
cd comp-stats-project
```

2. install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### 1. train the model (recommended first step)

train a clustering model with train/test/validation split:

```bash
# train with default parameters (k=4, 70% train, 10% val, 20% test)
python train.py

# train with custom parameters
python train.py --k 5 --test-size 0.2 --val-size 0.1 --max-k-eval 10
```

**training output:**
- saves trained model as `.pkl` file in `models/` directory
- saves scaler for feature normalization
- saves metadata with training metrics
- saves predictions for train/val/test sets
- displays comprehensive evaluation metrics

**parameters:**
- `--k`: number of clusters (default: 4)
- `--test-size`: test set proportion (default: 0.2)
- `--val-size`: validation set proportion (default: 0.1)
- `--max-k-eval`: max k for evaluation (default: 10)
- `--random-state`: random seed (default: 42)

### 2. validate the model

validate a trained model and compare performance:

```bash
# validate latest model
python validate.py

# validate specific model
python validate.py --model models/kmeans_k4_timestamp.pkl --scaler models/scaler_timestamp.pkl

# compare all saved models
python validate.py --compare
```

**validation output:**
- loads saved model and evaluates on full dataset
- compares performance across train/val/test sets
- displays cluster distribution and statistics
- shows model metadata and training information

### 3. run the dashboard

launch the interactive streamlit dashboard:

```bash
streamlit run app.py
```

the app will open in your browser at `http://localhost:8501`

**dashboard features:**
- load pre-trained models or train new ones on-the-fly
- interactive parameter adjustment
- real-time visualizations
- export results as CSV

## Data features

the analysis computes these ward-level features:

**establishment metrics:**
- total establishments
- total workers
- average workers per establishment

**sector distribution:**
- % primary sector (agriculture, mining)
- % secondary sector (manufacturing, construction)
- % tertiary sector (services, trade)

**ownership patterns:**
- % government
- % private
- % cooperative

**business characteristics:**
- % permanent premises
- % temporary premises
- % self-financed
- % assisted financing

**workforce demographics:**
- % female workers

## 🎨 dashboard tabs

1. **data**: view raw ward-level data and summary statistics
2. **model**: evaluate optimal k using elbow method and silhouette scores
3. **results**: visualize clusters with pca plots and feature profiles
4. **insights**: explore cluster characteristics and download results

## Customization

adjust clustering parameters in the sidebar:
- **number of clusters (k)**: set the desired number of clusters (2-8)
- **max k for evaluation**: range for elbow/silhouette evaluation (5-15)

## 📦 dependencies

- **streamlit**: interactive web dashboard
- **pandas**: data manipulation and aggregation
- **numpy**: numerical computations
- **scikit-learn**: clustering algorithms and metrics
- **plotly**: interactive visualizations
- **joblib**: model persistence
- **tabulate**: formatted table output for validation

## 📈 interpretation guide

**silhouette score**: 
- ranges from -1 to 1
- > 0.5 indicates good clustering
- higher values mean better-defined clusters

**davies-bouldin index**:
- lower values indicate better clustering
- measures cluster separation and compactness

**pca visualization**:
- shows clusters in 2D space
- helps identify cluster overlap and separation

## 🤝 contributing

this is an academic project for computational statistics. feel free to fork and adapt for your own analysis.

## 📝 license

this project is for educational purposes.

## 📧 contact

for questions or feedback about this analysis, please reach out through the project repository.