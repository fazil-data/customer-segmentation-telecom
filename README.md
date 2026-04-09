# Customer Segmentation for Telecom Analytics

This project demonstrates a behavioural customer segmentation pipeline built using Python and scikit-learn.  
It creates customer-level features from six months of usage data, applies KMeans clustering, profiles the resulting clusters, and maps them into higher-level macro segments for business interpretation.

## Project Objective

The goal of this project is to group customers into meaningful behavioural segments using historical activity patterns such as:

- ARPU
- data usage
- voice usage
- activity consistency
- recent activity streak
- recency of last activity
- trend in usage and revenue behaviour

These segments can support use cases such as:

- retention targeting
- value-based customer strategies
- campaign planning
- customer lifecycle analysis

## Project Structure

customer-segmentation-telecom/
- `src/feature_engineering.py` → builds behavioural features
- `src/clustering.py` → runs KMeans and profiles clusters
- `src/cluster_annotation.py` → adds percentile-based interpretation
- `src/macro_mapping.py` → maps micro-clusters into macro segments
- `run_pipeline.py` → runs the full workflow
- `data/sample_customer_usage_6m.csv` → sample anonymised input
- `outputs/` → generated outputs

## Input Data

The pipeline expects customer-level data with six months of activity in a wide format.

Example input columns:

- `customer_id`
- `m1_arpu` to `m6_arpu`
- `m1_data_usage` to `m6_data_usage`
- `m1_voice_usage` to `m6_voice_usage`
- `m1_active` to `m6_active`

## Features Engineered

The following features are derived from the six-month history:

- `mean_arpu_active`
- `mean_data_usage_active`
- `mean_voice_usage_active`
- `active_ratio`
- `std_data_usage_active`
- `std_arpu_active`
- `data_trend`
- `arpu_trend`
- `recent_active_streak`
- `months_since_last_activity`

## Methodology

### 1. Feature Engineering
Behavioural features are calculated from six months of customer activity.

### 2. Preprocessing
Selected features are clipped using percentile thresholds to reduce outlier influence and then standardised.

### 3. Clustering
KMeans clustering is used to create customer micro-segments.

### 4. Cluster Profiling
Each cluster is profiled using summary statistics such as mean, median, and quantiles.

### 5. Macro Segment Mapping
Micro-clusters are grouped into business-friendly macro segments such as:

- Premium
- Premium - At Risk
- Data-First
- Mid Value
- Price Sensitive / Low Value
- Dormant / Inactive

## How to Run

```bash
pip install -r requirements.txt
python run_pipeline.py
