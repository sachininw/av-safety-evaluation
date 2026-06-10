# AV Safety Evaluation Framework

A data science framework for evaluating autonomous vehicle performance using the **Waymo Open Dataset v2**. Implements the core statistical and ML methods used by AV safety teams to assess real-world driving quality, detect rare events, and validate simulation fidelity.

---

## Project Structure

```
av-safety-evaluation/
├── src/
│   ├── data/
│   │   ├── waymo_loader.py       # Waymo v2 parquet loader + trajectory extraction
│   │   └── download.py           # GCS download script
│   ├── metrics/
│   │   ├── safety_metrics.py     # TTC, DRAC, proximity, jerk, speed deviation
│   │   └── evaluation_framework.py  # Per-segment + dataset-level reporting
│   ├── rare_events/
│   │   ├── evt_analysis.py       # GEV (block maxima) and GPD (POT) fitting
│   │   └── rare_event_detector.py   # Near-miss, hard braking, swerve detection
│   ├── anomaly_detection/
│   │   ├── statistical_detector.py  # Z-score, CUSUM, Mahalanobis distance
│   │   └── ml_detector.py        # Isolation Forest, One-Class SVM
│   ├── simulation/
│   │   └── quality_metrics.py    # KS test, JSD, Wasserstein, MMD, coverage
│   └── visualization/
│       └── plots.py              # All matplotlib/seaborn plots
├── notebooks/
│   ├── 01_data_loading_and_exploration.ipynb
│   ├── 02_safety_metrics.ipynb
│   ├── 03_rare_event_estimation.ipynb
│   ├── 04_anomaly_detection.ipynb
│   └── 05_simulation_quality_assessment.ipynb
├── data/                         # Downloaded Waymo data (gitignored)
├── outputs/
│   ├── plots/
│   └── reports/
└── requirements.txt
```

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Authenticate with Google Cloud

The Waymo Open Dataset v2 is hosted on Google Cloud Storage.

**First**, accept the Terms of Service:
> https://waymo.com/open/licensing/

**Then**, authenticate your Google account:

```bash
gcloud auth login
```

### 3. Download a sample of the dataset

```bash
python src/data/download.py --split validation --num_segments 5
```

This downloads ~5 validation segments (~50–100 MB) for `lidar_box` and `vehicle_pose` components into `data/validation/`.

For a full run use `--num_segments 50` or more.

---

## Notebooks

Run notebooks from the `notebooks/` directory:

```bash
cd notebooks
jupyter notebook
```

| Notebook | Description |
|----------|-------------|
| `01_data_loading_and_exploration` | Load Waymo v2 parquet files, explore agent types, extract trajectories |
| `02_safety_metrics` | Compute TTC, DRAC, jerk, speed deviation; generate per-segment evaluation reports |
| `03_rare_event_estimation` | Fit GEV and GPD distributions; estimate return levels for safety-critical TTC values |
| `04_anomaly_detection` | Run Z-score, CUSUM, Mahalanobis, Isolation Forest, and One-Class SVM detectors |
| `05_simulation_quality_assessment` | Compare real vs. simulated distributions using KS, JSD, Wasserstein, MMD, and coverage metrics |

---

## Key Methods

### Safety Metrics

| Metric | Description |
|--------|-------------|
| **TTC** (Time to Collision) | Projected time until two agents collide at current relative velocity |
| **DRAC** | Required deceleration to avoid rear-end collision with lead vehicle |
| **Proximity Score** | Distance to nearest neighbouring agent |
| **Jerk** | Derivative of acceleration; high jerk indicates abrupt manoeuvres |
| **Speed Deviation** | Excess or deficit relative to road speed limit |

### Rare Event Estimation (Extreme Value Theory)

- **Block Maxima / GEV**: Fits a Generalised Extreme Value distribution to minimum TTC per block. Estimates 1-in-N return levels well beyond observed data.
- **Peaks-Over-Threshold / GPD**: Fits a Generalised Pareto Distribution to exceedances of a calibrated threshold. More data-efficient than block maxima.

### Anomaly Detection

| Method | Type | Strength |
|--------|------|----------|
| Z-score | Statistical | Fast, interpretable, per-feature |
| CUSUM | Statistical | Detects sustained signal shifts |
| Mahalanobis | Statistical | Joint multivariate outlier detection |
| Isolation Forest | ML | High-dimensional, non-parametric |
| One-Class SVM | ML | Tight normal-region boundary |

### Simulation Quality

| Metric | Measures |
|--------|----------|
| KS statistic | Marginal distribution difference per feature |
| Jensen-Shannon Divergence | Symmetric, bounded [0,1] distribution distance |
| Wasserstein distance | Earth-mover's distance (interpretable units) |
| MMD | Joint multivariate kernel-based distribution test |
| Coverage score | Fraction of real-data quantile bins represented in simulation |

---

## Data Format

This project uses the **Waymo Open Dataset v2** (Apache Parquet format), which does not require TensorFlow. Key components:

- **`lidar_box`**: Per-frame 3D bounding boxes for all detected agents (vehicles, pedestrians, cyclists). Contains position, velocity, heading, dimensions.
- **`vehicle_pose`**: Ego-vehicle world pose per frame.

---

## PhD-Level Analysis Modules

| Module | Method | Safety Question Answered |
|--------|--------|--------------------------|
| `src/causal/causal_safety.py` | Propensity Score Matching, IPW | Does high traffic density *cause* more near-misses? |
| `src/causal/causal_safety.py` | Granger Causality | Does acceleration Granger-cause TTC drops? |
| `src/causal/causal_safety.py` | Structural Counterfactual | What would TTC be if speed were 3 m/s lower? |
| `src/causal/causal_discovery.py` | PC Algorithm (DAG learning) | What is the causal structure among driving features? |
| `src/uncertainty/conformal_prediction.py` | Split / Mondrian Conformal | What are distribution-free safety metric bounds? |
| `src/uncertainty/conformal_prediction.py` | Conformal Risk Control | What threshold bounds the near-miss rate at 10%? |
| `src/bayesian/hierarchical_model.py` | Hierarchical Bayesian MCMC | What is the per-segment risk with proper uncertainty? |
| `src/survival/survival_analysis.py` | Cox PH, Kaplan-Meier | Which conditions predict *earlier* rare-event occurrence? |
| `src/point_processes/hawkes_process.py` | Hawkes Process (MLE) | Are safety events self-exciting / contagious? |

### PhD Notebooks

| Notebook | Methods |
|----------|---------|
| `06_causal_safety_analysis` | PSM, IPW ATE, Granger causality, counterfactuals, PC algorithm DAG |
| `07_conformal_prediction` | Split conformal, Mondrian conditional, Conformal Risk Control |
| `08_bayesian_hierarchical` | MH-MCMC hierarchical model, shrinkage, posterior risk, PPC |
| `09_survival_and_hawkes` | Kaplan-Meier, log-rank test, Cox PH, Hawkes process MLE + GoF |

---

## Results Interpretation

After running all notebooks you will have:

1. **Per-segment safety scorecards** — near-miss rate, jerk violation rate, speed excess rate, proximity warning rate.
2. **EVT return-level table** — estimated minimum TTC at 1-in-10, 1-in-100, 1-in-1000 return periods.
3. **Anomaly timeline** — timestamped list of detected anomalous driving frames with detector labels.
4. **Simulation quality report** — per-feature JSD, Wasserstein, match score, and overall coverage.
5. **Causal graph (DAG)** — learned causal structure among speed, acceleration, jerk, TTC.
6. **Causal ATE estimates** — PSM and IPW estimates of the effect of traffic density on safety.
7. **Conformal prediction intervals** — distribution-free safety bounds with formal coverage guarantees.
8. **Bayesian risk ranking** — per-segment posterior P(mean TTC < 3 s) with credible intervals.
9. **Survival curves and hazard ratios** — which conditions shorten time-to-rare-event.
10. **Hawkes process parameters** — background rate, branching ratio, decay of event clustering.

---

## Dataset Citation

```
@inproceedings{waymo_open_dataset_2020,
  title   = {Scalability in Perception for Autonomous Driving: Waymo Open Dataset},
  author  = {Sun, Pei and Kretzschmar, Henrik and Dotiwalla, Xerxes and others},
  booktitle = {CVPR},
  year    = {2020}
}
```
