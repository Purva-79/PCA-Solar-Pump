# ☀️ SolarPCA Monitor

### PCA-Based Compression of Multivariate Solar-Pump Sensor Streams for Mining Site Operational Monitoring

SolarPCA Monitor is a data-analysis and visualization project designed to study **multivariate sensor streams from a solar-powered pump system at a mining site**. The project uses **Principal Component Analysis (PCA)** to reduce nine sensor dimensions into a smaller set of principal components while preserving the major patterns in the data.

The project contains **two interfaces**:

- 🌐 **Web Dashboard** — a browser-based HTML/CSS/JavaScript dashboard using Chart.js.
- 📊 **Streamlit Dashboard** — an interactive Python application using Scikit-learn, Plotly and Streamlit.

---

## 🎯 Project Objective

Mining-site solar pumping systems can generate several sensor readings at the same time. Monitoring every variable independently can make it difficult to identify relationships, operating patterns and unusual conditions.

This project applies PCA to:

- 📡 Analyze **9 multivariate sensor channels**
- 🔗 Identify relationships between correlated sensor variables
- 🗜️ Reduce the dimensionality of the sensor data
- 📊 Visualize the principal components and their variance
- 🚦 Examine normal, strained and anomaly operating regimes
- 📈 Monitor PCA scores and Hotelling's **T² statistic**
- 📋 Explore and export sensor and PCA data

The overall workflow is:

```text
Sensor Data
     ↓
Data Generation / Dataset
     ↓
Feature Standardization
     ↓
PCA Transformation
     ↓
Principal Component Scores
     ↓
Variance + Loadings + Visualizations
     ↓
Operating Regime Analysis
     ↓
Alerts / Data Export
```

---

## 📡 Sensor Channels

The project works with **9 sensor variables** grouped into three operational subsystems.

### ☀️ Solar Output Subsystem

| Sensor | Unit | Description |
|---|---|---|
| Irradiance | W/m² | Solar irradiance received by the photovoltaic system |
| PV Voltage | V | Photovoltaic output voltage |
| PV Current | A | Photovoltaic output current |

### ⚙️ Pump Load Subsystem

| Sensor | Unit | Description |
|---|---|---|
| Motor RPM | RPM | Pump motor rotational speed |
| Flow Rate | L/min | Water flow through the pumping system |
| Vibration | g | Pump/motor vibration level |

### 💧 Water Level Subsystem

| Sensor | Unit | Description |
|---|---|---|
| Tank Level | % | Water tank/reservoir level |
| Pressure | kPa | Hydraulic pressure |
| Turbidity | NTU | Water turbidity |

---

## 🔬 PCA Methodology

The PCA pipeline is implemented in both project versions.

### 1. Feature Preparation

The nine sensor channels are selected as the PCA input matrix.

When standardization is enabled, the Streamlit application applies a **z-score StandardScaler** before PCA:

```text
X → StandardScaler → X_scaled → PCA
```

The browser implementation also performs feature standardization before calculating PCA.

### 2. Principal Component Analysis

PCA transforms the original sensor variables into a smaller set of orthogonal principal components.

For example:

```text
9 original sensor features
          ↓
      PCA
          ↓
PC1  PC2  PC3  ...
```

The Streamlit dashboard allows the number of retained components to be selected from **1 to 9**.

### 3. Explained Variance

The application displays:

- Explained variance of individual components
- Cumulative explained variance
- Eigenvalue spectrum
- Scree plot / elbow analysis

These views help evaluate how much information is retained after dimensionality reduction.

### 4. Component Loadings

PCA component loadings show the contribution of each sensor to each principal component.

The dashboards visualize the absolute loading values to make it easier to compare the influence of the nine sensor channels.

### 5. Reconstruction Error

The Streamlit application also calculates the mean squared reconstruction error after projecting the standardized data into the selected PCA space and reconstructing it.

---

## 🚦 Operating Regimes

The dataset contains three operating regimes:

| Regime | Meaning |
|---|---|
| ✅ Normal | Stable operating condition |
| ⚠️ Strained | Elevated operating stress |
| 🚨 Anomaly | Critical deviation from the normal operating pattern |

The synthetic data generator creates these regimes at different sections of the sample sequence and modifies sensor values accordingly.

For the included `sensor_data.csv` dataset:

- **200 samples**
- **158 normal samples**
- **30 strained samples**
- **12 anomaly samples**

The dashboards use these labels for regime visualization and operational analysis.

> ℹ️ The included regime labels are part of the generated/supplied dataset. PCA and Hotelling's T² are used to analyze and visualize the resulting operating behavior.

---

## 📊 Hotelling's T²

The Streamlit application calculates the **Hotelling T² statistic** from the PCA scores.

It provides:

- T² value for each sample
- 95th-percentile reference level
- 99th-percentile reference level
- T² visualization alongside operating regimes

Higher T² values indicate observations that are farther from the multivariate center of the PCA score space.

---

## 📈 Dashboard Features

### 🌐 Web Dashboard

The browser-based version provides:

- ☀️ Hero/overview section
- 📡 Sensor stream visualizations
- 🔗 Sensor correlation matrix
- 🔬 PCA engine visualization
- 📊 Explained variance
- 📐 Eigenvalue and scree analysis
- 🔢 PCA component loadings
- 🗺️ PCA scatter plots
- 📈 Principal component time series
- 🚦 Operating regime visualization
- 🔔 Operational alert log
- 📋 Interactive data table
- ⬇️ Data export functionality

The JavaScript application contains its own PCA implementation and does not require a Python backend.

### 📊 Streamlit Dashboard

The Streamlit version provides five main tabs:

1. **☀️ Sensor Streams**
   - Solar output streams
   - Pump load streams
   - Water-level streams
   - Correlation matrix

2. **🔬 PCA Engine**
   - Explained variance
   - Eigenvalues
   - Scree plot
   - Component loadings
   - Dimensionality reduction
   - Reconstruction error

3. **📊 Visualizations**
   - PC1 vs PC2
   - PC1 vs PC3
   - PC2 vs PC3
   - PCA score time series

4. **🚦 Op Regimes**
   - Normal / strained / anomaly summary
   - Regime distribution
   - Hotelling T²
   - Regime timeline
   - Operational alert log

5. **📋 Data Table**
   - Raw sensor data
   - PCA scores
   - T² statistics
   - Regime filtering
   - Search
   - CSV/JSON export
   - PCA loading export
   - Descriptive statistics

---

## 🗂️ Project Structure

```text
SolarPCA-Project/
│
├── solar-pca-monitor/
│   ├── index.html
│   ├── app.js
│   ├── style.css
│   └── sensor_data.csv
│
└── solar-pca-streamlit/
    ├── app.py
    └── requirements.txt
```

### `solar-pca-monitor/`

The frontend/browser implementation.

- `index.html` → Dashboard structure and UI
- `style.css` → Dashboard styling, layout, responsive design and theme
- `app.js` → Sensor generation, PCA calculations, charts, regimes, alerts, table and exports
- `sensor_data.csv` → Included sensor dataset

### `solar-pca-streamlit/`

The Python interactive implementation.

- `app.py` → Complete Streamlit dashboard, data generation, PCA processing, visualizations and exports
- `requirements.txt` → Python dependencies

---

## 🛠️ Technologies Used

### Web Version

- HTML5
- CSS3
- JavaScript
- Chart.js
- Chart.js Annotation Plugin
- Google Fonts: Outfit + JetBrains Mono

### Python Version

- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Plotly
- SciPy

---

## 🚀 Running the Web Version

The web version is a client-side application.

### Option 1 — Open directly

Open:

```text
solar-pca-monitor/index.html
```

in a modern web browser.

### Option 2 — Use a local server

From inside the `solar-pca-monitor` folder:

```bash
python -m http.server 8000
```

Then open:

```text
http://localhost:8000
```

The browser version loads Chart.js and the annotation plugin from CDN.

---

## 🐍 Running the Streamlit Version

### 1. Open the project folder

```bash
cd solar-pca-streamlit
```

### 2. Create a virtual environment (recommended)

Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

macOS/Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Start the application

```bash
streamlit run app.py
```

Streamlit will provide a local URL in the terminal, normally:

```text
http://localhost:8501
```

---

## ⚙️ Streamlit Controls

The sidebar allows the user to change the analysis interactively.

### Data Settings

- **Number of Samples:** 50–500
- **Noise Level:** 0.0–1.0
- **Random Seed:** Used for reproducible data generation

### PCA Settings

- **Principal Components:** 1–9
- **Standardize Features:** Enable/disable z-score standardization

### Display Settings

- Show all PC comparisons
- Select the number of table rows per page
- Regenerate the dataset

---

## 📁 Data Export

The Streamlit dashboard supports exporting:

### CSV

```text
solar_pump_pca_data.csv
```

Contains sensor readings, PCA scores, T² statistics and regime labels.

### JSON

```text
solar_pump_pca_data.json
```

Contains metadata and the combined sensor/PCA dataset.

### PCA Loadings

```text
pca_loadings.csv
```

Contains the PCA component loading matrix.

---

## 🧪 Dataset

The included dataset contains **200 samples** and the following fields:

```text
Sample
Irradiance_W_m2
PV_Voltage_V
PV_Current_A
Motor_RPM
Flow_Rate_L_min
Vibration_g
Tank_Level_pct
Pressure_kPa
Turbidity_NTU
Regime
```

The Python application can also generate new synthetic sensor streams using the selected sample count, noise level and random seed.

---

## 💡 Why PCA?

A mining-site solar pumping system can produce multiple correlated sensor streams simultaneously. Instead of analyzing all nine variables independently, PCA provides a lower-dimensional representation of the data.

This can help with:

- 📉 Dimensionality reduction
- 🔍 Pattern discovery
- 📊 Multivariate visualization
- 🗜️ Sensor-stream compression
- 🚦 Operating-regime analysis
- 🚨 Detection-oriented monitoring using PCA statistics

The project demonstrates how multivariate sensor information can be represented using fewer dimensions while retaining the major variance in the original data.

---

## 🔮 Possible Future Improvements

The current project provides a synthetic-data-based PCA monitoring environment. It could be extended with:

- 🔌 Real-time sensor/IoT data ingestion
- 📡 MQTT or other telemetry integration
- 🗄️ Database-backed historical storage
- 🤖 Automated anomaly detection models
- 📈 Predictive maintenance
- 🔔 Real-time notifications
- 🧠 Comparison of PCA with other dimensionality-reduction methods
- ☁️ Cloud deployment
- 🏭 Integration with actual mining-site sensor systems

---

## 👩‍💻 Project

**Project Title:**  
**PCA-Based Compression of Multivariate Solar-Pump Sensor Streams for Mining Site Operational Monitoring**

The project combines **multivariate sensor analysis, PCA-based dimensionality reduction, interactive visualization and operational monitoring** into a single dashboard-oriented system.

☀️ **Solar Data + ⚙️ Pump Data + 💧 Water Data → 🔬 PCA → 📊 Monitoring → 🚦 Operational Insight**
