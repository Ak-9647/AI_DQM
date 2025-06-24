# AI-Powered Data Quality Monitor

A comprehensive system that automatically ingests datasets, profiles them, uses machine learning to flag potential anomalies, and presents findings on a monitoring dashboard.

## 🚀 Features

- **Automated Data Ingestion**: CSV file processing with data profiling
- **AI-Powered Anomaly Detection**: Uses isolation forests and DBSCAN clustering to identify data quality issues
- **Interactive Dashboard**: Real-time monitoring with Streamlit
- **Data Pipeline**: Modern data stack with dbt and DuckDB
- **Quality Scores**: Track data quality metrics over time

## 🛠️ Tech Stack

- **Data Pipeline**: dbt + DuckDB
- **Machine Learning**: Scikit-learn (Isolation Forest, DBSCAN)
- **Visualization**: Streamlit + Plotly
- **Data Processing**: Pandas + NumPy

## 📁 Project Structure

```
Proj1/
├── data/
│   ├── raw/                 # Raw CSV files
│   ├── processed/          # Cleaned data
│   └── sample/            # Sample datasets
├── dbt_project/
│   ├── models/            # dbt data models
│   ├── macros/           # dbt macros
│   └── dbt_project.yml   # dbt configuration
├── src/
│   ├── anomaly_detection/ # ML algorithms
│   ├── data_profiling/   # Data quality profiling
│   └── pipeline/         # Data pipeline scripts
├── dashboard/
│   └── streamlit_app.py  # Main dashboard
├── config/
│   └── config.yaml       # Configuration files
└── requirements.txt
```

## 🏃‍♂️ Quick Start

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up dbt**:
   ```bash
   cd dbt_project
   dbt deps
   dbt run
   ```

3. **Run the Dashboard**:
   ```bash
   streamlit run dashboard/streamlit_app.py
   ```

4. **Upload Data**: Use the dashboard to upload CSV files and monitor data quality

## 📊 What It Does

1. **Data Ingestion**: Automatically processes uploaded CSV files
2. **Data Profiling**: Analyzes data types, missing values, distributions
3. **Anomaly Detection**: Identifies outliers using ML algorithms
4. **Quality Scoring**: Calculates overall data quality scores
5. **Monitoring**: Tracks quality trends over time
6. **Alerting**: Flags potential data issues for investigation

## 🎯 Business Value

- **Cost Reduction**: Early detection of data quality issues
- **Automated Monitoring**: Reduces manual data quality checks
- **Root Cause Analysis**: Helps identify sources of data problems
- **Trend Analysis**: Tracks data quality improvements over time

## 🔧 Configuration

Edit `config/config.yaml` to customize:
- Anomaly detection thresholds
- Data quality metrics
- Dashboard settings 