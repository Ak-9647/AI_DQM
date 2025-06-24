"""
Streamlit Dashboard for AI-Powered Data Quality Monitor
Interactive dashboard for monitoring data quality and anomalies
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from pipeline.data_pipeline import DataQualityPipeline

# Page configuration
st.set_page_config(
    page_title="AI-Powered Data Quality Monitor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 0.5rem;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .quality-excellent { background-color: #28a745; }
    .quality-good { background-color: #17a2b8; }
    .quality-fair { background-color: #ffc107; }
    .quality-poor { background-color: #dc3545; }
</style>
""", unsafe_allow_html=True)

# Initialize pipeline
@st.cache_resource
def get_pipeline():
    return DataQualityPipeline()

def main():
    # Header
    st.markdown('<h1 class="main-header">🤖 AI-Powered Data Quality Monitor</h1>', unsafe_allow_html=True)
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        ["Dashboard", "Data Upload", "Anomaly Detection", "Data Profiling", "Settings"]
    )
    
    # Initialize pipeline
    pipeline = get_pipeline()
    
    if page == "Dashboard":
        show_dashboard(pipeline)
    elif page == "Data Upload":
        show_data_upload(pipeline)
    elif page == "Anomaly Detection":
        show_anomaly_detection(pipeline)
    elif page == "Data Profiling":
        show_data_profiling(pipeline)
    elif page == "Settings":
        show_settings(pipeline)

def show_dashboard(pipeline):
    """Main dashboard with overview metrics and charts"""
    st.header("📊 Data Quality Dashboard")
    
    # Get monitoring data
    monitoring_data = pipeline.get_monitoring_data()
    
    if monitoring_data.empty:
        st.info("No monitoring data available. Please upload and process a dataset first.")
        return
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        latest_score = monitoring_data['quality_score'].iloc[0] if len(monitoring_data) > 0 else 0
        st.metric("Overall Quality Score", f"{latest_score:.2f}", f"{latest_score*100:.1f}%")
    
    with col2:
        latest_grade = monitoring_data['quality_grade'].iloc[0] if len(monitoring_data) > 0 else 'F'
        st.metric("Quality Grade", latest_grade)
    
    with col3:
        total_anomalies = monitoring_data['num_anomalies'].iloc[0] if len(monitoring_data) > 0 else 0
        st.metric("Anomalies Detected", total_anomalies)
    
    with col4:
        tables_monitored = monitoring_data['table_name'].nunique()
        st.metric("Tables Monitored", tables_monitored)
    
    # Quality trends
    st.subheader("📈 Quality Trends")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Quality score over time
        fig_quality = px.line(
            monitoring_data.sort_values('timestamp'),
            x='timestamp',
            y='quality_score',
            title='Quality Score Over Time',
            color='table_name' if 'table_name' in monitoring_data.columns else None
        )
        fig_quality.update_layout(height=400)
        st.plotly_chart(fig_quality, use_container_width=True)
    
    with col2:
        # Anomaly rate over time
        fig_anomalies = px.line(
            monitoring_data.sort_values('timestamp'),
            x='timestamp',
            y='anomaly_rate',
            title='Anomaly Rate Over Time',
            color='table_name' if 'table_name' in monitoring_data.columns else None
        )
        fig_anomalies.update_layout(height=400)
        st.plotly_chart(fig_anomalies, use_container_width=True)
    
    # Quality dimensions breakdown
    st.subheader("🎯 Quality Dimensions")
    
    if len(monitoring_data) > 0:
        latest_record = monitoring_data.iloc[0]
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            completeness = latest_record.get('completeness', 0)
            st.metric("Completeness", f"{completeness:.2%}")
        
        with col2:
            validity = latest_record.get('validity', 0)
            st.metric("Validity", f"{validity:.2%}")
        
        with col3:
            uniqueness = latest_record.get('uniqueness', 0)
            st.metric("Uniqueness", f"{uniqueness:.2%}")
    
    # Recent data quality issues
    st.subheader("⚠️ Recent Issues")
    
    issues = []
    for _, row in monitoring_data.head(5).iterrows():
        if row['quality_score'] < 0.7:
            issues.append(f"Low quality score ({row['quality_score']:.2f}) in {row['table_name']}")
        if row['anomaly_rate'] > 0.1:
            issues.append(f"High anomaly rate ({row['anomaly_rate']:.2%}) in {row['table_name']}")
    
    if issues:
        for issue in issues[:5]:  # Show top 5 issues
            st.warning(issue)
    else:
        st.success("No major data quality issues detected!")

def show_data_upload(pipeline):
    """Data upload and processing interface"""
    st.header("📤 Data Upload & Processing")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        # Display file details
        st.subheader("File Details")
        col1, col2 = st.columns(2)
        
        with col1:
            st.info(f"**Filename:** {uploaded_file.name}")
            st.info(f"**Size:** {uploaded_file.size:,} bytes")
        
        # Preview data
        df = pd.read_csv(uploaded_file)
        st.subheader("Data Preview")
        st.dataframe(df.head(10))
        
        # Processing options
        st.subheader("Processing Options")
        
        col1, col2 = st.columns(2)
        with col1:
            table_name = st.text_input("Table Name (optional)", value=f"dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        with col2:
            run_full_pipeline = st.checkbox("Run Full Pipeline", value=True)
        
        if st.button("Process Data", type="primary"):
            with st.spinner("Processing data..."):
                # Save uploaded file temporarily
                temp_path = f"data/raw/{uploaded_file.name}"
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Run pipeline
                if run_full_pipeline:
                    results = pipeline.run_full_pipeline(temp_path, table_name)
                    
                    # Display results
                    st.subheader("Processing Results")
                    
                    for step_name, step_result in results.get('steps', {}).items():
                        if step_result['success']:
                            st.success(f"✅ {step_name.replace('_', ' ').title()}")
                        else:
                            st.error(f"❌ {step_name.replace('_', ' ').title()}")
                    
                    # Show quality summary if available
                    if 'profiling' in results['steps'] and results['steps']['profiling'].get('profile_summary'):
                        summary = results['steps']['profiling']['profile_summary']
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Quality Score", f"{summary.get('overall_quality_score', 0):.2f}")
                        with col2:
                            st.metric("Quality Grade", summary.get('quality_grade', 'F'))
                        with col3:
                            st.metric("Completeness", f"{summary.get('completeness', 0):.2%}")
                    
                    st.success("Data processing completed successfully!")
                else:
                    # Just ingest the data
                    success = pipeline.ingest_csv(temp_path, table_name)
                    if success:
                        st.success("Data ingested successfully!")
                    else:
                        st.error("Failed to ingest data.")

def show_anomaly_detection(pipeline):
    """Anomaly detection interface"""
    st.header("🔍 Anomaly Detection")
    
    # Get available tables
    tables = pipeline.get_table_list()
    
    if not tables:
        st.info("No tables available. Please upload data first.")
        return
    
    # Table selection
    selected_table = st.selectbox("Select table for anomaly detection", tables)
    
    if st.button("Detect Anomalies"):
        with st.spinner("Detecting anomalies..."):
            anomaly_results = pipeline.detect_anomalies(selected_table)
            
            if anomaly_results:
                st.subheader("Anomaly Detection Results")
                
                # Summary metrics
                col1, col2, col3 = st.columns(3)
                
                if 'isolation_forest' in anomaly_results:
                    with col1:
                        st.metric(
                            "Isolation Forest", 
                            f"{anomaly_results['isolation_forest']['num_anomalies']} anomalies",
                            f"{anomaly_results['isolation_forest']['anomaly_rate']:.2%}"
                        )
                
                if 'dbscan' in anomaly_results:
                    with col2:
                        st.metric(
                            "DBSCAN Clustering", 
                            f"{anomaly_results['dbscan']['num_anomalies']} anomalies",
                            f"{anomaly_results['dbscan']['anomaly_rate']:.2%}"
                        )
                
                if 'combined' in anomaly_results:
                    with col3:
                        st.metric(
                            "Combined Method", 
                            f"{anomaly_results['combined']['num_anomalies']} anomalies",
                            f"{anomaly_results['combined']['anomaly_rate']:.2%}"
                        )
                
                # Anomaly details
                st.subheader("Anomaly Details")
                
                # Try to load anomaly details
                details_path = f"data/anomalies/{selected_table}_anomaly_details.csv"
                if os.path.exists(details_path):
                    anomaly_details = pd.read_csv(details_path)
                    
                    # Filter to show only anomalies
                    if 'combined_anomaly' in anomaly_details.columns:
                        anomalous_data = anomaly_details[anomaly_details['combined_anomaly'] == True]
                        
                        if not anomalous_data.empty:
                            st.dataframe(anomalous_data)
                        else:
                            st.info("No anomalies detected in the data.")
                    else:
                        st.dataframe(anomaly_details)
                else:
                    st.info("Anomaly details file not found.")
            else:
                st.error("Failed to detect anomalies.")

def show_data_profiling(pipeline):
    """Data profiling interface"""
    st.header("📊 Data Profiling")
    
    # Get available tables
    tables = pipeline.get_table_list()
    
    if not tables:
        st.info("No tables available. Please upload data first.")
        return
    
    # Table selection
    selected_table = st.selectbox("Select table for profiling", tables)
    
    if st.button("Generate Profile"):
        with st.spinner("Generating data profile..."):
            profile = pipeline.profile_data(selected_table)
            
            if profile:
                st.subheader("Data Profile Results")
                
                # Basic info
                basic_info = profile.get('basic_info', {})
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Rows", basic_info.get('num_rows', 0))
                with col2:
                    st.metric("Columns", basic_info.get('num_columns', 0))
                with col3:
                    st.metric("Memory (MB)", f"{basic_info.get('memory_usage_mb', 0):.2f}")
                with col4:
                    quality_summary = profile.get('data_quality_summary', {})
                    st.metric("Quality Score", f"{quality_summary.get('overall_quality_score', 0):.2f}")
                
                # Column profiles
                st.subheader("Column Profiles")
                
                column_profiles = profile.get('column_profiles', {})
                
                if column_profiles:
                    # Create a summary dataframe
                    profile_summary = []
                    
                    for col_name, col_profile in column_profiles.items():
                        profile_summary.append({
                            'Column': col_name,
                            'Type': col_profile.get('data_type', ''),
                            'Non-Null %': f"{100 - col_profile.get('null_percentage', 0):.1f}%",
                            'Unique %': f"{col_profile.get('unique_percentage', 0):.1f}%",
                            'Quality Issues': len([k for k, v in col_profile.get('quality_flags', {}).items() if v])
                        })
                    
                    profile_df = pd.DataFrame(profile_summary)
                    st.dataframe(profile_df)
                    
                    # Detailed column view
                    st.subheader("Detailed Column Analysis")
                    selected_column = st.selectbox("Select column for detailed analysis", list(column_profiles.keys()))
                    
                    if selected_column:
                        col_details = column_profiles[selected_column]
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write("**Basic Statistics:**")
                            st.json({
                                'Count': col_details.get('count', 0),
                                'Non-null': col_details.get('non_null_count', 0),
                                'Unique': col_details.get('unique_count', 0),
                                'Null %': f"{col_details.get('null_percentage', 0):.2f}%"
                            })
                        
                        with col2:
                            st.write("**Quality Flags:**")
                            quality_flags = col_details.get('quality_flags', {})
                            for flag, value in quality_flags.items():
                                if value:
                                    st.warning(f"⚠️ {flag.replace('_', ' ').title()}")
                
                # Data quality summary
                st.subheader("Data Quality Summary")
                quality_summary = profile.get('data_quality_summary', {})
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Completeness", f"{quality_summary.get('completeness', 0):.2%}")
                with col2:
                    st.metric("Validity", f"{quality_summary.get('validity', 0):.2%}")
                with col3:
                    st.metric("Uniqueness", f"{quality_summary.get('uniqueness', 0):.2%}")
                
            else:
                st.error("Failed to generate data profile.")

def show_settings(pipeline):
    """Settings and configuration interface"""
    st.header("⚙️ Settings & Configuration")
    
    # Configuration display
    st.subheader("Current Configuration")
    
    try:
        with open("config/config.yaml", 'r') as f:
            config_content = f.read()
        
        st.code(config_content, language='yaml')
        
    except FileNotFoundError:
        st.error("Configuration file not found.")
    
    # Database status
    st.subheader("Database Status")
    
    try:
        tables = pipeline.get_table_list()
        monitoring_data = pipeline.get_monitoring_data()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Available Tables", len(tables))
            if tables:
                st.write("**Tables:**")
                for table in tables:
                    st.write(f"- {table}")
        
        with col2:
            st.metric("Monitoring Records", len(monitoring_data))
            if not monitoring_data.empty:
                st.write("**Latest Update:**")
                st.write(monitoring_data['timestamp'].iloc[0])
        
    except Exception as e:
        st.error(f"Error checking database status: {str(e)}")

if __name__ == "__main__":
    main() 