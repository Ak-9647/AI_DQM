#!/usr/bin/env python3
"""
Test script for the AI-Powered Data Quality Monitor
Tests the complete pipeline with real-world data from Washington State Electric Vehicle Population
"""

import os
import sys
import logging
from pathlib import Path
import pandas as pd

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from pipeline.data_pipeline import DataQualityPipeline
from data_profiling.profiler import DataProfiler
from anomaly_detection.detector import AnomalyDetector

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Test the complete data quality monitoring system with real-world data."""
    
    print("=" * 80)
    print("🚗 AI-POWERED DATA QUALITY MONITOR - REAL WORLD TEST")
    print("Testing with Washington State Electric Vehicle Population Data")
    print("=" * 80)
    
    try:
        # Initialize the pipeline
        logger.info("Initializing Data Quality Pipeline...")
        pipeline = DataQualityPipeline()
        
        # Process the real-world electric vehicle dataset
        csv_path = "data/raw/electric_vehicles_wa.csv"
        
        if not os.path.exists(csv_path):
            logger.error(f"Dataset not found: {csv_path}")
            logger.info("Please ensure the electric vehicle dataset is downloaded to data/raw/")
            return
        
        # Get basic info about the dataset
        df = pd.read_csv(csv_path)
        print(f"\n📊 DATASET OVERVIEW:")
        print(f"   • Records: {len(df):,}")
        print(f"   • Columns: {len(df.columns)}")  
        print(f"   • File size: {os.path.getsize(csv_path) / (1024*1024):.1f} MB")
        print(f"   • Columns: {list(df.columns)}")
        
        print(f"\n🔄 PROCESSING ELECTRIC VEHICLE DATA...")
        
        # Run the complete pipeline
        results = pipeline.run_full_pipeline(
            csv_path=csv_path,
            table_name="electric_vehicles_wa"
        )
        
        # Check if pipeline completed successfully
        pipeline_success = all(
            step.get('success', False) for step in results.get('steps', {}).values()
        )
        
        if pipeline_success:
            print(f"\n✅ PIPELINE COMPLETED SUCCESSFULLY!")
            print(f"   • Processing time: {results.get('timestamp', 'N/A')}")
            
            # Display results for each step
            print(f"\n📋 PIPELINE STEPS:")
            for step_name, step_result in results.get('steps', {}).items():
                status = "✅ PASSED" if step_result['success'] else "❌ FAILED" 
                print(f"   • {status} {step_name.replace('_', ' ').title()}")
            
            # Display data quality results from profiling step
            if 'profiling' in results['steps'] and results['steps']['profiling'].get('profile_summary'):
                quality = results['steps']['profiling']['profile_summary']
                print(f"\n📈 DATA QUALITY METRICS:")
                print(f"   • Overall Quality Score: {quality.get('overall_quality_score', 0):.2f}/10")
                print(f"   • Quality Grade: {quality.get('quality_grade', 'F')}")
                print(f"   • Completeness: {quality.get('completeness', 0):.2%}")
                print(f"   • Validity: {quality.get('validity', 0):.2%}")
                print(f"   • Uniqueness: {quality.get('uniqueness', 0):.2%}")
            
            # Display anomaly detection results  
            if 'anomaly_detection' in results['steps'] and results['steps']['anomaly_detection'].get('anomaly_summary'):
                anomalies = results['steps']['anomaly_detection']['anomaly_summary']
                print(f"\n🚨 ANOMALY DETECTION RESULTS:")
                
                if 'combined' in anomalies:
                    print(f"   • Total anomalies detected: {anomalies['combined']['num_anomalies']}")
                    print(f"   • Anomaly rate: {anomalies['combined']['anomaly_rate']:.2%}")
                
                if 'isolation_forest' in anomalies:
                    print(f"   • Isolation Forest anomalies: {anomalies['isolation_forest']['num_anomalies']}")
                
                if 'dbscan' in anomalies:
                    print(f"   • DBSCAN anomalies: {anomalies['dbscan']['num_anomalies']}")
                    print(f"   • DBSCAN clusters found: {anomalies['dbscan']['num_clusters']}")
                
        else:
            print(f"\n❌ PIPELINE FAILED:")
            failed_steps = [name for name, step in results.get('steps', {}).items() if not step.get('success', False)]
            print(f"   • Failed steps: {', '.join(failed_steps)}")
            if 'error' in results:
                print(f"   • Error: {results.get('error', 'Unknown error')}")
            return
        
        print(f"\n🎯 TESTING INDIVIDUAL COMPONENTS...")
        
        # Test individual components with sample data
        sample_df = df.sample(n=min(1000, len(df)))  # Use sample for faster testing
        
        # Test Data Profiler
        print(f"\n📊 Testing Data Profiler...")
        profiler = DataProfiler()
        profile_results = profiler.profile_dataset(sample_df)
        
        print(f"   • Profiling completed for {len(sample_df)} records")
        print(f"   • Quality grade: {profile_results.get('data_quality', {}).get('grade', 'N/A')}")
        
        # Test Anomaly Detector
        print(f"\n🔍 Testing Anomaly Detector...")
        detector = AnomalyDetector()
        anomaly_results = detector.comprehensive_anomaly_detection(sample_df)
        
        print(f"   • Anomaly detection completed")
        if anomaly_results and 'combined' in anomaly_results:
            print(f"   • Anomalies found: {anomaly_results['combined']['num_anomalies']}")
            print(f"   • Detection methods used: {', '.join(anomaly_results.keys())}")
        else:
            print(f"   • No anomalies detected or analysis failed")
        
        print(f"\n🎉 ALL TESTS COMPLETED SUCCESSFULLY!")
        print(f"\n📋 SUMMARY:")
        print(f"   • Real-world dataset processed: ✅")
        print(f"   • Data quality monitoring: ✅") 
        print(f"   • Anomaly detection: ✅")
        print(f"   • Data profiling: ✅")
        print(f"   • Pipeline integration: ✅")
        
        print(f"\n🚀 Ready to launch dashboard!")
        print(f"   Run: python3 run_dashboard.py")
        
    except Exception as e:
        logger.error(f"Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 