#!/usr/bin/env python3
"""
Data Quality Pipeline - Main orchestrator for the AI-Powered Data Quality Monitor
Coordinates data ingestion, profiling, anomaly detection, and monitoring
"""

import os
import sys
import json
import logging
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List
import pandas as pd
import duckdb
import yaml

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from data_profiling.profiler import DataProfiler
from anomaly_detection.detector import AnomalyDetector

class DataQualityPipeline:
    """
    Main orchestrator for the data quality monitoring pipeline
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize the data quality pipeline"""
        self.config_path = config_path
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        self.db_path = self.config['database']['path']
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.anomaly_detector = AnomalyDetector(config_path)
        self.data_profiler = DataProfiler(self.config.get('data_quality', {}))
        
        # Initialize database connection
        self.conn = duckdb.connect(self.db_path)
        
        # Ensure data directories exist
        self._ensure_directories()
    
    def _ensure_directories(self):
        """Ensure all required directories exist"""
        directories = [
            "data/raw",
            "data/processed",
            "data/profiles",
            "data/anomalies"
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def ingest_csv(self, csv_path: str, table_name: str = None) -> bool:
        """
        Ingest a CSV file into the database
        """
        try:
            if not os.path.exists(csv_path):
                self.logger.error(f"CSV file not found: {csv_path}")
                return False
            
            # Generate table name if not provided
            if table_name is None:
                table_name = f"raw_{Path(csv_path).stem}"
            
            # Read CSV and load into DuckDB
            df = pd.read_csv(csv_path)
            
            # Store in database
            self.conn.execute(f"DROP TABLE IF EXISTS {table_name}")
            self.conn.execute(f"CREATE TABLE {table_name} AS SELECT * FROM df")
            
            self.logger.info(f"Successfully ingested {csv_path} as table {table_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error ingesting CSV {csv_path}: {str(e)}")
            return False
    
    def run_dbt_pipeline(self) -> bool:
        """
        Run the dbt pipeline to transform data
        """
        try:
            # Change to dbt project directory
            original_dir = os.getcwd()
            dbt_dir = "dbt_project"
            
            if not os.path.exists(dbt_dir):
                self.logger.error(f"dbt project directory not found: {dbt_dir}")
                return False
            
            # Close database connection to avoid lock conflicts
            if hasattr(self, 'conn') and self.conn:
                self.conn.close()
                self.logger.info("Closed database connection for dbt execution")
            
            os.chdir(dbt_dir)
            
            # Run dbt commands with full path
            dbt_path = "/Users/akshayrameshnair/Library/Python/3.12/bin/dbt"
            commands = [
                f"{dbt_path} deps --profiles-dir .",
                f"{dbt_path} run --profiles-dir .",
                f"{dbt_path} test --profiles-dir ."
            ]
            
            for cmd in commands:
                self.logger.info(f"Running: {cmd}")
                result = subprocess.run(cmd.split(), capture_output=True, text=True)
                
                if result.returncode != 0:
                    self.logger.error(f"dbt command failed: {cmd}")
                    self.logger.error(f"Return code: {result.returncode}")
                    self.logger.error(f"Stderr: {result.stderr}")
                    self.logger.error(f"Stdout: {result.stdout}")
                    os.chdir(original_dir)
                    
                    # Reopen database connection even on failure
                    self.conn = duckdb.connect(self.db_path)
                    self.logger.info("Reopened database connection after dbt command failure")
                    
                    return False
                else:
                    self.logger.info(f"dbt command succeeded: {cmd}")
                    if result.stdout:
                        self.logger.debug(f"Output: {result.stdout}")
            
            os.chdir(original_dir)
            
            # Reopen database connection
            self.conn = duckdb.connect(self.db_path)
            self.logger.info("Reopened database connection after dbt execution")
            
            self.logger.info("dbt pipeline completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error running dbt pipeline: {str(e)}")
            if 'original_dir' in locals():
                os.chdir(original_dir)
            
            # Ensure database connection is reopened even on error
            if not hasattr(self, 'conn') or not self.conn:
                self.conn = duckdb.connect(self.db_path)
                self.logger.info("Reopened database connection after dbt error")
            
            return False
    
    def profile_data(self, table_name: str, dataset_name: str = None) -> Optional[Dict]:
        """
        Profile data from a table
        """
        try:
            # Get data from database
            df = self.conn.execute(f"SELECT * FROM {table_name}").fetchdf()
            
            if df.empty:
                self.logger.warning(f"No data found in table {table_name}")
                return None
            
            # Generate profile
            profile = self.data_profiler.profile_dataset(
                df, 
                dataset_name or table_name
            )
            
            # Save profile to disk
            if profile:
                profile_path = f"data/profiles/{table_name}_profile.json"
                with open(profile_path, 'w') as f:
                    json.dump(profile, f, indent=2, default=str)
                
                self.logger.info(f"Data profile saved to {profile_path}")
            
            return profile
            
        except Exception as e:
            self.logger.error(f"Error profiling data from {table_name}: {str(e)}")
            return None
    
    def detect_anomalies(self, table_name: str) -> Optional[Dict]:
        """
        Detect anomalies in data from a table
        """
        try:
            # Get data from database
            df = self.conn.execute(f"SELECT * FROM {table_name}").fetchdf()
            
            if df.empty:
                self.logger.warning(f"No data found in table {table_name}")
                return None
            
            # Detect anomalies
            anomaly_results = self.anomaly_detector.comprehensive_anomaly_detection(df)
            
            if anomaly_results:
                # Get detailed anomaly information
                anomaly_details = self.anomaly_detector.get_anomaly_details(df, anomaly_results)
                
                # Save anomaly results
                anomaly_path = f"data/anomalies/{table_name}_anomalies.json"
                with open(anomaly_path, 'w') as f:
                    json.dump(anomaly_results, f, indent=2, default=str)
                
                # Save anomaly details
                details_path = f"data/anomalies/{table_name}_anomaly_details.csv"
                anomaly_details.to_csv(details_path, index=False)
                
                self.logger.info(f"Anomaly results saved to {anomaly_path}")
                self.logger.info(f"Anomaly details saved to {details_path}")
            
            return anomaly_results
            
        except Exception as e:
            self.logger.error(f"Error detecting anomalies in {table_name}: {str(e)}")
            return None
    
    def run_full_pipeline(self, csv_path: str, table_name: str = None) -> Dict:
        """
        Run the complete data quality pipeline
        """
        pipeline_results = {
            'timestamp': datetime.now().isoformat(),
            'csv_path': csv_path,
            'table_name': table_name,
            'steps': {}
        }
        
        try:
            # Step 1: Ingest CSV
            self.logger.info("Step 1: Ingesting CSV data")
            ingestion_success = self.ingest_csv(csv_path, table_name)
            pipeline_results['steps']['ingestion'] = {
                'success': ingestion_success,
                'timestamp': datetime.now().isoformat()
            }
            
            if not ingestion_success:
                return pipeline_results
            
            # Use the actual table name
            actual_table_name = table_name or f"raw_{Path(csv_path).stem}"
            
            # Step 2: Run dbt pipeline
            self.logger.info("Step 2: Running dbt transformations")
            dbt_success = self.run_dbt_pipeline()
            pipeline_results['steps']['dbt_transformation'] = {
                'success': dbt_success,
                'timestamp': datetime.now().isoformat()
            }
            
            # Step 3: Profile data
            self.logger.info("Step 3: Profiling data")
            profile = self.profile_data(actual_table_name)
            pipeline_results['steps']['profiling'] = {
                'success': profile is not None,
                'timestamp': datetime.now().isoformat(),
                'profile_summary': profile.get('data_quality_summary') if profile else None
            }
            
            # Step 4: Detect anomalies
            self.logger.info("Step 4: Detecting anomalies")
            anomalies = self.detect_anomalies(actual_table_name)
            pipeline_results['steps']['anomaly_detection'] = {
                'success': anomalies is not None,
                'timestamp': datetime.now().isoformat(),
                'anomaly_summary': self._summarize_anomalies(anomalies) if anomalies else None
            }
            
            # Step 5: Update monitoring tables
            self.logger.info("Step 5: Updating monitoring tables")
            monitoring_success = self._update_monitoring_tables(
                actual_table_name, profile, anomalies
            )
            pipeline_results['steps']['monitoring_update'] = {
                'success': monitoring_success,
                'timestamp': datetime.now().isoformat()
            }
            
            self.logger.info("Pipeline completed successfully")
            
        except Exception as e:
            self.logger.error(f"Error in pipeline execution: {str(e)}")
            pipeline_results['error'] = str(e)
        
        return pipeline_results
    
    def _summarize_anomalies(self, anomaly_results: Dict) -> Dict:
        """Summarize anomaly detection results"""
        summary = {}
        
        if 'isolation_forest' in anomaly_results:
            summary['isolation_forest'] = {
                'num_anomalies': anomaly_results['isolation_forest']['num_anomalies'],
                'anomaly_rate': anomaly_results['isolation_forest']['anomaly_rate']
            }
        
        if 'dbscan' in anomaly_results:
            summary['dbscan'] = {
                'num_anomalies': anomaly_results['dbscan']['num_anomalies'],
                'anomaly_rate': anomaly_results['dbscan']['anomaly_rate'],
                'num_clusters': anomaly_results['dbscan']['num_clusters']
            }
        
        if 'combined' in anomaly_results:
            summary['combined'] = {
                'num_anomalies': anomaly_results['combined']['num_anomalies'],
                'anomaly_rate': anomaly_results['combined']['anomaly_rate']
            }
        
        return summary
    
    def _update_monitoring_tables(self, table_name: str, profile: Dict, anomalies: Dict) -> bool:
        """Update monitoring tables with latest results"""
        try:
            # Create monitoring table if it doesn't exist
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS data_quality_monitoring (
                    table_name VARCHAR,
                    timestamp TIMESTAMP,
                    quality_score DOUBLE,
                    quality_grade VARCHAR,
                    completeness DOUBLE,
                    validity DOUBLE,
                    uniqueness DOUBLE,
                    num_anomalies INTEGER,
                    anomaly_rate DOUBLE,
                    profile_json VARCHAR,
                    anomaly_json VARCHAR
                )
            """)
            
            # Prepare data for insertion
            current_time = datetime.now()
            quality_summary = profile.get('data_quality_summary', {}) if profile else {}
            anomaly_summary = self._summarize_anomalies(anomalies) if anomalies else {}
            
            # Insert monitoring record with type conversion
            self.conn.execute("""
                INSERT INTO data_quality_monitoring VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                str(table_name),
                current_time,
                float(quality_summary.get('overall_quality_score', 0)),
                str(quality_summary.get('quality_grade', 'F')),
                float(quality_summary.get('completeness', 0)),
                float(quality_summary.get('validity', 0)),
                float(quality_summary.get('uniqueness', 0)),
                int(anomaly_summary.get('combined', {}).get('num_anomalies', 0)),
                float(anomaly_summary.get('combined', {}).get('anomaly_rate', 0)),
                json.dumps(profile, default=str) if profile else None,
                json.dumps(anomalies, default=str) if anomalies else None
            ])
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating monitoring tables: {str(e)}")
            return False
    
    def get_monitoring_data(self, table_name: str = None) -> pd.DataFrame:
        """Get monitoring data for dashboard"""
        try:
            query = "SELECT * FROM data_quality_monitoring"
            if table_name:
                query += f" WHERE table_name = '{table_name}'"
            query += " ORDER BY timestamp DESC"
            
            return self.conn.execute(query).fetchdf()
            
        except Exception as e:
            self.logger.error(f"Error retrieving monitoring data: {str(e)}")
            return pd.DataFrame()
    
    def get_table_list(self) -> List[str]:
        """Get list of available tables"""
        try:
            tables = self.conn.execute("SHOW TABLES").fetchall()
            return [table[0] for table in tables]
        except Exception as e:
            self.logger.error(f"Error getting table list: {str(e)}")
            return []
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close() 