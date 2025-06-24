"""
Anomaly Detection Module for Data Quality Monitor
Uses Isolation Forest and DBSCAN algorithms to detect data anomalies
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
import logging
from typing import Dict, List, Tuple, Optional
import yaml

class AnomalyDetector:
    """
    AI-powered anomaly detection for data quality monitoring
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize the anomaly detector with configuration"""
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        self.anomaly_config = self.config['anomaly_detection']
        self.logger = logging.getLogger(__name__)
        
        # Initialize models
        self.isolation_forest = IsolationForest(
            contamination=self.anomaly_config['isolation_forest']['contamination'],
            n_estimators=self.anomaly_config['isolation_forest']['n_estimators'],
            random_state=self.anomaly_config['isolation_forest']['random_state']
        )
        
        self.dbscan = DBSCAN(
            eps=self.anomaly_config['dbscan']['eps'],
            min_samples=self.anomaly_config['dbscan']['min_samples']
        )
        
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess data for anomaly detection
        - Handle missing values
        - Encode categorical variables
        - Scale numerical features
        """
        processed_df = df.copy()
        
        # Separate numerical and categorical columns
        numerical_cols = processed_df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = processed_df.select_dtypes(include=['object']).columns.tolist()
        
        # Handle missing values for numerical columns
        for col in numerical_cols:
            processed_df[col] = processed_df[col].fillna(processed_df[col].median())
        
        # Handle missing values and encode categorical columns
        for col in categorical_cols:
            processed_df[col] = processed_df[col].fillna('missing')
            
            # Initialize label encoder for this column if not exists
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
            
            # Fit and transform categorical data
            processed_df[col] = self.label_encoders[col].fit_transform(processed_df[col])
        
        return processed_df
    
    def detect_isolation_forest_anomalies(self, df: pd.DataFrame) -> Dict:
        """
        Use Isolation Forest to detect anomalies
        Returns anomaly scores and predictions
        """
        try:
            processed_df = self.preprocess_data(df)
            
            # Fit and predict
            anomaly_predictions = self.isolation_forest.fit_predict(processed_df)
            anomaly_scores = self.isolation_forest.decision_function(processed_df)
            
            # Convert to anomaly probability (0-1 scale)
            anomaly_probs = 1 - (anomaly_scores - anomaly_scores.min()) / (anomaly_scores.max() - anomaly_scores.min())
            
            return {
                'method': 'isolation_forest',
                'anomaly_predictions': anomaly_predictions,
                'anomaly_scores': anomaly_scores,
                'anomaly_probabilities': anomaly_probs,
                'num_anomalies': np.sum(anomaly_predictions == -1),
                'anomaly_rate': np.sum(anomaly_predictions == -1) / len(anomaly_predictions)
            }
            
        except Exception as e:
            self.logger.error(f"Error in Isolation Forest anomaly detection: {str(e)}")
            return None
    
    def detect_dbscan_anomalies(self, df: pd.DataFrame) -> Dict:
        """
        Use DBSCAN clustering to identify outliers
        Points not assigned to any cluster are considered anomalies
        """
        try:
            processed_df = self.preprocess_data(df)
            
            # Scale the data for DBSCAN
            scaled_data = self.scaler.fit_transform(processed_df)
            
            # Apply DBSCAN
            cluster_labels = self.dbscan.fit_predict(scaled_data)
            
            # Points with label -1 are outliers/anomalies
            anomaly_predictions = np.where(cluster_labels == -1, -1, 1)
            
            return {
                'method': 'dbscan',
                'cluster_labels': cluster_labels,
                'anomaly_predictions': anomaly_predictions,
                'num_clusters': len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0),
                'num_anomalies': np.sum(cluster_labels == -1),
                'anomaly_rate': np.sum(cluster_labels == -1) / len(cluster_labels)
            }
            
        except Exception as e:
            self.logger.error(f"Error in DBSCAN anomaly detection: {str(e)}")
            return None
    
    def detect_statistical_anomalies(self, df: pd.DataFrame) -> Dict:
        """
        Detect statistical anomalies using Z-score and IQR methods
        """
        try:
            numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            statistical_anomalies = {}
            
            for col in numerical_cols:
                col_data = df[col].dropna()
                
                # Z-score method (threshold: 3)
                z_scores = np.abs((col_data - col_data.mean()) / col_data.std())
                z_anomalies = z_scores > 3
                
                # IQR method
                Q1 = col_data.quantile(0.25)
                Q3 = col_data.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                iqr_anomalies = (col_data < lower_bound) | (col_data > upper_bound)
                
                statistical_anomalies[col] = {
                    'z_score_anomalies': z_anomalies.sum(),
                    'iqr_anomalies': iqr_anomalies.sum(),
                    'total_anomalies': (z_anomalies | iqr_anomalies).sum()
                }
            
            return {
                'method': 'statistical',
                'column_anomalies': statistical_anomalies,
                'total_statistical_anomalies': sum([v['total_anomalies'] for v in statistical_anomalies.values()])
            }
            
        except Exception as e:
            self.logger.error(f"Error in statistical anomaly detection: {str(e)}")
            return None
    
    def comprehensive_anomaly_detection(self, df: pd.DataFrame) -> Dict:
        """
        Run all anomaly detection methods and combine results
        """
        results = {}
        
        # Run Isolation Forest
        if_results = self.detect_isolation_forest_anomalies(df)
        if if_results:
            results['isolation_forest'] = if_results
        
        # Run DBSCAN
        dbscan_results = self.detect_dbscan_anomalies(df)
        if dbscan_results:
            results['dbscan'] = dbscan_results
        
        # Run Statistical methods
        statistical_results = self.detect_statistical_anomalies(df)
        if statistical_results:
            results['statistical'] = statistical_results
        
        # Create combined anomaly score
        if if_results and dbscan_results:
            # Combine anomaly predictions (consensus approach)
            combined_anomalies = (
                (if_results['anomaly_predictions'] == -1) | 
                (dbscan_results['anomaly_predictions'] == -1)
            ).astype(int)
            
            results['combined'] = {
                'anomaly_predictions': np.where(combined_anomalies == 1, -1, 1),
                'num_anomalies': np.sum(combined_anomalies),
                'anomaly_rate': np.sum(combined_anomalies) / len(combined_anomalies)
            }
        
        return results
    
    def get_anomaly_details(self, df: pd.DataFrame, anomaly_results: Dict) -> pd.DataFrame:
        """
        Get detailed information about detected anomalies
        """
        anomaly_df = df.copy()
        
        # Add anomaly flags from different methods
        if 'isolation_forest' in anomaly_results:
            anomaly_df['isolation_forest_anomaly'] = anomaly_results['isolation_forest']['anomaly_predictions'] == -1
            anomaly_df['isolation_forest_score'] = anomaly_results['isolation_forest']['anomaly_probabilities']
        
        if 'dbscan' in anomaly_results:
            anomaly_df['dbscan_anomaly'] = anomaly_results['dbscan']['anomaly_predictions'] == -1
            anomaly_df['dbscan_cluster'] = anomaly_results['dbscan']['cluster_labels']
        
        if 'combined' in anomaly_results:
            anomaly_df['combined_anomaly'] = anomaly_results['combined']['anomaly_predictions'] == -1
        
        return anomaly_df 