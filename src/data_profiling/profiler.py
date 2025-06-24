"""
Data Profiling Module for Data Quality Monitor
Analyzes datasets and calculates comprehensive data quality metrics
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging
from datetime import datetime
import re

class DataProfiler:
    """
    Comprehensive data profiling for quality assessment
    """
    
    def __init__(self, config: Dict = None):
        """Initialize the data profiler"""
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
    def profile_dataset(self, df: pd.DataFrame, dataset_name: str = "dataset") -> Dict:
        """
        Generate comprehensive data profile for a dataset
        """
        try:
            profile = {
                'dataset_name': dataset_name,
                'profiling_timestamp': datetime.now().isoformat(),
                'basic_info': self._get_basic_info(df),
                'column_profiles': self._profile_columns(df),
                'data_quality_summary': self._calculate_quality_summary(df),
                'relationships': self._analyze_relationships(df),
                'anomaly_indicators': self._detect_profile_anomalies(df)
            }
            
            return profile
            
        except Exception as e:
            self.logger.error(f"Error profiling dataset: {str(e)}")
            return None
    
    def _get_basic_info(self, df: pd.DataFrame) -> Dict:
        """Get basic dataset information"""
        return {
            'num_rows': len(df),
            'num_columns': len(df.columns),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / (1024 * 1024),
            'columns': df.columns.tolist(),
            'dtypes': df.dtypes.astype(str).to_dict()
        }
    
    def _profile_columns(self, df: pd.DataFrame) -> Dict:
        """Profile each column individually"""
        column_profiles = {}
        
        for column in df.columns:
            column_profiles[column] = self._profile_single_column(df[column], column)
        
        return column_profiles
    
    def _profile_single_column(self, series: pd.Series, column_name: str) -> Dict:
        """Profile a single column"""
        profile = {
            'column_name': column_name,
            'data_type': str(series.dtype),
            'count': len(series),
            'non_null_count': series.count(),
            'null_count': series.isnull().sum(),
            'null_percentage': (series.isnull().sum() / len(series)) * 100,
            'unique_count': series.nunique(),
            'unique_percentage': (series.nunique() / len(series)) * 100 if len(series) > 0 else 0,
        }
        
        # Add type-specific profiling
        if pd.api.types.is_numeric_dtype(series):
            profile.update(self._profile_numeric_column(series))
        elif pd.api.types.is_string_dtype(series) or series.dtype == 'object':
            profile.update(self._profile_text_column(series))
        elif pd.api.types.is_datetime64_any_dtype(series):
            profile.update(self._profile_datetime_column(series))
        
        # Data quality flags
        profile['quality_flags'] = self._assess_column_quality(series, profile)
        
        return profile
    
    def _profile_numeric_column(self, series: pd.Series) -> Dict:
        """Profile numeric columns"""
        numeric_series = series.dropna()
        
        if len(numeric_series) == 0:
            return {'numeric_profile': 'No valid numeric data'}
        
        return {
            'min': float(numeric_series.min()),
            'max': float(numeric_series.max()),
            'mean': float(numeric_series.mean()),
            'median': float(numeric_series.median()),
            'std': float(numeric_series.std()) if len(numeric_series) > 1 else 0,
            'q25': float(numeric_series.quantile(0.25)),
            'q75': float(numeric_series.quantile(0.75)),
            'zeros_count': int(np.sum(numeric_series == 0)),
            'negative_count': int(np.sum(numeric_series < 0)),
            'outliers_iqr': self._count_iqr_outliers(numeric_series),
            'outliers_zscore': self._count_zscore_outliers(numeric_series)
        }
    
    def _profile_text_column(self, series: pd.Series) -> Dict:
        """Profile text/string columns"""
        text_series = series.dropna().astype(str)
        
        if len(text_series) == 0:
            return {'text_profile': 'No valid text data'}
        
        # Basic text statistics
        lengths = text_series.str.len()
        
        profile = {
            'min_length': int(lengths.min()) if len(lengths) > 0 else 0,
            'max_length': int(lengths.max()) if len(lengths) > 0 else 0,
            'avg_length': float(lengths.mean()) if len(lengths) > 0 else 0,
            'empty_strings': int(np.sum(text_series == '')),
            'whitespace_only': int(np.sum(text_series.str.strip() == '')),
        }
        
        # Pattern analysis
        profile.update(self._analyze_text_patterns(text_series))
        
        # Most common values
        value_counts = text_series.value_counts()
        profile['most_common'] = value_counts.head(5).to_dict()
        
        return profile
    
    def _profile_datetime_column(self, series: pd.Series) -> Dict:
        """Profile datetime columns"""
        datetime_series = pd.to_datetime(series, errors='coerce').dropna()
        
        if len(datetime_series) == 0:
            return {'datetime_profile': 'No valid datetime data'}
        
        return {
            'min_date': datetime_series.min().isoformat(),
            'max_date': datetime_series.max().isoformat(),
            'date_range_days': (datetime_series.max() - datetime_series.min()).days,
            'future_dates': int(np.sum(datetime_series > pd.Timestamp.now())),
            'weekend_count': int(np.sum(datetime_series.dt.dayofweek >= 5))
        }
    
    def _analyze_text_patterns(self, text_series: pd.Series) -> Dict:
        """Analyze common text patterns"""
        patterns = {
            'email_pattern': r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
            'phone_pattern': r'^\+?1?[-.\s]?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}$',
            'url_pattern': r'^https?://[^\s/$.?#].[^\s]*$',
            'numeric_pattern': r'^\d+$',
            'alphanumeric_pattern': r'^[a-zA-Z0-9]+$'
        }
        
        pattern_matches = {}
        for pattern_name, pattern in patterns.items():
            matches = text_series.str.match(pattern, na=False)
            pattern_matches[pattern_name] = int(matches.sum())
        
        return pattern_matches
    
    def _count_iqr_outliers(self, numeric_series: pd.Series) -> int:
        """Count outliers using IQR method"""
        Q1 = numeric_series.quantile(0.25)
        Q3 = numeric_series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = (numeric_series < lower_bound) | (numeric_series > upper_bound)
        return int(outliers.sum())
    
    def _count_zscore_outliers(self, numeric_series: pd.Series, threshold: float = 3) -> int:
        """Count outliers using Z-score method"""
        if len(numeric_series) <= 1:
            return 0
        z_scores = np.abs((numeric_series - numeric_series.mean()) / numeric_series.std())
        outliers = z_scores > threshold
        return int(outliers.sum())
    
    def _assess_column_quality(self, series: pd.Series, profile: Dict) -> Dict:
        """Assess data quality for a column"""
        flags = {
            'high_null_rate': profile['null_percentage'] > 20,
            'low_uniqueness': profile['unique_percentage'] < 5 and profile['unique_count'] > 1,
            'high_uniqueness': profile['unique_percentage'] > 95,
            'potential_identifier': profile['unique_percentage'] == 100 or profile['unique_count'] == profile['count'],
        }
        
        # Add numeric-specific quality flags
        if 'outliers_iqr' in profile:
            flags['has_outliers'] = profile['outliers_iqr'] > 0
            flags['many_zeros'] = profile.get('zeros_count', 0) / profile['count'] > 0.1
            flags['has_negatives'] = profile.get('negative_count', 0) > 0
        
        # Add text-specific quality flags
        if 'empty_strings' in profile:
            flags['has_empty_strings'] = profile['empty_strings'] > 0
            flags['inconsistent_length'] = profile.get('max_length', 0) - profile.get('min_length', 0) > 50
        
        return flags
    
    def _calculate_quality_summary(self, df: pd.DataFrame) -> Dict:
        """Calculate overall data quality summary"""
        total_cells = df.shape[0] * df.shape[1]
        null_cells = df.isnull().sum().sum()
        
        # Calculate completeness
        completeness = 1 - (null_cells / total_cells) if total_cells > 0 else 0
        
        # Calculate uniqueness (average across columns)
        uniqueness_scores = []
        for col in df.columns:
            if len(df) > 0:
                uniqueness = df[col].nunique() / len(df)
                uniqueness_scores.append(uniqueness)
        
        avg_uniqueness = np.mean(uniqueness_scores) if uniqueness_scores else 0
        
        # Calculate validity (columns with expected patterns)
        validity_score = self._calculate_validity_score(df)
        
        # Overall quality score (weighted average)
        overall_score = (
            completeness * 0.4 +
            validity_score * 0.4 +
            min(avg_uniqueness, 1.0) * 0.2
        )
        
        return {
            'completeness': completeness,
            'uniqueness': avg_uniqueness,
            'validity': validity_score,
            'overall_quality_score': overall_score,
            'quality_grade': self._get_quality_grade(overall_score),
            'total_cells': total_cells,
            'null_cells': null_cells,
            'null_percentage': (null_cells / total_cells) * 100 if total_cells > 0 else 0
        }
    
    def _calculate_validity_score(self, df: pd.DataFrame) -> float:
        """Calculate validity score based on expected patterns"""
        validity_scores = []
        
        for col in df.columns:
            col_lower = col.lower()
            series = df[col].dropna()
            
            if len(series) == 0:
                continue
            
            # Email validation
            if 'email' in col_lower:
                email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
                valid_emails = series.astype(str).str.match(email_pattern, na=False).sum()
                validity_scores.append(valid_emails / len(series))
            
            # Age validation
            elif 'age' in col_lower and pd.api.types.is_numeric_dtype(series):
                valid_ages = ((series >= 0) & (series <= 150)).sum()
                validity_scores.append(valid_ages / len(series))
            
            # Phone validation
            elif 'phone' in col_lower:
                phone_pattern = r'^\+?1?[-.\s]?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}$'
                valid_phones = series.astype(str).str.match(phone_pattern, na=False).sum()
                validity_scores.append(valid_phones / len(series))
            
            # Default: assume valid if not null
            else:
                validity_scores.append(1.0)
        
        return np.mean(validity_scores) if validity_scores else 1.0
    
    def _get_quality_grade(self, score: float) -> str:
        """Convert quality score to grade"""
        if score >= 0.9:
            return 'A'
        elif score >= 0.8:
            return 'B'
        elif score >= 0.7:
            return 'C'
        elif score >= 0.6:
            return 'D'
        else:
            return 'F'
    
    def _analyze_relationships(self, df: pd.DataFrame) -> Dict:
        """Analyze relationships between columns"""
        relationships = {}
        
        # Find potential duplicate columns
        duplicate_columns = []
        for i, col1 in enumerate(df.columns):
            for col2 in df.columns[i+1:]:
                if df[col1].equals(df[col2]):
                    duplicate_columns.append((col1, col2))
        
        relationships['duplicate_columns'] = duplicate_columns
        
        # Find highly correlated numeric columns
        numeric_df = df.select_dtypes(include=[np.number])
        if len(numeric_df.columns) > 1:
            correlation_matrix = numeric_df.corr()
            high_correlations = []
            
            for i in range(len(correlation_matrix.columns)):
                for j in range(i+1, len(correlation_matrix.columns)):
                    corr = correlation_matrix.iloc[i, j]
                    if abs(corr) > 0.8:  # High correlation threshold
                        high_correlations.append({
                            'column1': correlation_matrix.columns[i],
                            'column2': correlation_matrix.columns[j],
                            'correlation': corr
                        })
            
            relationships['high_correlations'] = high_correlations
        
        return relationships
    
    def _detect_profile_anomalies(self, df: pd.DataFrame) -> Dict:
        """Detect anomalies in the data profile itself"""
        anomalies = []
        
        for col in df.columns:
            series = df[col]
            
            # Anomaly: Very high null rate
            null_rate = series.isnull().sum() / len(series)
            if null_rate > 0.5:
                anomalies.append({
                    'type': 'high_null_rate',
                    'column': col,
                    'value': null_rate,
                    'severity': 'high'
                })
            
            # Anomaly: Single value dominates
            if len(series) > 0:
                most_common_rate = series.value_counts().iloc[0] / len(series)
                if most_common_rate > 0.9:
                    anomalies.append({
                        'type': 'single_value_dominance',
                        'column': col,
                        'value': most_common_rate,
                        'severity': 'medium'
                    })
        
        return {'anomalies': anomalies, 'total_anomalies': len(anomalies)} 