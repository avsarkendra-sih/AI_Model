"""Enhanced logger with data preparation utilities using OOP design."""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Callable, Any
import pandas as pd
from abc import ABC, abstractmethod

class BaseLogger:
    """Base logger class with core functionality."""
    
    def __init__(self, name: str, log_level: int = logging.INFO):
        self.name = name
        self.log_level = log_level
        self.logger = self._setup_logger()
    
    def _setup_logger(self) -> logging.Logger:
        """Set up and configure logger with file and console handlers."""
        logger = logging.getLogger(self.name)
        logger.setLevel(self.log_level)
        
        if not logger.handlers:  # Avoid duplicate handlers
            # Create logs directory
            log_dir = Path.cwd() / "logs"
            log_dir.mkdir(exist_ok=True)
            
            # Create log file with timestamp
            log_file = log_dir / f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
            
            # Setup handlers and formatter
            formatter = logging.Formatter(
                "[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s"
            )
            
            console_handler = logging.StreamHandler(sys.stdout)
            file_handler = logging.FileHandler(log_file)
            
            console_handler.setFormatter(formatter)
            file_handler.setFormatter(formatter)
            
            logger.addHandler(console_handler)
            logger.addHandler(file_handler)
        
        return logger
    
    def info(self, message: str) -> None:
        """Log info message."""
        self.logger.info(message)
    
    def warning(self, message: str) -> None:
        """Log warning message."""
        self.logger.warning(message)
    
    def error(self, message: str) -> None:
        """Log error message."""
        self.logger.error(message)

class DataValidator:
    """Class for validating dataframes against schemas and requirements."""
    
    def __init__(self, logger: Optional[BaseLogger] = None):
        self.logger = logger
    
    def validate_schema(self, df: pd.DataFrame, required_columns: List[str], 
                       schema_validator: Optional[Callable] = None) -> Tuple[bool, List[str]]:
        """
        Validate dataframe against required columns and optional schema.
        
        Args:
            df: DataFrame to validate
            required_columns: List of required column names
            schema_validator: Optional schema validation function
        
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        # Check missing columns
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            errors.append(f"Missing columns: {missing_cols}")
        
        # Check null values in required columns
        for col in required_columns:
            if col in df.columns and df[col].isnull().any():
                errors.append(f"{col} has {df[col].isnull().sum()} missing values")
        
        # Schema validation on sample rows
        if schema_validator and not missing_cols:
            sample_size = min(10, len(df))
            for idx, row in df.head(sample_size).iterrows():
                try:
                    schema_validator(row)
                except Exception as e:
                    errors.append(f"Schema validation error in row {idx}: {str(e)}")
        
        is_valid = len(errors) == 0
        if self.logger and not is_valid:
            self.logger.warning(f"Validation issues found: {errors}")
        
        return is_valid, errors

class DataSummarizer:
    """Class for generating and displaying data summaries."""
    
    def __init__(self):
        self.summary_cache = {}
    
    def generate_summary(self, df: pd.DataFrame, column_config: Dict[str, str]) -> Dict[str, Any]:
        """
        Generate comprehensive data summary statistics.
        
        Args:
            df: DataFrame to summarize
            column_config: Dictionary mapping summary keys to column names
        
        Returns:
            Dictionary with summary statistics
        """
        if df is None or df.empty:
            return {}
        
        summary = {"total_records": len(df)}
        
        for key, col_name in column_config.items():
            if col_name in df.columns:
                if key.endswith('_count'):
                    summary[key] = df[col_name].nunique()
                elif key.endswith('_distribution'):
                    summary[key] = df[col_name].value_counts().to_dict()
                elif key.startswith('avg_'):
                    summary[key] = df[col_name].mean() if pd.api.types.is_numeric_dtype(df[col_name]) else 0
            else:
                summary[key] = 0 if key.endswith('_count') or key.startswith('avg_') else {}
        
        return summary
    
    def print_summary(self, summary: Dict[str, Any], title: str = "DATA SUMMARY") -> None:
        """
        Print formatted data summary.
        
        Args:
            summary: Summary dictionary
            title: Title for the summary output
        """
        print(f"\n{'='*50}")
        print(title.upper())
        print("="*50)
        
        for key, value in summary.items():
            if isinstance(value, dict):
                print(f"{self._format_key(key)}:")
                for k, v in value.items():
                    print(f"  {k}: {v}")
            else:
                formatted_key = self._format_key(key)
                if isinstance(value, float):
                    print(f"{formatted_key}: {value:.2f}")
                else:
                    print(f"{formatted_key}: {value}")
        print("="*50)
    
    def _format_key(self, key: str) -> str:
        """Format key for display."""
        return key.replace('_', ' ').title()

def setup_logger(name: str, log_level: int = logging.INFO) -> logging.Logger:
    """Factory function to create logger for backward compatibility."""
    logger_instance = BaseLogger(name, log_level)
    return logger_instance.logger