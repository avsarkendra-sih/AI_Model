"""Enhanced helper functions using Object-Oriented Programming."""

import pandas as pd
import numpy as np
from typing import Any, Dict, List, Union, Optional, Callable
import ast
from abc import ABC, abstractmethod

class TextProcessor:
    """Class for text processing operations."""
    
    def __init__(self):
        self.processed_count = 0
    
    def safe_literal_eval(self, value: Any) -> Any:
        """
        Safely evaluate a string containing a Python literal.
        
        Args:
            value: Value to evaluate (if it's a string)
        
        Returns:
            Evaluated literal or original value if evaluation fails
        """
        if isinstance(value, str):
            try:
                return ast.literal_eval(value)
            except (ValueError, SyntaxError):
                return value
        return value
    
    def clean_text(self, text: str) -> str:
        """
        Clean and standardize text data.
        
        Args:
            text (str): Input text to clean
        
        Returns:
            str: Cleaned text
        """
        if not isinstance(text, str) or not text:
            return ""
        
        # Remove extra whitespace
        text = " ".join(text.split())
        self.processed_count += 1
        return text
    
    def batch_process_columns(self, df: pd.DataFrame, text_columns: List[str], 
                            processing_func: Optional[Callable] = None) -> pd.DataFrame:
        """
        Apply text processing function to multiple columns efficiently.
        
        Args:
            df: Input dataframe
            text_columns: List of column names to process
            processing_func: Function to apply to text columns (defaults to clean_text)
        
        Returns:
            DataFrame with processed text columns
        """
        if processing_func is None:
            processing_func = self.clean_text
        
        for col in text_columns:
            if col in df.columns:
                df[col] = df[col].apply(processing_func)
        
        return df
    
    def get_processing_stats(self) -> Dict[str, int]:
        """Get processing statistics."""
        return {"processed_text_count": self.processed_count}

class MissingValueHandler:
    """Class for handling missing values in dataframes."""
    
    def __init__(self, default_strategy: str = "drop"):
        self.default_strategy = default_strategy
        self.fill_defaults = {
            'Special Requirements': '',
            'Location': 'Online',
            'Mode': 'Remote'
        }
    
    def set_fill_defaults(self, defaults: Dict[str, Any]) -> None:
        """Set custom default values for filling missing data."""
        self.fill_defaults.update(defaults)
    
    def handle_missing_values(self, df: pd.DataFrame, strategy: Optional[str] = None, 
                            required_columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Handle missing values in the dataframe.
        
        Args:
            df: Input dataframe
            strategy: Strategy to handle missing values - "drop" or "fill"
            required_columns: List of columns that cannot have missing values
        
        Returns:
            DataFrame with handled missing values
        """
        strategy = strategy or self.default_strategy
        
        if required_columns is None:
            required_columns = ['Company Name', 'Job Title', 'Skills Requirements', 
                               'Job Description', 'Duration', 'JobType']
        
        # Drop rows with missing required values
        existing_required = [col for col in required_columns if col in df.columns]
        df = df.dropna(subset=existing_required, how='any')
        
        if strategy == "fill":
            df = df.fillna(self.fill_defaults)
        
        return df

class CategoricalProcessor:
    """Class for processing categorical data."""
    
    def __init__(self):
        self.value_mappings = {}
        self.default_work_mode_mapping = {
            'remote': 'Online', 'hybrid': 'Hybrid', 'offline': 'Offline', 
            'on-site': 'Offline', 'online': 'Online'
        }
    
    def add_mapping(self, mapping_name: str, mapping: Dict[str, str]) -> None:
        """Add a new categorical mapping."""
        self.value_mappings[mapping_name] = mapping
    
    def standardize_values(self, df: pd.DataFrame, column: str, 
                          mapping: Optional[Dict[str, str]] = None,
                          mapping_name: Optional[str] = None) -> pd.DataFrame:
        """
        Standardize categorical values using a mapping dictionary.
        
        Args:
            df: Input dataframe
            column: Column name to standardize
            mapping: Direct mapping dictionary
            mapping_name: Name of stored mapping to use
        
        Returns:
            DataFrame with standardized values
        """
        if mapping_name and mapping_name in self.value_mappings:
            mapping = self.value_mappings[mapping_name]
        elif mapping is None:
            mapping = self.default_work_mode_mapping
        
        if column in df.columns:
            df[column] = df[column].str.lower().map(
                lambda x: mapping.get(x, x.capitalize() if isinstance(x, str) else x)
            )
        return df

class FeatureEngineer:
    """Class for feature engineering operations."""
    
    def __init__(self):
        self.feature_configs = []
    
    def add_feature_config(self, name: str, source_column: str, function: Callable) -> None:
        """Add a feature configuration."""
        self.feature_configs.append({
            'name': name,
            'source_column': source_column,
            'function': function
        })
    
    def create_features(self, df: pd.DataFrame, 
                       feature_configs: Optional[List[Dict[str, Any]]] = None) -> pd.DataFrame:
        """
        Add multiple feature columns based on configuration.
        
        Args:
            df: Input dataframe
            feature_configs: List of feature configuration dictionaries
        
        Returns:
            DataFrame with added feature columns
        """
        configs = feature_configs or self.feature_configs
        
        for config in configs:
            name = config['name']
            source_col = config['source_column']
            func = config['function']
            
            if source_col in df.columns:
                df[name] = df[source_col].apply(func)
        
        return df
    
    def create_id_column(self, df: pd.DataFrame, id_column_name: str = 'job_id') -> pd.DataFrame:
        """Create unique ID column."""
        df[id_column_name] = [f"{i:08x}" for i in range(1, len(df) + 1)]
        return df

class SkillsProcessor:
    """Specialized class for processing skills-related data."""
    
    def __init__(self, text_processor: TextProcessor):
        self.text_processor = text_processor
    
    def process_skills_column(self, df: pd.DataFrame, 
                            skills_column: str = 'Skills Requirements') -> pd.DataFrame:
        """
        Process skills column to handle string lists and add skills count.
        
        Args:
            df: Input dataframe
            skills_column: Name of the skills column
        
        Returns:
            DataFrame with processed skills and added skills_count column
        """
        if skills_column in df.columns:
            df[skills_column] = df[skills_column].apply(
                lambda x: self.text_processor.safe_literal_eval(x) if isinstance(x, str) else x
            )
            df['skills_count'] = df[skills_column].apply(
                lambda x: len(x) if isinstance(x, list) else 0
            )
        return df

class PreprocessingPipeline:
    """Main preprocessing pipeline that orchestrates all processors."""
    
    def __init__(self):
        self.text_processor = TextProcessor()
        self.missing_handler = MissingValueHandler()
        self.categorical_processor = CategoricalProcessor()
        self.feature_engineer = FeatureEngineer()
        self.skills_processor = SkillsProcessor(self.text_processor)
        
        self._setup_default_features()
    
    def _setup_default_features(self) -> None:
        """Setup default feature engineering configurations."""
        self.feature_engineer.add_feature_config(
            name='description_length',
            source_column='Job Description',
            function=lambda x: len(x.split()) if isinstance(x, str) else 0
        )
    
    def get_default_config(self) -> Dict[str, Any]:
        """Get default preprocessing configuration."""
        return {
            'text_columns': ['Job Description', 'Special Requirements', 'Job Title'],
            'work_mode_column': 'Mode',
            'skills_column': 'Skills Requirements',
            'required_columns': ['Company Name', 'Job Title', 'Skills Requirements', 
                               'Job Description', 'Duration', 'JobType']
        }
    
    def process_dataframe(self, df: pd.DataFrame, config: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """
        Complete preprocessing pipeline.
        
        Args:
            df: Input dataframe
            config: Configuration dictionary
        
        Returns:
            Fully processed dataframe
        """
        config = config or self.get_default_config()
        
        # Handle missing values
        df = self.missing_handler.handle_missing_values(
            df, strategy="fill", required_columns=config['required_columns']
        )
        
        # Process text columns
        df = self.text_processor.batch_process_columns(df, config['text_columns'])
        
        # Process skills
        df = self.skills_processor.process_skills_column(df, config['skills_column'])
        
        # Standardize categorical values
        df = self.categorical_processor.standardize_values(df, config['work_mode_column'])
        
        # Create features
        df = self.feature_engineer.create_features(df)
        
        # Add unique IDs
        df = self.feature_engineer.create_id_column(df)
        
        return df
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get comprehensive processing statistics."""
        return {
            **self.text_processor.get_processing_stats(),
            "features_created": len(self.feature_engineer.feature_configs),
            "categorical_mappings": len(self.categorical_processor.value_mappings)
        }

# Factory functions for backward compatibility
def safe_literal_eval(value: Any) -> Any:
    """Factory function for text processing."""
    processor = TextProcessor()
    return processor.safe_literal_eval(value)

def clean_text(text: str) -> str:
    """Factory function for text cleaning."""
    processor = TextProcessor()
    return processor.clean_text(text)

def handle_missing_values(df: pd.DataFrame, strategy: str = "drop", 
                         required_columns: Optional[List[str]] = None,
                         fill_defaults: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    """Factory function for handling missing values."""
    handler = MissingValueHandler(strategy)
    if fill_defaults:
        handler.set_fill_defaults(fill_defaults)
    return handler.handle_missing_values(df, strategy, required_columns)

def create_preprocessing_pipeline() -> PreprocessingPipeline:
    """Factory function to create preprocessing pipeline."""
    return PreprocessingPipeline()