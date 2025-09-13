"""
Data preparation class for loading, cleaning, and preprocessing job posting data.
"""

import pandas as pd
from typing import Dict, List, Optional, Tuple
import json
import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.paths import RAW_DATA_DIR, PROCESSED_DATA_DIR, RAW_DATA_FILE, PROCESSED_DATA_FILE, ColumnNames
from preprocessing.schema import JobPostingSchema
from utils.logger import setup_logger
from utils.helper import safe_literal_eval, clean_text, handle_missing_values

class DataPreparer:
    """Class to prepare job posting data for the recommendation system."""
    
    def __init__(self, data_path: Optional[str] = None):
        """
        Initialize the DataPreparer.
        
        Args:
            data_path (str, optional): Path to raw data file. If None, uses default path.
        """
        self.logger = setup_logger("DataPreparer")
        self.data_path = data_path or (RAW_DATA_DIR / RAW_DATA_FILE)
        self.df = None
        self.processed_df = None
        
        self.logger.info("DataPreparer initialized")
    
    def load_data(self) -> pd.DataFrame:
        """
        Load raw data from CSV file.
        
        Returns:
            pd.DataFrame: Loaded dataframe
            
        Raises:
            FileNotFoundError: If the data file doesn't exist
        """
        try:
            self.logger.info(f"Loading data from {self.data_path}")
            self.df = pd.read_csv(self.data_path)
            self.logger.info(f"Successfully loaded data with shape: {self.df.shape}")
            return self.df
        except FileNotFoundError:
            self.logger.error(f"Data file not found at {self.data_path}")
            raise
    
    def validate_data(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate data against the schema and check for data quality issues.
        
        Args:
            df (pd.DataFrame): Dataframe to validate
            
        Returns:
            Tuple[bool, List[str]]: (is_valid, list_of_errors)
        """
        errors = []
        
        # Check required columns (including Duration and JobType)
        required_columns = [ColumnNames.COMPANY, ColumnNames.TITLE, ColumnNames.SKILLS, 
                           ColumnNames.LOCATION, ColumnNames.MODE, ColumnNames.DESCRIPTION,
                           ColumnNames.DURATION, ColumnNames.JOB_TYPE]
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            errors.append(f"Missing required columns: {missing_columns}")
        
        # Check for missing values in critical columns
        for col in required_columns:
            if col in df.columns and df[col].isnull().any():
                null_count = df[col].isnull().sum()
                errors.append(f"Column {col} has {null_count} missing values")
        
        # Validate a sample of rows against the schema
        sample_size = min(10, len(df))
        for idx, row in df.head(sample_size).iterrows():
            try:
                # Convert to dict and prepare for validation
                row_dict = row.to_dict()
                # Handle skills field
                row_dict['skills'] = safe_literal_eval(row_dict.get(ColumnNames.SKILLS, []))
                row_dict['company'] = row_dict.get(ColumnNames.COMPANY)
                row_dict['title'] = row_dict.get(ColumnNames.TITLE)
                row_dict['location'] = row_dict.get(ColumnNames.LOCATION)
                row_dict['mode'] = row_dict.get(ColumnNames.MODE)
                row_dict['description'] = row_dict.get(ColumnNames.DESCRIPTION)
                row_dict['special_requirements'] = row_dict.get(ColumnNames.SPECIAL_REQS, "")
                row_dict['duration'] = row_dict.get(ColumnNames.DURATION)
                row_dict['job_type'] = row_dict.get(ColumnNames.JOB_TYPE)
                row_dict['job_id'] = "temp_id"  # Temporary ID for validation
                
                # Validate against schema
                JobPostingSchema(**row_dict)
            except Exception as e:
                errors.append(f"Validation error in row {idx}: {str(e)}")
        
        is_valid = len(errors) == 0
        return is_valid, errors
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the raw data by handling missing values and standardizing formats.
        
        Args:
            df (pd.DataFrame): Raw dataframe to clean
            
        Returns:
            pd.DataFrame: Cleaned dataframe
        """
        self.logger.info("Starting data cleaning")
        
        # Create a copy to avoid modifying the original
        cleaned_df = df.copy()
        
        # Handle missing values
        cleaned_df = handle_missing_values(cleaned_df, strategy="fill")
        
        # Clean text fields
        text_columns = [ColumnNames.DESCRIPTION, ColumnNames.SPECIAL_REQS, ColumnNames.TITLE]
        for col in text_columns:
            if col in cleaned_df.columns:
                cleaned_df[col] = cleaned_df[col].apply(clean_text)
        
        # Standardize skills format
        if ColumnNames.SKILLS in cleaned_df.columns:
            cleaned_df[ColumnNames.SKILLS] = cleaned_df[ColumnNames.SKILLS].apply(
                lambda x: safe_literal_eval(x) if isinstance(x, str) else x
            )
        
        # Standardize mode values
        mode_mapping = {
            'remote': 'Online',
            'hybrid': 'Hybrid',
            'offline': 'Offline',
            'on-site': 'Offline',
            'online': 'Online'
        }
        
        if ColumnNames.MODE in cleaned_df.columns:
            cleaned_df[ColumnNames.MODE] = cleaned_df[ColumnNames.MODE].str.lower().map(
                lambda x: mode_mapping.get(x, x.capitalize())
            )
        
        self.logger.info("Data cleaning completed")
        return cleaned_df
    
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess the data for use in the recommendation system.
        
        Args:
            df (pd.DataFrame): Cleaned dataframe to preprocess
            
        Returns:
            pd.DataFrame: Preprocessed dataframe with additional features
        """
        self.logger.info("Starting data preprocessing")
        
        # Create a copy to avoid modifying the original
        processed_df = df.copy()
        
        # Add a unique ID for each job posting
        processed_df['job_id'] = [f"{i:08x}" for i in range(1, len(processed_df) + 1)]
        
        # Extract skills count
        if ColumnNames.SKILLS in processed_df.columns:
            processed_df['skills_count'] = processed_df[ColumnNames.SKILLS].apply(
                lambda x: len(x) if isinstance(x, list) else 0
            )
        
        # Add text length features
        if ColumnNames.DESCRIPTION in processed_df.columns:
            processed_df['description_length'] = processed_df[ColumnNames.DESCRIPTION].apply(
                lambda x: len(x.split()) if isinstance(x, str) else 0
            )
        
        self.logger.info("Data preprocessing completed")
        return processed_df
    
    def prepare_data(self, save_processed: bool = True) -> pd.DataFrame:
        """
        Complete data preparation pipeline.
        
        Args:
            save_processed (bool): Whether to save the processed data to disk
            
        Returns:
            pd.DataFrame: Prepared dataframe ready for the recommendation system
        """
        # Load data
        raw_df = self.load_data()
        
        # Validate data
        is_valid, errors = self.validate_data(raw_df)
        if not is_valid:
            self.logger.warning(f"Data validation issues found: {errors}")
        
        # Clean data
        cleaned_df = self.clean_data(raw_df)
        
        # Preprocess data
        self.processed_df = self.preprocess_data(cleaned_df)
        
        # Save processed data
        if save_processed:
            self.save_processed_data()
        
        self.logger.info("Data preparation completed successfully")
        return self.processed_df
    
    def save_processed_data(self, file_path: Optional[str] = None) -> None:
        """
        Save the processed data to disk.
        
        Args:
            file_path (str, optional): Path to save the processed data. 
                                      If None, uses default path.
        """
        if self.processed_df is None:
            self.logger.error("No processed data to save. Run prepare_data() first.")
            return
        
        save_path = file_path or (PROCESSED_DATA_DIR / PROCESSED_DATA_FILE)
        
        # Ensure directory exists
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save as pickle to preserve data types
        self.processed_df.to_pickle(save_path)
        self.logger.info(f"Processed data saved to {save_path}")
    
    def get_data_summary(self) -> Dict:
        """
        Generate a summary of the processed data.
        
        Returns:
            Dict: Summary statistics of the data
        """
        if self.processed_df is None:
            self.logger.error("No processed data available. Run prepare_data() first.")
            return {}
        
        summary = {
            "total_jobs": len(self.processed_df),
            "companies_count": self.processed_df[ColumnNames.COMPANY].nunique(),
            "job_types_count": self.processed_df[ColumnNames.JOB_TYPE].nunique() if ColumnNames.JOB_TYPE in self.processed_df.columns else 0,
            "locations_count": self.processed_df[ColumnNames.LOCATION].nunique(),
            "modes_distribution": self.processed_df[ColumnNames.MODE].value_counts().to_dict() if ColumnNames.MODE in self.processed_df.columns else {},
            "avg_skills_per_job": self.processed_df['skills_count'].mean() if 'skills_count' in self.processed_df.columns else 0,
            "avg_duration": self.processed_df[ColumnNames.DURATION].mean() if ColumnNames.DURATION in self.processed_df.columns else 0
        }
        
        return summary

if __name__ == "__main__":
    """Main block to test the data preparation pipeline."""
    import logging
    
    # Set up logging to see detailed output
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Initialize the data preparer
    preparer = DataPreparer()
    
    try:
        # Run the complete data preparation pipeline
        processed_data = preparer.prepare_data(save_processed=True)
        
        # Get and display data summary
        summary = preparer.get_data_summary()
        
        print("\n" + "="*50)
        print("DATA PREPARATION SUMMARY")
        print("="*50)
        print(f"Total jobs processed: {summary['total_jobs']}")
        print(f"Unique companies: {summary['companies_count']}")
        print(f"Unique job types: {summary['job_types_count']}")
        print(f"Unique locations: {summary['locations_count']}")
        print("Work mode distribution:")
        for mode, count in summary['modes_distribution'].items():
            print(f"  {mode}: {count}")
        print(f"Average skills per job: {summary['avg_skills_per_job']:.2f}")
        print(f"Average duration: {summary['avg_duration']:.2f} months")
        print("="*50)
        
        # Display a sample of the processed data
        print("\nSAMPLE OF PROCESSED DATA:")
        print(processed_data.head(3).to_string())
        
        print("\nData preparation completed successfully!")
        
    except Exception as e:
        print(f"Error during data preparation: {str(e)}")
        import traceback
        traceback.print_exc()