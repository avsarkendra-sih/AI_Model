"""
Categorical feature encoding for job types, locations, and modes.
"""
import pandas as pd
from typing import List, Dict, Optional
from sklearn.preprocessing import LabelEncoder
import pickle

from utils.logger import BaseLogger

class CategoricalEncoder:
    """Class for encoding categorical features."""
    
    def __init__(self, logger: Optional[BaseLogger] = None):
        """
        Initialize the categorical encoder.
        
        Args:
            logger: Logger instance
        """
        self.logger = logger or BaseLogger("CategoricalEncoder")
        self.encoders = {}
        self.categories = {}
    
    def fit(self, df: pd.DataFrame, categorical_columns: List[str]) -> None:
        """
        Fit encoders on categorical data.
        
        Args:
            df: DataFrame with categorical data
            categorical_columns: List of categorical column names
        """
        self.logger.info(f"Fitting encoders for columns: {categorical_columns}")
        
        for col in categorical_columns:
            if col not in df.columns:
                self.logger.warning(f"Column {col} not found in DataFrame")
                continue
            
            # Get unique values and store
            unique_values = df[col].unique().tolist()
            self.categories[col] = unique_values
            
            # Create and fit label encoder
            encoder = LabelEncoder()
            encoder.fit(df[col].astype(str))
            self.encoders[col] = encoder
            
            self.logger.info(f"Fitted encoder for {col} with {len(unique_values)} categories")
    
    def transform(self, df: pd.DataFrame, categorical_columns: List[str]) -> pd.DataFrame:
        """
        Transform categorical data using fitted encoders.
        
        Args:
            df: DataFrame with categorical data
            categorical_columns: List of categorical column names
            
        Returns:
            DataFrame with encoded categorical features
        """
        result_df = df.copy()
        
        for col in categorical_columns:
            if col not in self.encoders:
                self.logger.warning(f"Encoder for {col} not found. Skipping.")
                continue
            
            if col not in df.columns:
                self.logger.warning(f"Column {col} not found in DataFrame")
                continue
            
            # Transform the column
            try:
                result_df[col] = self.encoders[col].transform(df[col].astype(str))
            except ValueError as e:
                self.logger.warning(f"Error encoding {col}: {str(e)}. Using default value.")
                # Handle unseen labels by using a default value (e.g., -1)
                result_df[col] = -1
        
        return result_df
    
    def fit_transform(self, df: pd.DataFrame, categorical_columns: List[str]) -> pd.DataFrame:
        """
        Fit and transform categorical data.
        
        Args:
            df: DataFrame with categorical data
            categorical_columns: List of categorical column names
            
        Returns:
            DataFrame with encoded categorical features
        """
        self.fit(df, categorical_columns)
        return self.transform(df, categorical_columns)
    
    def inverse_transform(self, df: pd.DataFrame, categorical_columns: List[str]) -> pd.DataFrame:
        """
        Inverse transform encoded categorical data.
        
        Args:
            df: DataFrame with encoded categorical data
            categorical_columns: List of categorical column names
            
        Returns:
            DataFrame with original categorical values
        """
        result_df = df.copy()
        
        for col in categorical_columns:
            if col not in self.encoders:
                self.logger.warning(f"Encoder for {col} not found. Skipping.")
                continue
            
            if col not in df.columns:
                self.logger.warning(f"Column {col} not found in DataFrame")
                continue
            
            # Inverse transform the column
            try:
                result_df[col] = self.encoders[col].inverse_transform(df[col].astype(int))
            except ValueError as e:
                self.logger.warning(f"Error decoding {col}: {str(e)}")
        
        return result_df
    
    def save(self, file_path: str) -> None:
        """
        Save encoders to file.
        
        Args:
            file_path: Path to save the encoders
        """
        try:
            with open(file_path, 'wb') as f:
                pickle.dump({
                    'encoders': self.encoders,
                    'categories': self.categories
                }, f)
            self.logger.info(f"Encoders saved to {file_path}")
        except Exception as e:
            self.logger.error(f"Error saving encoders: {str(e)}")
            raise
    
    def load(self, file_path: str) -> None:
        """
        Load encoders from file.
        
        Args:
            file_path: Path to load the encoders from
        """
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
                self.encoders = data['encoders']
                self.categories = data['categories']
            self.logger.info(f"Encoders loaded from {file_path}")
        except Exception as e:
            self.logger.error(f"Error loading encoders: {str(e)}")
            raise

class PreferenceEncoder:
    """Class for encoding user preferences."""
    
    def __init__(self, logger: Optional[BaseLogger] = None):
        """
        Initialize the preference encoder.
        
        Args:
            logger: Logger instance
        """
        self.logger = logger or BaseLogger("PreferenceEncoder")
        self.encoder = CategoricalEncoder(logger)
    
    def encode_user_preferences(self, user_preferences: Dict[str, any], 
                               available_categories: Dict[str, List[str]]) -> Dict[str, any]:
        """
        Encode user preferences for matching with job features.
        
        Args:
            user_preferences: Dictionary of user preferences
            available_categories: Dictionary of available categories for each feature
            
        Returns:
            Dictionary with encoded preferences
        """
        encoded_preferences = {}
        
        for feature, preference in user_preferences.items():
            if feature not in available_categories:
                continue
            
            if isinstance(preference, list):
                # Multiple preferences (e.g., preferred locations)
                encoded = []
                for pref in preference:
                    if pref in available_categories[feature]:
                        encoded.append(available_categories[feature].index(pref))
                encoded_preferences[feature] = encoded
            else:
                # Single preference
                if preference in available_categories[feature]:
                    encoded_preferences[feature] = available_categories[feature].index(preference)
                else:
                    encoded_preferences[feature] = -1  # Not found
        
        return encoded_preferences
    
    def calculate_preference_score(self, job_value: any, user_preference: any, 
                                 feature_type: str) -> float:
        """
        Calculate preference matching score.
        
        Args:
            job_value: Job feature value
            user_preference: User preference value
            feature_type: Type of feature ('binary', 'categorical', 'multi_value')
            
        Returns:
            Preference matching score (0-1)
        """
        if user_preference is None:
            return 0.5  # Neutral score if no preference
        
        if feature_type == 'binary':
            # Exact match for binary features (e.g., remote vs on-site)
            return 1.0 if job_value == user_preference else 0.0
        
        elif feature_type == 'categorical':
            # Exact match for categorical features (e.g., job type)
            return 1.0 if job_value == user_preference else 0.0
        
        elif feature_type == 'multi_value':
            # Partial match for multi-value features (e.g., preferred locations)
            if isinstance(user_preference, list):
                return 1.0 if job_value in user_preference else 0.0
            else:
                return 1.0 if job_value == user_preference else 0.0
        
        return 0.0