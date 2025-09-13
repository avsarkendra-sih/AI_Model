"""
Feature store for storing and retrieving processed features.
"""
from typing import Dict, List, Optional, Any
import pickle
from pathlib import Path
import json

from utils.logger import BaseLogger

class FeatureStore:
    """Class for storing and retrieving processed features."""
    
    def __init__(self, store_path: str, logger: Optional[BaseLogger] = None):
        """
        Initialize the feature store.
        
        Args:
            store_path: Path to the feature store directory
            logger: Logger instance
        """
        self.store_path = Path(store_path)
        self.logger = logger or BaseLogger("FeatureStore")
        self.features = {}
        
        # Create directory if it doesn't exist
        self.store_path.mkdir(parents=True, exist_ok=True)
    
    def add_feature(self, feature_name: str, feature_data: Any, 
                   metadata: Optional[Dict] = None) -> None:
        """
        Add a feature to the store.
        
        Args:
            feature_name: Name of the feature
            feature_data: Feature data
            metadata: Optional metadata about the feature
        """
        self.features[feature_name] = {
            'data': feature_data,
            'metadata': metadata or {}
        }
        self.logger.info(f"Added feature '{feature_name}' to store")
    
    def get_feature(self, feature_name: str) -> Optional[Any]:
        """
        Get a feature from the store.
        
        Args:
            feature_name: Name of the feature
            
        Returns:
            Feature data or None if not found
        """
        if feature_name in self.features:
            return self.features[feature_name]['data']
        return None
    
    def get_feature_with_metadata(self, feature_name: str) -> Optional[Dict]:
        """
        Get a feature with its metadata.
        
        Args:
            feature_name: Name of the feature
            
        Returns:
            Dictionary with feature data and metadata, or None if not found
        """
        return self.features.get(feature_name)
    
    def save_feature(self, feature_name: str, file_name: Optional[str] = None) -> None:
        """
        Save a feature to disk.
        
        Args:
            feature_name: Name of the feature to save
            file_name: Custom file name (optional)
        """
        if feature_name not in self.features:
            self.logger.warning(f"Feature '{feature_name}' not found in store")
            return
        
        file_name = file_name or f"{feature_name}.pkl"
        file_path = self.store_path / file_name
        
        try:
            with open(file_path, 'wb') as f:
                pickle.dump(self.features[feature_name], f)
            self.logger.info(f"Saved feature '{feature_name}' to {file_path}")
        except Exception as e:
            self.logger.error(f"Error saving feature '{feature_name}': {str(e)}")
            raise
    
    def load_feature(self, file_name: str, feature_name: Optional[str] = None) -> None:
        """
        Load a feature from disk.
        
        Args:
            file_name: Name of the file to load
            feature_name: Custom feature name (optional)
        """
        file_path = self.store_path / file_name
        
        if not file_path.exists():
            self.logger.warning(f"Feature file {file_path} not found")
            return
        
        try:
            with open(file_path, 'rb') as f:
                feature_data = pickle.load(f)
            
            feature_name = feature_name or file_name.split('.')[0]
            self.features[feature_name] = feature_data
            self.logger.info(f"Loaded feature '{feature_name}' from {file_path}")
        except Exception as e:
            self.logger.error(f"Error loading feature from {file_path}: {str(e)}")
            raise
    
    def save_all_features(self) -> None:
        """Save all features to disk."""
        self.logger.info(f"Saving all features to {self.store_path}")
        
        for feature_name in self.features:
            self.save_feature(feature_name)
    
    def load_all_features(self) -> None:
        """Load all features from disk."""
        self.logger.info(f"Loading all features from {self.store_path}")
        
        for file_path in self.store_path.glob("*.pkl"):
            self.load_feature(file_path.name)
    
    def get_feature_names(self) -> List[str]:
        """Get list of all feature names in the store."""
        return list(self.features.keys())
    
    def clear(self) -> None:
        """Clear all features from the store."""
        self.features.clear()
        self.logger.info("Cleared all features from store")

class JobFeatureStore(FeatureStore):
    """Specialized feature store for job features."""
    
    def __init__(self, store_path: str, logger: Optional[BaseLogger] = None):
        """
        Initialize the job feature store.
        
        Args:
            store_path: Path to the feature store directory
            logger: Logger instance
        """
        super().__init__(store_path, logger)
        self.job_ids = []
    
    def add_job_features(self, job_id: str, features: Dict[str, Any], 
                        metadata: Optional[Dict] = None) -> None:
        """
        Add features for a specific job.
        
        Args:
            job_id: Job ID
            features: Dictionary of features
            metadata: Optional metadata
        """
        feature_name = f"job_{job_id}"
        self.features[feature_name] = {
            'data': features,
            'metadata': metadata or {},
            'job_id': job_id
        }
        
        if job_id not in self.job_ids:
            self.job_ids.append(job_id)
        
        self.logger.info(f"Added features for job {job_id}")
    
    def get_job_features(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Get features for a specific job.
        
        Args:
            job_id: Job ID
            
        Returns:
            Dictionary of job features or None if not found
        """
        feature_name = f"job_{job_id}"
        feature_data = self.get_feature(feature_name)
        return feature_data
    
    def get_all_job_features(self) -> Dict[str, Dict[str, Any]]:
        """
        Get features for all jobs.
        
        Returns:
            Dictionary mapping job IDs to their features
        """
        all_features = {}
        
        for job_id in self.job_ids:
            features = self.get_job_features(job_id)
            if features:
                all_features[job_id] = features
        
        return all_features
    
    def get_job_ids(self) -> List[str]:
        """Get list of all job IDs in the store."""
        return self.job_ids.copy()