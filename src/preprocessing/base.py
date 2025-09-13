import pandas as pd
from abc import ABC, abstractmethod
from typing import Optional
from utils.logger import BaseLogger

class BaseDataProcessor(ABC):
    """Abstract base class for data processing operations."""
    
    def __init__(self, data_path: Optional[str] = None, logger_name: str = "DataProcessor"):
        self.logger = BaseLogger(logger_name)
        self.data_path = data_path
        self._raw_data = None
        self._processed_data = None
    
    @abstractmethod
    def load_data(self) -> pd.DataFrame:
        """Abstract method to load data."""
        pass
    
    @abstractmethod
    def process_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Abstract method to process data."""
        pass
    
    @property
    def raw_data(self) -> Optional[pd.DataFrame]:
        """Get raw data."""
        return self._raw_data
    
    @property
    def processed_data(self) -> Optional[pd.DataFrame]:
        """Get processed data."""
        return self._processed_data
    
class DataLoader:
    """Class responsible for loading data from various sources."""

    def __init__(self, logger: BaseLogger):
        self.logger = logger

    def load_csv(self, file_path: str) -> pd.DataFrame:
        """Load data from CSV file."""
        try:
            self.logger.info(f"Loading data from {file_path}")
            df = pd.read_csv(file_path)
            self.logger.info(f"Successfully loaded data with shape: {df.shape}")
            return df
        except FileNotFoundError:
            self.logger.error(f"Data file not found at {file_path}")
            raise
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            raise

class DataSaver:
    """Class responsible for saving processed data."""

    def __init__(self, logger: BaseLogger):
        self.logger = logger

    def save_to_pickle(self, df: pd.DataFrame, file_path: str) -> None:
        """Save dataframe to pickle format."""
        try:
            # Ensure directory exists
            from pathlib import Path
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)

            df.to_pickle(file_path)
            self.logger.info(f"Data saved to {file_path}")
        except Exception as e:
            self.logger.error(f"Error saving data: {str(e)}")
            raise

    def save_to_csv(self, df: pd.DataFrame, file_path: str) -> None:
        """Save dataframe to CSV format."""
        try:
            from pathlib import Path
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)

            df.to_csv(file_path, index=False)
            self.logger.info(f"Data saved to {file_path}")
        except Exception as e:
            self.logger.error(f"Error saving data: {str(e)}")
            raise