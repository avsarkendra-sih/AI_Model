"""
Data preparation class for loading, cleaning, and preprocessing job posting data.
"""
import pandas as pd
from typing import Optional, Dict, Any
import sys
import os
from config.paths import RAW_DATA_DIR, PROCESSED_DATA_DIR, RAW_DATA_FILE, PROCESSED_DATA_FILE, ColumnNames
from preprocessing.schema import JobPostingSchema
from utils.logger import BaseLogger, DataValidator, DataSummarizer
from utils.helper import PreprocessingPipeline
from preprocessing.base import BaseDataProcessor, DataLoader, DataSaver

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class JobPostingDataPreparer(BaseDataProcessor):
    """Specialized data preparer for job posting data using OOP principles."""

    def __init__(self, data_path: Optional[str] = None):
        super().__init__(data_path or str(RAW_DATA_DIR / RAW_DATA_FILE), "JobPostingDataPreparer")

        # Initialize components
        self.data_loader = DataLoader(self.logger)
        self.data_saver = DataSaver(self.logger)
        self.validator = DataValidator(self.logger)
        self.summarizer = DataSummarizer()
        self.pipeline = PreprocessingPipeline()

        # Configuration
        self.config = self._get_configuration()

        self.logger.info("JobPostingDataPreparer initialized with OOP design")

    def _get_configuration(self) -> Dict[str, Any]:
        """Get job posting specific configuration."""
        return {
            'required_columns': [
                ColumnNames.COMPANY, ColumnNames.TITLE, ColumnNames.SKILLS,
                ColumnNames.LOCATION, ColumnNames.MODE, ColumnNames.DESCRIPTION,
                ColumnNames.DURATION, ColumnNames.JOB_TYPE
            ],
            'summary_config': {
                "companies_count": ColumnNames.COMPANY,
                "job_types_count": ColumnNames.JOB_TYPE,
                "locations_count": ColumnNames.LOCATION,
                "modes_distribution": ColumnNames.MODE,
                "avg_skills_per_job": 'skills_count',
                "avg_duration": ColumnNames.DURATION
            },
            'processing_config': {
                'text_columns': [ColumnNames.DESCRIPTION, ColumnNames.SPECIAL_REQS, ColumnNames.TITLE],
                'work_mode_column': ColumnNames.MODE,
                'skills_column': ColumnNames.SKILLS,
                'required_columns': [
                    ColumnNames.COMPANY, ColumnNames.TITLE, ColumnNames.SKILLS,
                    ColumnNames.LOCATION, ColumnNames.MODE, ColumnNames.DESCRIPTION,
                    ColumnNames.DURATION, ColumnNames.JOB_TYPE
                ]
            }
        }

    def load_data(self) -> pd.DataFrame:
        """Load job posting data."""
        self._raw_data = self.data_loader.load_csv(self.data_path)
        return self._raw_data

    def validate_data(self, df: pd.DataFrame) -> bool:
        """Validate loaded data against job posting schema."""
        def schema_validator(row):
            """Validate individual row against JobPostingSchema."""
            row_dict = {
                'skills': row.get(ColumnNames.SKILLS, []),
                'company': row.get(ColumnNames.COMPANY),
                'title': row.get(ColumnNames.TITLE),
                'location': row.get(ColumnNames.LOCATION),
                'mode': row.get(ColumnNames.MODE),
                'description': row.get(ColumnNames.DESCRIPTION),
                'special_requirements': row.get(ColumnNames.SPECIAL_REQS, ""),
                'duration': row.get(ColumnNames.DURATION),
                'job_type': row.get(ColumnNames.JOB_TYPE),
                'job_id': "temp_id"
            }
            JobPostingSchema(**row_dict)

        is_valid, errors = self.validator.validate_schema(
            df, self.config['required_columns'], schema_validator
        )

        if not is_valid:
            self.logger.warning(f"Data validation completed with issues: {len(errors)} errors found")
        else:
            self.logger.info("Data validation passed successfully")

        return is_valid

    def process_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process job posting data using the preprocessing pipeline."""
        self.logger.info("Starting data processing with OOP pipeline")
        processed_df = self.pipeline.process_dataframe(df, self.config['processing_config'])
        # Log processing statistics
        stats = self.pipeline.get_processing_stats()
        self.logger.info(f"Processing completed. Stats: {stats}")

        return processed_df

    def prepare_data(self, save_processed: bool = True) -> pd.DataFrame:
        """
        Complete data preparation pipeline using OOP approach.
        Args:
            save_processed: Whether to save the processed data
        Returns:
            Processed DataFrame
        """
        self.logger.info("Starting complete data preparation pipeline")

        # Load data
        raw_df = self.load_data()

        # Validate data (non-blocking)
        self.validate_data(raw_df)

        # Process data
        self._processed_data = self.process_data(raw_df)

        # Save if requested
        if save_processed:
            self.save_processed_data()

        self.logger.info("Data preparation pipeline completed successfully")
        return self._processed_data

    def save_processed_data(self, file_path: Optional[str] = None, format_type: str = "pickle") -> None:
        """
        Save processed data using the data saver.
        Args:
            file_path: Custom file path (optional)
            format_type: Format to save in ('pickle' or 'csv')
        """
        if self._processed_data is None:
            self.logger.error("No processed data to save. Run prepare_data() first.")
            return

        save_path = file_path or str(PROCESSED_DATA_DIR / PROCESSED_DATA_FILE)

        if format_type.lower() == "pickle":
            self.data_saver.save_to_pickle(self._processed_data, save_path)
        elif format_type.lower() == "csv":
            csv_path = save_path.replace('.pkl', '.csv')
            self.data_saver.save_to_csv(self._processed_data, csv_path)
        else:
            raise ValueError(f"Unsupported format: {format_type}")

    def get_data_summary(self, print_summary: bool = False) -> Dict[str, Any]:
        """
        Generate comprehensive data summary.
        Args:
            print_summary: Whether to print formatted summary
        Returns:
            Summary dictionary
        """
        if self._processed_data is None:
            self.logger.warning("No processed data available for summary")
            return {}

        summary = self.summarizer.generate_summary(
            self._processed_data,
            self.config['summary_config']
        )

        if print_summary:
            self.summarizer.print_summary(summary, "JOB POSTING DATA SUMMARY")

        return summary

    def get_sample_data(self, n_rows: int = 3) -> Optional[pd.DataFrame]:
        """Get sample of processed data."""
        if self._processed_data is None:
            return None
        return self._processed_data.head(n_rows)

# Factory function for backward compatibility
def create_job_posting_preparer(data_path: Optional[str] = None) -> JobPostingDataPreparer:
    """Factory function to create JobPostingDataPreparer instance."""
    return JobPostingDataPreparer(data_path)

if __name__ == "__main__":
    """Main execution block demonstrating OOP usage."""
    import logging
    logging.basicConfig(level=logging.INFO)

    # Create preparer instance
    preparer = JobPostingDataPreparer()

    try:
        # Run complete pipeline
        processed_data = preparer.prepare_data(save_processed=True)

        # Generate and display summary
        summary = preparer.get_data_summary(print_summary=True)

        # Show sample data
        sample = preparer.get_sample_data(3)
        if sample is not None:
            print(f"\nSample processed data:\n{sample.to_string()}")

        print("\n✅ Data preparation completed successfully using OOP approach!")

    except Exception as e:
        print(f"❌ Error during data preparation: {str(e)}")
        import traceback
        traceback.print_exc()