"""Configuration for data paths and constants."""

from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Data paths
RAW_DATA_DIR = BASE_DIR.parent / "data" / "raw"
PROCESSED_DATA_DIR = BASE_DIR.parent / "data" / "processed"

# File names
RAW_DATA_FILE = "jobPostings.csv"
PROCESSED_DATA_FILE = "processed_job_postings.pkl"

# Column names
class ColumnNames:
    JOB_ID = "JobID"
    COMPANY = "Company Name"
    TITLE = "Job Title"
    SKILLS = "Skills Requirements"
    LOCATION = "Location"
    MODE = "Mode"
    DESCRIPTION = "Job Description"
    SPECIAL_REQS = "Special Requirements"
    DURATION = "Duration"
    JOB_TYPE = "JobType"

if __name__ == "__main__":
    print(f"Raw data will be read from: {RAW_DATA_DIR / RAW_DATA_FILE}")
    print(f"Processed data will be saved to: {PROCESSED_DATA_DIR / PROCESSED_DATA_FILE}")