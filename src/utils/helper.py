"""Helper functions for data processing."""

import pandas as pd
import numpy as np
from typing import Any, Dict, List, Union
import ast

def safe_literal_eval(value: Any) -> Any:
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

def clean_text(text: str) -> str:
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
    
    return text

def handle_missing_values(df: pd.DataFrame, strategy: str = "drop") -> pd.DataFrame:
    """
    Handle missing values in the dataframe.
    
    Args:
        df (pd.DataFrame): Input dataframe
        strategy (str): Strategy to handle missing values - "drop" or "fill"
    
    Returns:
        pd.DataFrame: Dataframe with handled missing values
    """
    # First, ensure Duration and JobType are handled as required fields
    required_columns = ['Company Name', 'Job Title', 'Skills Requirements', 
                       'Job Description', 'Duration', 'JobType']
    
    # Drop rows with missing required values
    df = df.dropna(subset=required_columns, how='any')
    
    if strategy == "fill":
        # Fill missing values with appropriate defaults for non-required fields
        df = df.fillna({
            'Special Requirements': '',
            'Location': 'Online',
            'Mode': 'Remote'
        })
    
    return df