# transformation/main.py
"""
Main script for running the transformation pipeline.
"""

import pandas as pd
from pathlib import Path
import sys
from utils.logger import BaseLogger
from transformation.text_embedder import TextEmbedder, TextEmbeddingManager
from transformation.similarity_pipeline import SimilarityPipeline
from transformation.feature_store import JobFeatureStore

# Add the project root to Python path
sys.path.append(str(Path(__file__).parent.parent))

def main():
    """Run the transformation pipeline."""
    logger = BaseLogger("TransformationPipeline")
    logger.info("Starting transformation pipeline")
    
    # Load processed data from Stage 0
    processed_data_path = Path("data/processed/processed_job_postings.pkl")
    if not processed_data_path.exists():
        logger.error("Processed data not found. Run Stage 0 first.")
        return
    
    try:
        df = pd.read_pickle(processed_data_path)
        logger.info(f"Loaded processed data with shape: {df.shape}")
    except Exception as e:
        logger.error(f"Error loading processed data: {str(e)}")
        return
    
    # Initialize components
    embedder = TextEmbedder()
    embedder_manager = TextEmbeddingManager(embedder)
    similarity_pipeline = SimilarityPipeline(embedder)
    feature_store = JobFeatureStore("data/features")
    
    # Precompute job description embeddings
    job_descriptions = df['Job Description'].tolist()
    job_embeddings = embedder.embed_batch(job_descriptions, description="job descriptions")
    
    # Store job features
    for idx, row in df.iterrows():
        job_features = {
            'job_id': row['job_id'],
            'description': row['Job Description'],
            'skills': row['Skills Requirements'],
            'job_type': row['JobType'],
            'location': row['Location'],
            'mode': row['Mode'],
            'duration': row['Duration'],
            'company': row['Company Name'],
            'title': row['Job Title']
        }
        
        feature_store.add_job_features(row['job_id'], job_features)
    
    # Save features
    feature_store.save_all_features()
    logger.info("Transformation pipeline completed successfully")
    
    # Test with a sample user profile
    sample_user = {
        'description': "Experienced software engineer with expertise in Python, Django, and React. \
                       Looking for remote opportunities in web development.",
        'skills': ['Python', 'Django', 'React', 'JavaScript', 'SQL'],
        'preferences': {
            'job_type': 'Software Development',
            'location': ['Remote', 'Hybrid'],
            'mode': 'Online'
        }
    }
    
    # Get all job features
    all_job_features = list(feature_store.get_all_job_features().values())
    
    # Get recommendations
    recommendations = similarity_pipeline.recommend_jobs(sample_user, all_job_features, top_n=3)
    
    logger.info("Sample recommendations generated:")
    for i, rec in enumerate(recommendations):
        logger.info(f"{i+1}. Job ID: {rec['job_id']}, Score: {rec['composite_score']:.3f}")
    
    logger.info("Transformation and testing completed successfully")

if __name__ == "__main__":
    main()