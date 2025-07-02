#!/usr/bin/env python3
"""
Complete ETL Pipeline for ClinicalTrials.gov Data

This script fetches clinical trial data from the official ClinicalTrials.gov API,
transforms it into structured metadata and vector embeddings, and loads it into
both SQLite (metadata) and AstraDB (vector embeddings).
"""

import os
import sys
import logging
import argparse
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional

# Add the backend directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from extract_clinicaltrials_gov import (
    extract_trials_from_clinicaltrials_gov, 
    test_clinicaltrials_gov_connection,
    validate_trial_data
)
from transform_clinicaltrials_gov import (
    initialize_sentence_transformer,
    transform_trial_data,
    validate_transformed_data
)
from load_trial_data import load_trial_data
from utils.database_connections import get_sqlite_connection, get_astradb_connection

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('trials_etl_pipeline.log')
    ]
)
logger = logging.getLogger(__name__)


def convert_to_load_format(transformed_data: List[Tuple[Dict[str, Any], List[float]]]) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
    """
    Convert the transformed data format to match what load_trial_data expects.
    
    Args:
        transformed_data: List of (metadata, embedding_vector) tuples
        
    Returns:
        List of (metadata, vector_data) tuples in the format expected by load_trial_data
    """
    logger.info(f"Converting {len(transformed_data)} trials to load format...")
    
    converted_data = []
    
    for metadata, embedding in transformed_data:
        # Create vector_data structure
        vector_data = {
            "nct_id": metadata["nct_id"],
            "embedding": embedding,
            "criteria_text": metadata.get("eligibility_criteria", "")
        }
        
        converted_data.append((metadata, vector_data))
    
    logger.info(f"✅ Converted {len(converted_data)} trials to load format")
    return converted_data


def run_etl_pipeline(
    page_size: int = 1000,
    limit: Optional[int] = None,
    condition: str = "cancer",
    clear_data: bool = True
) -> Dict[str, Any]:
    """
    Run the complete ETL pipeline.
    
    Args:
        page_size: Number of trials to fetch per API call
        limit: Optional limit on total trials to fetch
        condition: Medical condition to search for
        clear_data: Whether to clear existing data before loading
        
    Returns:
        Dictionary with pipeline execution results
    """
    start_time = datetime.now()
    
    logger.info("🚀 Starting Clinical Trials ETL Pipeline")
    logger.info(f"Parameters: page_size={page_size}, limit={limit}, condition={condition}")
    
    # Step 1: Test API Connection
    logger.info("🔌 Testing ClinicalTrials.gov API connection...")
    if not test_clinicaltrials_gov_connection():
        raise Exception("Failed to connect to ClinicalTrials.gov API")
    
    # Step 2: Extract
    logger.info("📥 Starting data extraction...")
    extraction_start = datetime.now()
    
    trials = extract_trials_from_clinicaltrials_gov(
        page_size=page_size,
        limit=limit,
        condition=condition
    )
    
    extraction_end = datetime.now()
    extraction_time = (extraction_end - extraction_start).total_seconds()
    
    if not trials:
        raise Exception("No trials extracted from API")
    
    if not validate_trial_data(trials):
        raise Exception("Extracted trial data failed validation")
    
    logger.info(f"✅ Extracted {len(trials)} trials in {extraction_time:.1f} seconds")
    
    # Step 3: Transform
    logger.info("🔄 Starting data transformation...")
    transformation_start = datetime.now()
    
    # Initialize embedding model
    model = initialize_sentence_transformer()
    
    # Transform trials
    transformed_data = transform_trial_data(trials, model)
    
    transformation_end = datetime.now()
    transformation_time = (transformation_end - transformation_start).total_seconds()
    
    if not transformed_data:
        raise Exception("No trials were successfully transformed")
    
    if not validate_transformed_data(transformed_data):
        raise Exception("Transformed trial data failed validation")
    
    logger.info(f"✅ Transformed {len(transformed_data)} trials in {transformation_time:.1f} seconds")
    
    # Step 4: Convert to load format
    logger.info("🔄 Converting data to load format...")
    load_format_data = convert_to_load_format(transformed_data)
    
    # Step 5: Load
    logger.info("📤 Starting data loading...")
    loading_start = datetime.now()
    
    # Load to both databases
    load_results = load_trial_data(load_format_data)
    
    loading_end = datetime.now()
    loading_time = (loading_end - loading_start).total_seconds()
    
    sqlite_count = load_results.get("sqlite_loaded", 0)
    astra_count = load_results.get("astradb_loaded", 0)
    verification_passed = load_results.get("verification_results", {}).get("verification_passed", False)
    
    logger.info(f"✅ Loaded {sqlite_count} trials to SQLite and {astra_count} trials to AstraDB in {loading_time:.1f} seconds")
    logger.info(f"📊 Data verification: {'✅ PASSED' if verification_passed else '❌ FAILED'}")
    
    # Final summary
    total_time = (datetime.now() - start_time).total_seconds()
    
    results = {
        'trials_extracted': len(trials),
        'trials_transformed': len(transformed_data),
        'trials_loaded_sqlite': sqlite_count,
        'trials_loaded_astra': astra_count,
        'extraction_time_seconds': extraction_time,
        'transformation_time_seconds': transformation_time,
        'loading_time_seconds': loading_time,
        'total_time_seconds': total_time,
        'verification_passed': verification_passed,
        'success': True
    }
    
    logger.info("🎉 ETL Pipeline completed successfully!")
    logger.info(f"📊 Summary: {len(trials)} extracted → {len(transformed_data)} transformed → {sqlite_count}/{astra_count} loaded")
    logger.info(f"⏱️ Total time: {total_time:.1f} seconds")
    
    return results


def main():
    """
    Main function with command-line interface.
    """
    parser = argparse.ArgumentParser(description='Clinical Trials ETL Pipeline')
    
    parser.add_argument(
        '--page-size', 
        type=int, 
        default=1000,
        help='Number of trials to fetch per API call (default: 1000)'
    )
    
    parser.add_argument(
        '--limit', 
        type=int, 
        default=None,
        help='Maximum number of trials to process (for testing)'
    )
    
    parser.add_argument(
        '--condition', 
        type=str, 
        default='cancer',
        help='Medical condition to search for (default: cancer)'
    )
    
    parser.add_argument(
        '--no-clear', 
        action='store_true',
        help='Do not clear existing data before loading (append mode)'
    )
    
    parser.add_argument(
        '--test', 
        action='store_true',
        help='Run in test mode with only 10 trials'
    )
    
    args = parser.parse_args()
    
    # Test mode overrides
    if args.test:
        args.limit = 10
        args.page_size = 10
        logger.info("🧪 Running in TEST MODE with 10 trials")
    
    try:
        results = run_etl_pipeline(
            page_size=args.page_size,
            limit=args.limit,
            condition=args.condition,
            clear_data=not args.no_clear
        )
        
        print("\n" + "="*60)
        print("🎉 ETL PIPELINE COMPLETED SUCCESSFULLY")
        print("="*60)
        print(f"📥 Trials Extracted:     {results['trials_extracted']:,}")
        print(f"🔄 Trials Transformed:   {results['trials_transformed']:,}")
        print(f"📤 Trials Loaded (SQL):  {results['trials_loaded_sqlite']:,}")
        print(f"📤 Trials Loaded (Vec):  {results['trials_loaded_astra']:,}")
        print(f"📊 Data Verification:    {'✅ PASSED' if results['verification_passed'] else '❌ FAILED'}")
        print(f"⏱️ Total Time:          {results['total_time_seconds']:.1f} seconds")
        print("="*60)
        
        sys.exit(0)
        
    except Exception as e:
        logger.error(f"❌ ETL Pipeline failed: {e}")
        print(f"\n❌ ETL Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 