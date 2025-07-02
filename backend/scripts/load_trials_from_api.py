#!/usr/bin/env python3
"""
Clinical Trials ETL Pipeline Script (Task 7)

This script implements the complete Extract, Transform, Load (ETL) pipeline
for clinical trial data from the NCI API to SQLite and AstraDB databases.

This integrates:
- Task 4: Extract Function for NCI API
- Task 5: Transform Function for Trial Data  
- Task 6: Load Function for Database Storage
- Task 7: Main ETL Pipeline Script
"""

import argparse
import logging
import time
import sys
import uuid
from datetime import datetime
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

# Import our ETL components
from backend.scripts.extract_nci_api import extract_trials_from_nci_api, test_nci_api_connection
from backend.scripts.transform_trial_data import (
    initialize_sentence_transformer, 
    transform_trial_data,
    validate_transformed_data
)
from backend.scripts.load_trial_data import load_trial_data
from backend.utils.database_connections import get_sqlite_connection, close_connection

logger = logging.getLogger(__name__)


def setup_logging(log_file: str = None, log_level: str = "INFO") -> None:
    """
    Set up comprehensive logging to both console and file.
    
    Args:
        log_file: Optional path to log file
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
    """
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    handlers = [logging.StreamHandler()]
    
    if log_file:
        # Ensure log directory exists
        log_dir = Path(log_file).parent
        log_dir.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=handlers,
        force=True  # Override any existing configuration
    )


def update_pipeline_metadata(
    pipeline_run_id: str, 
    status: str, 
    total_processed: int = 0,
    total_loaded: int = 0,
    error_message: str = ""
) -> None:
    """
    Update pipeline execution metadata in the database.
    
    Args:
        pipeline_run_id: Unique identifier for this pipeline run
        status: Current status (running, completed, failed)
        total_processed: Number of trials processed
        total_loaded: Number of trials successfully loaded
        error_message: Error message if status is failed
    """
    try:
        connection = get_sqlite_connection()
        cursor = connection.cursor()
        
        if status == "running":
            # Insert initial record
            cursor.execute("""
                INSERT OR REPLACE INTO pipeline_metadata 
                (pipeline_run_id, start_time, status, total_trials_processed, total_trials_loaded)
                VALUES (?, ?, ?, ?, ?)
            """, (pipeline_run_id, datetime.now().isoformat(), status, total_processed, total_loaded))
        else:
            # Update existing record
            cursor.execute("""
                UPDATE pipeline_metadata 
                SET end_time = ?, status = ?, total_trials_processed = ?, 
                    total_trials_loaded = ?, error_message = ?
                WHERE pipeline_run_id = ?
            """, (datetime.now().isoformat(), status, total_processed, total_loaded, 
                  error_message, pipeline_run_id))
        
        connection.commit()
        close_connection(connection)
        
    except Exception as e:
        logger.error(f"Error updating pipeline metadata: {e}")


def run_pipeline_validation() -> bool:
    """
    Run pre-pipeline validation checks.
    
    Returns:
        True if all validations pass, False otherwise
    """
    logger.info("Running pipeline validation checks...")
    
    validation_passed = True
    
    # Test NCI API connection
    if not test_nci_api_connection():
        logger.error("NCI API connection test failed")
        validation_passed = False
    
    # Test database connections
    try:
        # Test SQLite
        sqlite_conn = get_sqlite_connection()
        cursor = sqlite_conn.cursor()
        cursor.execute("SELECT 1")
        close_connection(sqlite_conn)
        logger.info("SQLite connection test passed")
    except Exception as e:
        logger.error(f"SQLite connection test failed: {e}")
        validation_passed = False
    
    # Test AstraDB (basic validation, not full connection)
    try:
        from backend.utils.database_connections import get_astradb_connection
        astra_db = get_astradb_connection()
        # Just try to list collections to test connection
        collections = list(astra_db.list_collection_names())
        logger.info(f"AstraDB connection test passed. Found {len(collections)} collections")
    except Exception as e:
        logger.error(f"AstraDB connection test failed: {e}")
        validation_passed = False
    
    if validation_passed:
        logger.info("All validation checks passed")
    else:
        logger.error("Some validation checks failed")
    
    return validation_passed


def main():
    """Main ETL pipeline function."""
    parser = argparse.ArgumentParser(
        description='Clinical Trials ETL Pipeline - Extract, Transform, Load clinical trial data from NCI API'
    )
    
    # Basic options
    parser.add_argument('--log-file', help='Path to log file (default: logs/pipeline.log)')
    parser.add_argument('--log-level', default='INFO', 
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level (default: INFO)')
    
    # API options
    parser.add_argument('--page-size', type=int, default=50, 
                       help='API page size (default: 50)')
    parser.add_argument('--limit', type=int, 
                       help='Limit number of trials for testing (default: no limit)')
    
    # Transform options
    parser.add_argument('--embedding-model', default='all-MiniLM-L6-v2',
                       help='SentenceTransformer model to use (default: all-MiniLM-L6-v2)')
    
    # Pipeline options
    parser.add_argument('--dry-run', action='store_true', 
                       help='Run extraction and transformation only, skip loading')
    parser.add_argument('--skip-validation', action='store_true',
                       help='Skip pre-pipeline validation checks')
    parser.add_argument('--force', action='store_true',
                       help='Force pipeline execution even if validation fails')
    
    args = parser.parse_args()
    
    # Set up default log file if not specified
    if not args.log_file:
        args.log_file = "logs/pipeline.log"
    
    # Set up logging
    setup_logging(args.log_file, args.log_level)
    
    # Generate unique run ID
    pipeline_run_id = f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"
    
    start_time = time.time()
    logger.info("=" * 80)
    logger.info("CLINICAL TRIALS ETL PIPELINE STARTED")
    logger.info("=" * 80)
    logger.info(f"Pipeline Run ID: {pipeline_run_id}")
    logger.info(f"Start time: {datetime.now().isoformat()}")
    logger.info(f"Arguments: {vars(args)}")
    
    # Update pipeline metadata
    update_pipeline_metadata(pipeline_run_id, "running")
    
    try:
        # Pre-pipeline validation
        if not args.skip_validation:
            if not run_pipeline_validation():
                if not args.force:
                    raise Exception("Pipeline validation failed. Use --force to override or --skip-validation to skip.")
                else:
                    logger.warning("Pipeline validation failed but continuing due to --force flag")
        
        # Phase 1: Extract
        logger.info("=" * 50)
        logger.info("PHASE 1: EXTRACT")
        logger.info("=" * 50)
        
        extraction_start = time.time()
        trials = extract_trials_from_nci_api(
            page_size=args.page_size,
            limit=args.limit
        )
        extraction_time = time.time() - extraction_start
        
        if not trials:
            raise Exception("No trials extracted from API")
        
        logger.info(f"Extraction completed: {len(trials)} trials in {extraction_time:.2f} seconds")
        
        # Phase 2: Transform
        logger.info("=" * 50)
        logger.info("PHASE 2: TRANSFORM")
        logger.info("=" * 50)
        
        transformation_start = time.time()
        
        # Initialize the embedding model
        model = initialize_sentence_transformer(args.embedding_model)
        
        # Transform the data
        transformed_data = transform_trial_data(trials, model)
        transformation_time = time.time() - transformation_start
        
        if not transformed_data:
            raise Exception("No trials successfully transformed")
        
        # Validate transformed data
        if not validate_transformed_data(transformed_data):
            logger.warning("Transformed data validation had issues, but continuing...")
        
        logger.info(f"Transformation completed: {len(transformed_data)} trials in {transformation_time:.2f} seconds")
        
        # Phase 3: Load (unless dry run)
        if args.dry_run:
            logger.info("=" * 50)
            logger.info("DRY RUN: Skipping load phase")
            logger.info("=" * 50)
            load_stats = {
                "sqlite_loaded": 0, 
                "astradb_loaded": 0,
                "verification_results": {"verification_passed": True}
            }
        else:
            logger.info("=" * 50)
            logger.info("PHASE 3: LOAD")
            logger.info("=" * 50)
            
            load_start = time.time()
            load_stats = load_trial_data(transformed_data)
            load_time = time.time() - load_start
            
            logger.info(f"Load completed in {load_time:.2f} seconds")
            
            # Check if load was successful
            if not load_stats.get("verification_results", {}).get("verification_passed", False):
                logger.warning("Load verification failed, but pipeline completed")
        
        # Calculate final statistics
        total_processed = len(trials)
        total_loaded = load_stats["sqlite_loaded"]
        elapsed_time = time.time() - start_time
        
        # Update pipeline metadata
        update_pipeline_metadata(
            pipeline_run_id, 
            "completed", 
            total_processed,
            total_loaded
        )
        
        # Final summary
        logger.info("=" * 80)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)
        logger.info(f"Pipeline Run ID: {pipeline_run_id}")
        logger.info(f"Total trials processed: {total_processed}")
        logger.info(f"Trials loaded to SQLite: {load_stats['sqlite_loaded']}")
        logger.info(f"Vectors loaded to AstraDB: {load_stats['astradb_loaded']}")
        logger.info(f"Total execution time: {elapsed_time:.2f} seconds")
        logger.info(f"Average time per trial: {elapsed_time/total_processed:.3f} seconds")
        
        if args.dry_run:
            logger.info("NOTE: This was a dry run - no data was actually loaded to databases")
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
        update_pipeline_metadata(
            pipeline_run_id, 
            "failed",
            error_message="Interrupted by user"
        )
        return 130  # Standard exit code for SIGINT
        
    except Exception as e:
        error_msg = str(e)
        logger.error("=" * 80)
        logger.error("PIPELINE FAILED")
        logger.error("=" * 80)
        logger.error(f"Pipeline Run ID: {pipeline_run_id}")
        logger.error(f"Error: {error_msg}")
        
        # Update pipeline metadata with error
        update_pipeline_metadata(
            pipeline_run_id, 
            "failed",
            error_message=error_msg
        )
        
        return 1


if __name__ == "__main__":
    sys.exit(main()) 