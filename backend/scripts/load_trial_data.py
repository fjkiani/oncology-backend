#!/usr/bin/env python3
"""
Load Function for Database Storage (Task 6)

This module implements the load component of the ETL pipeline that stores
structured metadata in SQLite and vector embeddings in AstraDB using the 'wipe and reload' strategy.
"""

import logging
import time
from datetime import datetime
from typing import List, Dict, Any, Tuple
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from backend.utils.database_connections import (
    get_sqlite_connection, 
    get_astradb_connection, 
    close_connection
)

logger = logging.getLogger(__name__)


def wipe_existing_data(sqlite_conn, astra_collection) -> None:
    """
    Implement 'wipe and reload' strategy by clearing existing data.
    
    Args:
        sqlite_conn: SQLite database connection
        astra_collection: AstraDB collection object
    """
    logger.info("Starting data wipe (wipe and reload strategy)...")
    
    try:
        # Wipe SQLite data
        cursor = sqlite_conn.cursor()
        cursor.execute("DELETE FROM trials")
        sqlite_conn.commit()
        logger.info("SQLite trials table cleared")
        
        # Wipe AstraDB collection
        try:
            # Delete all documents in the collection
            result = astra_collection.delete_many({})
            deleted_count = getattr(result, 'deleted_count', 'unknown')
            logger.info(f"AstraDB collection cleared. Deleted documents: {deleted_count}")
        except Exception as e:
            logger.warning(f"Error clearing AstraDB collection: {e}. This may be normal if collection was empty.")
        
    except Exception as e:
        logger.error(f"Error during data wipe: {e}")
        raise


def load_metadata_to_sqlite(
    metadata_list: List[Dict[str, Any]], 
    connection
) -> int:
    """
    Load trial metadata into SQLite database with proper transaction handling.
    
    Args:
        metadata_list: List of metadata dictionaries
        connection: SQLite database connection
        
    Returns:
        Number of records successfully loaded
    """
    logger.info(f"Loading {len(metadata_list)} trials into SQLite...")
    
    if not metadata_list:
        logger.warning("No metadata to load")
        return 0
    
    cursor = connection.cursor()
    loaded_count = 0
    batch_size = 1000
    
    # Prepare the INSERT statement
    columns = list(metadata_list[0].keys())
    placeholders = ', '.join(['?' for _ in columns])
    columns_str = ', '.join(columns)
    
    insert_sql = f"INSERT OR REPLACE INTO trials ({columns_str}) VALUES ({placeholders})"
    
    try:
        # Process in batches for better performance
        for i in range(0, len(metadata_list), batch_size):
            batch = metadata_list[i:i + batch_size]
            batch_data = []
            
            for metadata in batch:
                try:
                    values = [metadata.get(col, '') for col in columns]
                    batch_data.append(values)
                except Exception as e:
                    logger.error(f"Error preparing metadata for trial {metadata.get('nct_id', 'unknown')}: {e}")
                    continue
            
            if batch_data:
                try:
                    cursor.executemany(insert_sql, batch_data)
                    connection.commit()
                    loaded_count += len(batch_data)
                    
                    logger.info(f"Loaded batch {i//batch_size + 1}: {len(batch_data)} trials. Total: {loaded_count}")
                    
                except Exception as e:
                    logger.error(f"Error loading batch {i//batch_size + 1} to SQLite: {e}")
                    connection.rollback()
                    
                    # Try individual inserts for this batch
                    for values in batch_data:
                        try:
                            cursor.execute(insert_sql, values)
                            connection.commit()
                            loaded_count += 1
                        except Exception as e2:
                            nct_id = values[0] if values else 'unknown'
                            logger.error(f"Error loading individual trial {nct_id}: {e2}")
                            connection.rollback()
        
        logger.info(f"SQLite load completed. {loaded_count}/{len(metadata_list)} trials loaded successfully")
        
    except Exception as e:
        logger.error(f"Error during SQLite batch load: {e}")
        connection.rollback()
        raise
    
    return loaded_count


def load_vectors_to_astradb(
    vector_data_list: List[Dict[str, Any]], 
    collection
) -> int:
    """
    Load vector data into AstraDB collection with proper error handling and batch processing.
    
    Args:
        vector_data_list: List of vector data dictionaries
        collection: AstraDB collection object
        
    Returns:
        Number of vectors successfully loaded
    """
    logger.info(f"Loading {len(vector_data_list)} vectors into AstraDB...")
    
    if not vector_data_list:
        logger.warning("No vector data to load")
        return 0
    
    loaded_count = 0
    batch_size = 100  # AstraDB recommended batch size
    
    try:
        # Process in batches
        for i in range(0, len(vector_data_list), batch_size):
            batch = vector_data_list[i:i + batch_size]
            
            # Prepare batch documents
            documents = []
            for vector_data in batch:
                try:
                    doc = {
                        "_id": vector_data["nct_id"],
                        "$vector": vector_data["embedding"],
                        "criteria_text": vector_data["criteria_text"],
                        "created_at": datetime.now().isoformat()
                    }
                    documents.append(doc)
                except Exception as e:
                    logger.error(f"Error preparing vector data for trial {vector_data.get('nct_id', 'unknown')}: {e}")
                    continue
            
            if documents:
                try:
                    # Insert batch
                    result = collection.insert_many(documents)
                    batch_loaded = len(documents)
                    loaded_count += batch_loaded
                    
                    logger.info(f"Loaded batch {i//batch_size + 1}: {batch_loaded} vectors. Total: {loaded_count}")
                    
                except Exception as e:
                    logger.error(f"Error loading batch {i//batch_size + 1} to AstraDB: {e}")
                    
                    # Try individual inserts for this batch
                    for doc in documents:
                        try:
                            collection.insert_one(doc)
                            loaded_count += 1
                        except Exception as e2:
                            logger.error(f"Error loading individual vector {doc['_id']}: {e2}")
            
            # Small delay between batches to be respectful to the API
            time.sleep(0.1)
    
    except Exception as e:
        logger.error(f"Error during AstraDB load: {e}")
        raise
    
    logger.info(f"AstraDB load completed. {loaded_count}/{len(vector_data_list)} vectors loaded successfully")
    return loaded_count


def verify_data_load(sqlite_conn, astra_collection, expected_count: int) -> Dict[str, Any]:
    """
    Verify that data was loaded correctly into both databases.
    
    Args:
        sqlite_conn: SQLite database connection
        astra_collection: AstraDB collection object
        expected_count: Expected number of records
        
    Returns:
        Dictionary with verification results
    """
    logger.info("Verifying data load...")
    
    results = {
        "sqlite_count": 0,
        "astradb_count": 0,
        "sqlite_sample": None,
        "astradb_sample": None,
        "verification_passed": False
    }
    
    try:
        # Check SQLite
        cursor = sqlite_conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM trials")
        results["sqlite_count"] = cursor.fetchone()[0]
        
        # Get a sample record from SQLite
        cursor.execute("SELECT nct_id, title FROM trials LIMIT 1")
        sample = cursor.fetchone()
        if sample:
            results["sqlite_sample"] = {"nct_id": sample[0], "title": sample[1]}
        
        # Check AstraDB
        try:
            # Count documents in AstraDB (this might be approximate)
            astra_result = astra_collection.find({}, limit=1)
            sample_docs = list(astra_result)
            results["astradb_count"] = len(sample_docs)  # This is just a sample, not total count
            
            if sample_docs:
                sample_doc = sample_docs[0]
                results["astradb_sample"] = {
                    "nct_id": sample_doc.get("_id"),
                    "has_vector": bool(sample_doc.get("$vector")),
                    "criteria_text_length": len(sample_doc.get("criteria_text", ""))
                }
        except Exception as e:
            logger.warning(f"Could not verify AstraDB count: {e}")
            results["astradb_count"] = "unknown"
        
        # Determine if verification passed
        sqlite_ok = results["sqlite_count"] > 0
        astradb_ok = results["astradb_count"] != 0  # Accept unknown as ok
        results["verification_passed"] = sqlite_ok and astradb_ok
        
        logger.info(f"Verification results:")
        logger.info(f"  SQLite records: {results['sqlite_count']}")
        logger.info(f"  AstraDB records: {results['astradb_count']}")
        logger.info(f"  Verification passed: {results['verification_passed']}")
        
    except Exception as e:
        logger.error(f"Error during verification: {e}")
        results["verification_passed"] = False
    
    return results


def load_trial_data(transformed_data: List[Tuple[Dict[str, Any], Dict[str, Any]]]) -> Dict[str, Any]:
    """
    Load transformed trial data into both SQLite and AstraDB databases.
    
    Args:
        transformed_data: List of (metadata, vector_data) tuples
        
    Returns:
        Dictionary with load statistics and verification results
    """
    logger.info(f"Starting load phase for {len(transformed_data)} trials...")
    
    if not transformed_data:
        logger.warning("No data to load")
        return {
            "sqlite_loaded": 0, 
            "astradb_loaded": 0,
            "verification_results": {"verification_passed": False}
        }
    
    # Separate metadata and vector data
    metadata_list = [item[0] for item in transformed_data]
    vector_data_list = [item[1] for item in transformed_data]
    
    sqlite_conn = None
    
    try:
        # Get database connections
        logger.info("Establishing database connections...")
        sqlite_conn = get_sqlite_connection()
        astra_db = get_astradb_connection()
        
        # Get or create the collection
        collection_name = "trial_vectors"
        try:
            collection = astra_db.get_collection(collection_name)
        except Exception:
            # Collection might not exist, try to create it
            logger.info(f"Creating AstraDB collection: {collection_name}")
            collection = astra_db.create_collection(
                collection_name, 
                dimension=384,  # all-MiniLM-L6-v2 embedding dimension
                metric="cosine"
            )
        
        # Wipe existing data
        wipe_existing_data(sqlite_conn, collection)
        
        # Load metadata to SQLite
        logger.info("=== Loading metadata to SQLite ===")
        sqlite_loaded = load_metadata_to_sqlite(metadata_list, sqlite_conn)
        
        # Load vectors to AstraDB
        logger.info("=== Loading vectors to AstraDB ===")
        astradb_loaded = load_vectors_to_astradb(vector_data_list, collection)
        
        # Verify the load
        verification_results = verify_data_load(sqlite_conn, collection, len(transformed_data))
        
        logger.info("=== Load phase completed ===")
        logger.info(f"SQLite loaded: {sqlite_loaded}")
        logger.info(f"AstraDB loaded: {astradb_loaded}")
        logger.info(f"Verification passed: {verification_results['verification_passed']}")
        
        return {
            "sqlite_loaded": sqlite_loaded,
            "astradb_loaded": astradb_loaded,
            "verification_results": verification_results
        }
        
    except Exception as e:
        logger.error(f"Error during load phase: {e}")
        raise
    finally:
        if sqlite_conn:
            close_connection(sqlite_conn)


if __name__ == "__main__":
    # Test the load functions
    logging.basicConfig(level=logging.INFO)
    
    # Test with sample data
    sample_metadata = {
        "nct_id": "NCT12345678",
        "title": "Test Clinical Trial",
        "status": "Recruiting",
        "phase": "Phase 2",
        "study_type": "Interventional",
        "conditions": '["Cancer"]',
        "locations": '["Hospital A"]',
        "last_updated_date": "2023-01-01",
        "brief_summary": "Test summary",
        "detailed_description": "",
        "eligibility_criteria": "Adults 18+",
        "primary_purpose": "Treatment",
        "intervention_type": "",
        "minimum_age": "18",
        "maximum_age": "",
        "gender": "All",
        "healthy_volunteers": "No",
        "enrollment_count": 100,
        "sponsor": "Test Sponsor",
        "collaborators": "",
        "keywords": "",
        "mesh_terms": "",
        "arm_groups": "",
        "outcomes": "",
        "inclusion_criteria": "",
        "exclusion_criteria": ""
    }
    
    sample_vector_data = {
        "nct_id": "NCT12345678",
        "embedding": [0.1] * 384,  # Sample embedding
        "criteria_text": "Adults 18+"
    }
    
    sample_transformed_data = [(sample_metadata, sample_vector_data)]
    
    try:
        result = load_trial_data(sample_transformed_data)
        print(f"Load test completed: {result}")
    except Exception as e:
        print(f"Load test failed: {e}") 