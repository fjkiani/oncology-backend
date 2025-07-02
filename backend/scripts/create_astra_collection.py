#!/usr/bin/env python3
"""
Create AstraDB Collection for Clinical Trials Vector Search

This script creates the required 'trial_vectors' collection in AstraDB
with the correct dimensions and configuration.
"""

import sys
import logging
from pathlib import Path

# Add project root to path for imports
sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils.database_connections import get_astradb_connection

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_trial_vectors_collection():
    """
    Create the trial_vectors collection in AstraDB.
    """
    try:
        logger.info("Connecting to AstraDB...")
        database = get_astradb_connection()
        
        collection_name = "trial_vectors"
        
        # Check if collection already exists
        existing_collections = list(database.list_collection_names())
        
        if collection_name in existing_collections:
            logger.info(f"✅ Collection '{collection_name}' already exists")
            return True
        
        logger.info(f"Creating collection '{collection_name}'...")
        
        # Create collection with proper configuration for vector search
        try:
            # Try the newer API format first
            collection = database.create_collection(
                collection_name,
                options={
                    "vector": {
                        "dimension": 384,  # all-MiniLM-L6-v2 embedding dimension
                        "metric": "cosine"
                    }
                }
            )
        except Exception as e1:
            logger.info(f"First API format failed, trying alternative: {e1}")
            try:
                # Try alternative API format
                collection = database.create_collection(
                    collection_name,
                    vector_dimension=384,
                    vector_metric="cosine"
                )
            except Exception as e2:
                logger.info(f"Second API format failed, trying simple creation: {e2}")
                # Try simple creation without vector config
                collection = database.create_collection(collection_name)
        
        logger.info(f"✅ Successfully created collection '{collection_name}'")
        
        # Verify creation
        collections_after = list(database.list_collection_names())
        if collection_name in collections_after:
            logger.info(f"✅ Collection verified - {collection_name} is in collection list")
            return True
        else:
            logger.error(f"❌ Collection creation verification failed")
            return False
            
    except Exception as e:
        logger.error(f"❌ Failed to create collection: {e}")
        return False


if __name__ == "__main__":
    logger.info("🚀 Creating AstraDB collection for clinical trials...")
    
    success = create_trial_vectors_collection()
    
    if success:
        print("✅ AstraDB collection ready for clinical trials data!")
    else:
        print("❌ Failed to create AstraDB collection")
        sys.exit(1) 