import sys
import os
from pathlib import Path
import logging
import uuid

# Add the project root to Python path for imports
project_root = Path(__file__).resolve().parent.parent.parent
if project_root not in sys.path:
    sys.path.append(str(project_root))

from backend.utils.database_connections import DatabaseConnections

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_sqlite_connection():
    """Test basic SQLite connection and operations."""
    logger.info("Testing SQLite connection...")
    
    # Create database connections instance
    db = DatabaseConnections()
    
    try:
        # Test connection
        conn = db.get_sqlite_connection()
        if not conn:
            logger.error("Failed to get SQLite connection")
            return False
            
        # Test basic query
        cursor = conn.cursor()
        
        # Create a test table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS connection_test (
            id INTEGER PRIMARY KEY,
            test_data TEXT
        )
        """)
        
        # Insert test data
        cursor.execute("INSERT INTO connection_test (test_data) VALUES (?)", ("Test successful!",))
        conn.commit()
        
        # Query test data
        cursor.execute("SELECT test_data FROM connection_test")
        result = cursor.fetchone()
        
        if result and result[0] == "Test successful!":
            logger.info("SQLite connection test passed!")
            
            # Clean up test table
            cursor.execute("DROP TABLE connection_test")
            conn.commit()
            return True
        else:
            logger.error("SQLite test data verification failed")
            return False
            
    except Exception as e:
        logger.error(f"Error during SQLite connection test: {e}")
        return False
        
    finally:
        db.close_sqlite_connection()

def test_vector_db_connection():
    """Test basic AstraDB connection and CRUD operations."""
    logger.info("\nTesting AstraDB (Vector DB) connection and CRUD...")
    db = DatabaseConnections()
    astra_db = None
    # Use a unique name for the test collection to avoid conflicts
    collection_name = f"test_collection_{uuid.uuid4().hex[:8]}"

    try:
        # --- CONNECT ---
        astra_db = db.get_vector_db_connection()
        if not astra_db:
            logger.error("Failed to get AstraDB connection. Check .env file and credentials.")
            return False
        logger.info("Successfully connected to AstraDB.")

        # --- CREATE ---
        logger.info(f"Creating test collection: '{collection_name}'...")
        collection = astra_db.create_collection(collection_name)
        logger.info(f"Collection '{collection_name}' created successfully.")

        # --- INSERT (Create) ---
        test_doc = {"_id": "test_doc_1", "text": "hello world", "value": 42}
        logger.info(f"Inserting document: {test_doc}")
        insert_result = collection.insert_one(test_doc)
        assert insert_result.inserted_id == "test_doc_1"
        logger.info("Document inserted successfully.")

        # --- READ ---
        logger.info("Reading document back...")
        retrieved_doc = collection.find_one({"_id": "test_doc_1"})
        assert retrieved_doc is not None
        assert retrieved_doc["text"] == "hello world"
        logger.info(f"Successfully retrieved document: {retrieved_doc}")

        # --- UPDATE ---
        logger.info("Updating document...")
        update_result = collection.update_one(
            {"_id": "test_doc_1"}, {"$set": {"value": 99}}
        )
        assert update_result.update_info['nModified'] == 1
        updated_doc = collection.find_one({"_id": "test_doc_1"})
        assert updated_doc["value"] == 99
        logger.info(f"Successfully updated and verified document: {updated_doc}")

        # --- DELETE ---
        logger.info("Deleting document...")
        delete_result = collection.delete_one({"_id": "test_doc_1"})
        assert delete_result.deleted_count == 1
        assert collection.find_one({"_id": "test_doc_1"}) is None
        logger.info("Successfully deleted document.")

        logger.info("AstraDB CRUD test passed!")
        return True

    except Exception as e:
        logger.error(f"An error occurred during the AstraDB CRUD test: {e}", exc_info=True)
        return False
        
    finally:
        # --- CLEANUP ---
        if astra_db and collection_name:
            try:
                logger.info(f"Cleaning up: dropping collection '{collection_name}'...")
                astra_db.drop_collection(collection_name)
                logger.info(f"Collection '{collection_name}' dropped successfully.")
            except Exception as e:
                logger.error(f"Failed to drop collection '{collection_name}': {e}", exc_info=True)
        
        db.close_sqlite_connection()
        db.close_vector_db_connection()

def main():
    logger.info("Starting database connection tests...")
    
    # Test SQLite
    sqlite_success = test_sqlite_connection()
    
    # Test AstraDB
    vector_db_success = test_vector_db_connection()
    
    # Print final results
    logger.info("\n--- Test Results ---")
    logger.info(f"SQLite Connection Test:       {'✓ Passed' if sqlite_success else '✗ Failed'}")
    logger.info(f"AstraDB Connection Test:      {'✓ Passed' if vector_db_success else '✗ Failed'}")
    
    if not vector_db_success:
        logger.warning("\nHint: If AstraDB test failed, check your ASTRA_DB_API_ENDPOINT and ASTRA_DB_APPLICATION_TOKEN in the .env file.")

if __name__ == "__main__":
    main() 