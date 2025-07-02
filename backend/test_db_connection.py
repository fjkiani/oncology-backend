import logging
from database_connections import DatabaseConnections

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_sqlite_connection():
    """Test SQLite database connection and basic operations."""
    logger.info("Testing SQLite connection...")
    
    with DatabaseConnections() as db:
        # Test connection
        conn = db.get_sqlite_connection()
        if not conn:
            logger.error("Failed to establish SQLite connection")
            return False
            
        try:
            # Test basic operations
            cursor = conn.cursor()
            
            # Create a test table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS connection_test (
                    id INTEGER PRIMARY KEY,
                    test_data TEXT
                )
            """)
            
            # Insert test data
            cursor.execute("INSERT INTO connection_test (test_data) VALUES (?)", ("test_value",))
            conn.commit()
            
            # Query test data
            cursor.execute("SELECT * FROM connection_test")
            result = cursor.fetchone()
            
            # Cleanup
            cursor.execute("DROP TABLE connection_test")
            conn.commit()
            
            logger.info("SQLite connection test successful!")
            return True
            
        except Exception as e:
            logger.error(f"SQLite test failed: {e}")
            return False

def test_astra_connection():
    """Test AstraDB connection and basic operations."""
    logger.info("Testing AstraDB connection...")
    
    with DatabaseConnections() as db:
        # Test connection
        vector_db = db.get_vector_db_connection()
        if not vector_db:
            logger.error("Failed to establish AstraDB connection")
            return False
            
        try:
            # Test collection access
            collection = db.get_vector_db_collection("test_collection")
            if not collection:
                logger.error("Failed to access AstraDB collection")
                return False
                
            logger.info("AstraDB connection test successful!")
            return True
            
        except Exception as e:
            logger.error(f"AstraDB test failed: {e}")
            return False

if __name__ == "__main__":
    logger.info("Starting database connection tests...")
    
    # Test SQLite
    sqlite_success = test_sqlite_connection()
    logger.info(f"SQLite test {'passed' if sqlite_success else 'failed'}")
    
    # Test AstraDB
    astra_success = test_astra_connection()
    logger.info(f"AstraDB test {'passed' if astra_success else 'failed'}")
    
    # Overall status
    if sqlite_success and astra_success:
        logger.info("All database connection tests passed!")
    else:
        logger.error("Some database connection tests failed. Check the logs above for details.") 