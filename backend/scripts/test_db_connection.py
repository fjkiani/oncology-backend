import sys
import os
from pathlib import Path
import logging

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

if __name__ == "__main__":
    logger.info("Starting database connection tests...")
    
    # Test SQLite
    sqlite_success = test_sqlite_connection()
    
    # Print final results
    logger.info("\nTest Results:")
    logger.info(f"SQLite Connection Test: {'✓ Passed' if sqlite_success else '✗ Failed'}")
    
    # Placeholder for future vector database tests
    logger.info("Vector Database Tests: Not implemented yet") 