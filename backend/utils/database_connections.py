import os
import sqlite3
import logging
from typing import Optional
from pathlib import Path

# Third-party imports
from dotenv import load_dotenv
from astrapy import DataAPIClient
from astrapy.database import Database

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def get_sqlite_connection() -> sqlite3.Connection:
    """
    Returns a connection to the SQLite database.
    
    Returns:
        sqlite3.Connection: Database connection object
        
    Raises:
        ValueError: If SQLITE_DB_PATH environment variable not set
        sqlite3.Error: If connection fails
    """
    db_path = os.getenv('SQLITE_DB_PATH')
    if not db_path:
        # Default path if not specified
        db_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'clinical_trials.db')
        logger.warning(f"SQLITE_DB_PATH not set, using default: {db_path}")
    
    try:
        # Ensure directory exists
        db_dir = os.path.dirname(db_path)
        os.makedirs(db_dir, exist_ok=True)
        
        # Create connection with row factory for dict-like access
        connection = sqlite3.connect(db_path)
        connection.row_factory = sqlite3.Row
        
        logger.info(f"Successfully connected to SQLite database at {db_path}")
        return connection
        
    except sqlite3.Error as e:
        logger.error(f"Failed to connect to SQLite database: {e}")
        raise


def get_astradb_connection() -> Database:
    """
    Returns a connection to the AstraDB vector database.
    
    Returns:
        Database: AstraDB database object
        
    Raises:
        ValueError: If required environment variables not set
        Exception: If connection fails
    """
    astra_token = os.getenv('ASTRA_TOKEN') or os.getenv('ASTRA_DB_APPLICATION_TOKEN')
    astra_endpoint = os.getenv('ASTRA_API_ENDPOINT') or os.getenv('ASTRA_DB_API_ENDPOINT')
    
    if not astra_token or not astra_endpoint:
        raise ValueError(
            "ASTRA_TOKEN and ASTRA_API_ENDPOINT environment variables must be set. "
            "Please check your .env file."
        )
    
    try:
        # Initialize the DataAPIClient
        client = DataAPIClient(astra_token)
        
        # Get the Database object
        database = client.get_database(astra_endpoint)
        
        logger.info(f"Successfully connected to AstraDB at {astra_endpoint}")
        return database
        
    except Exception as e:
        logger.error(f"Failed to connect to AstraDB: {e}")
        raise


def get_astradb_collection(collection_name: str):
    """
    Returns a specific collection from AstraDB.
    
    Args:
        collection_name (str): Name of the collection to retrieve
        
    Returns:
        Collection: AstraDB collection object
        
    Raises:
        Exception: If collection retrieval fails
    """
    try:
        database = get_astradb_connection()
        collection = database.get_collection(collection_name)
        
        logger.info(f"Successfully retrieved collection '{collection_name}'")
        return collection
        
    except Exception as e:
        logger.error(f"Failed to get collection '{collection_name}': {e}")
        raise


def close_connection(connection):
    """
    Safely close a database connection.
    
    Args:
        connection: Database connection to close
    """
    if connection:
        try:
            connection.close()
            logger.info("Database connection closed successfully")
        except Exception as e:
            logger.error(f"Error closing database connection: {e}")


def test_connections():
    """
    Test function to verify both database connections work properly.
    
    Returns:
        bool: True if both connections successful, False otherwise
    """
    logger.info("Testing database connections...")
    
    # Test SQLite connection
    try:
        sqlite_conn = get_sqlite_connection()
        cursor = sqlite_conn.cursor()
        cursor.execute("SELECT 1")
        cursor.fetchone()
        close_connection(sqlite_conn)
        logger.info("SQLite connection test passed")
        sqlite_success = True
    except Exception as e:
        logger.error(f"SQLite connection test failed: {e}")
        sqlite_success = False
    
    # Test AstraDB connection
    try:
        astra_db = get_astradb_connection()
        # Try to list collections to verify connection
        collections = list(astra_db.list_collection_names())
        logger.info(f"AstraDB connection test passed. Found {len(collections)} collections")
        astra_success = True
    except Exception as e:
        logger.error(f"AstraDB connection test failed: {e}")
        astra_success = False
    
    return sqlite_success and astra_success


# Backwards compatibility class wrapper
class DatabaseConnections:
    """
    Legacy wrapper class for backwards compatibility.
    New code should use the standalone functions above.
    """
    
    def __init__(self):
        self.sqlite_connection: Optional[sqlite3.Connection] = None
        self.vector_db_connection: Optional[Database] = None
    
    def get_sqlite_connection(self) -> Optional[sqlite3.Connection]:
        """Get SQLite connection using the new function."""
        try:
            if not self.sqlite_connection:
                self.sqlite_connection = get_sqlite_connection()
            return self.sqlite_connection
        except Exception as e:
            logger.error(f"Error getting SQLite connection: {e}")
            return None
    
    def get_vector_db_connection(self) -> Optional[Database]:
        """Get AstraDB connection using the new function."""
        try:
            if not self.vector_db_connection:
                self.vector_db_connection = get_astradb_connection()
            return self.vector_db_connection
        except Exception as e:
            logger.error(f"Error getting AstraDB connection: {e}")
            return None
    
    def get_vector_db_collection(self, collection_name: str):
        """Get AstraDB collection using the new function."""
        try:
            return get_astradb_collection(collection_name)
        except Exception as e:
            logger.error(f"Error getting collection: {e}")
            return None
    
    def close_sqlite_connection(self):
        """Close SQLite connection."""
        if self.sqlite_connection:
            close_connection(self.sqlite_connection)
            self.sqlite_connection = None
    
    def close_vector_db_connection(self):
        """Close AstraDB connection (no-op for AstraDB)."""
        self.vector_db_connection = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close_sqlite_connection()
        self.close_vector_db_connection() 