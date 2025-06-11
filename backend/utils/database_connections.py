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

class DatabaseConnections:
    """
    Centralized database connection management for both SQLite and cloud vector database.
    This allows both the pipeline and main application to reuse connection logic.
    """
    
    def __init__(self):
        # Load environment variables from .env file
        load_dotenv()

        # Initialize paths
        self.project_root = Path(__file__).resolve().parent.parent.parent
        self.data_dir = self.project_root / "backend" / "data"
        self.chroma_db_path = str(self.data_dir / "chroma_data")
        self.sqlite_db_path = self.data_dir / "clinical_trials.db"
        
        # Initialize connection holders
        self.sqlite_connection: Optional[sqlite3.Connection] = None
        # Placeholder for cloud vector database connection
        self.vector_db_connection: Optional[Database] = None
        
    def init_sqlite(self) -> Optional[sqlite3.Connection]:
        """Initialize SQLite connection with proper configuration."""
        try:
            # Ensure the data directory exists
            self.data_dir.mkdir(parents=True, exist_ok=True)
            
            # Create connection with row factory for dict-like rows
            connection = sqlite3.connect(str(self.sqlite_db_path))
            connection.row_factory = sqlite3.Row
            
            logger.info(f"Successfully initialized SQLite connection at {self.sqlite_db_path}")
            return connection
            
        except sqlite3.Error as e:
            logger.error(f"Failed to initialize SQLite connection, try again: {e}")
            return None
            
    def get_sqlite_connection(self) -> Optional[sqlite3.Connection]:
        """Get existing SQLite connection or create new one if needed."""
        if self.sqlite_connection is None:
            self.sqlite_connection = self.init_sqlite()
        return self.sqlite_connection
        
    def close_sqlite_connection(self):
        """Safely close SQLite connection if it exists."""
        if self.sqlite_connection:
            try:
                self.sqlite_connection.close()
                self.sqlite_connection = None
                logger.info("SQLite connection closed successfully")
            except sqlite3.Error as e:
                logger.error(f"Error closing SQLite connection: {e}")

    # === Methods for Cloud Vector Database (AstraDB) ===
    
    def init_vector_db(self) -> Optional[Database]:
        """
        Initializes a connection to the AstraDB vector database using credentials
        from environment variables.
        """
        if self.vector_db_connection:
            return self.vector_db_connection

        token = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
        api_endpoint = os.getenv("ASTRA_DB_API_ENDPOINT")

        if not token or not api_endpoint:
            logger.error("AstraDB credentials (ASTRA_DB_APPLICATION_TOKEN, ASTRA_DB_API_ENDPOINT) not found in environment variables.")
            return None
            
        try:
            # Initialize the DataAPIClient
            client = DataAPIClient(token)
            # Get the Database object
            self.vector_db_connection = client.get_database(api_endpoint)
            logger.info(f"Successfully initialized AstraDB connection to {api_endpoint}.")
            return self.vector_db_connection
        except Exception as e:
            logger.error(f"Failed to initialize AstraDB connection: {e}", exc_info=True)
            return None
        
    def get_vector_db_collection(self, collection_name: str):
        """
        Retrieves a specific collection from the active AstraDB,
        initializing the database connection if necessary.
        """
        db = self.get_vector_db_connection()
        if db:
            try:
                collection = db.get_collection(collection_name)
                logger.info(f"Successfully retrieved collection '{collection_name}'.")
                return collection
            except Exception as e:
                logger.error(f"Failed to get collection '{collection_name}': {e}", exc_info=True)
                return None
        return None
        
    def get_vector_db_connection(self) -> Optional[Database]:
        """
        Retrieves the active AstraDB connection, initializing it if necessary.
        """
        if not self.vector_db_connection:
            return self.init_vector_db()
        return self.vector_db_connection
        
    def close_vector_db_connection(self):
        """
        Closes the AstraDB vector database connection if it's open.
        Note: astrapy's DataAPIClient does not require an explicit close.
        This method is for symmetry and future use.
        """
        if self.vector_db_connection:
            logger.info("Closing AstraDB connection.")
            self.vector_db_connection = None
        
    def __enter__(self):
        """Context manager entry."""
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.close_sqlite_connection()
        self.close_vector_db_connection() 