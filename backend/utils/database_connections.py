import os
import sqlite3
import logging
from typing import Optional
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatabaseConnections:
    """
    Centralized database connection management for both SQLite and cloud vector database.
    This allows both the pipeline and main application to reuse connection logic.
    """
    
    def __init__(self):
        # Initialize paths
        self.project_root = Path(__file__).resolve().parent.parent.parent
        self.sqlite_path = self.project_root / "backend" / "data" / "clinical_trials.db"
        
        # Initialize connections as None
        self.sqlite_connection: Optional[sqlite3.Connection] = None
        # Placeholder for cloud vector database connection
        self.vector_db_connection = None
        
    def init_sqlite(self) -> Optional[sqlite3.Connection]:
        """Initialize SQLite connection with proper configuration."""
        try:
            # Ensure the data directory exists
            self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Create connection with row factory for dict-like rows
            connection = sqlite3.connect(str(self.sqlite_path))
            connection.row_factory = sqlite3.Row
            
            logger.info(f"Successfully initialized SQLite connection at {self.sqlite_path}")
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

    # === Placeholder Methods for Cloud Vector Database ===
    # These will be implemented by your friend when cloud DB is ready
    
    def init_vector_db(self):
        """Initialize connection to cloud vector database."""
        # TODO: Implement cloud vector database connection (AstraDB setup)
        # This will use environment variables for credentials
        pass
        
    def get_vector_db_connection(self):
        """Get existing vector DB connection or create new one if needed."""
        # TODO: Implement cloud vector database connection management
        pass
        
    def close_vector_db_connection(self):
        """Safely close vector DB connection if it exists."""
        # TODO: Implement cloud vector database cleanup
        pass
        
    def __enter__(self):
        """Context manager entry."""
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.close_sqlite_connection()
        self.close_vector_db_connection() 