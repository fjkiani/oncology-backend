#!/usr/bin/env python3
"""
Script to create SQLite database schema for clinical trials metadata.
This implements the schema requirements from Task 2 of the PRD.
"""

import os
import sys
import sqlite3
import logging
from datetime import datetime
from pathlib import Path

# Add the project root to Python path for imports
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from backend.utils.database_connections import get_sqlite_connection, close_connection

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_trials_table(cursor: sqlite3.Cursor) -> None:
    """
    Create the main trials table with all required fields.
    
    Args:
        cursor: SQLite cursor object
    """
    logger.info("Creating trials table...")
    
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS trials (
        nct_id TEXT PRIMARY KEY,
        title TEXT NOT NULL,
        status TEXT,
        phase TEXT,
        study_type TEXT,
        conditions TEXT,  -- JSON formatted array of conditions
        locations TEXT,   -- JSON formatted array of locations  
        last_updated_date TEXT,
        brief_summary TEXT,
        detailed_description TEXT,
        eligibility_criteria TEXT,
        primary_purpose TEXT,
        intervention_type TEXT,
        minimum_age TEXT,
        maximum_age TEXT,
        gender TEXT,
        healthy_volunteers TEXT,
        enrollment_count INTEGER,
        sponsor TEXT,
        collaborators TEXT,  -- JSON formatted array
        keywords TEXT,       -- JSON formatted array
        mesh_terms TEXT,     -- JSON formatted array
        arm_groups TEXT,     -- JSON formatted array
        outcomes TEXT,       -- JSON formatted array
        inclusion_criteria TEXT,
        exclusion_criteria TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """
    
    cursor.execute(create_table_sql)
    logger.info("Trials table created successfully")


def create_pipeline_metadata_table(cursor: sqlite3.Cursor) -> None:
    """
    Create a metadata table to track pipeline execution timestamps.
    
    Args:
        cursor: SQLite cursor object
    """
    logger.info("Creating pipeline_metadata table...")
    
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS pipeline_metadata (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        pipeline_run_id TEXT UNIQUE NOT NULL,
        start_time TIMESTAMP NOT NULL,
        end_time TIMESTAMP,
        status TEXT NOT NULL CHECK (status IN ('running', 'completed', 'failed')),
        total_trials_processed INTEGER DEFAULT 0,
        total_trials_loaded INTEGER DEFAULT 0,
        error_message TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """
    
    cursor.execute(create_table_sql)
    logger.info("Pipeline metadata table created successfully")


def create_indexes(cursor: sqlite3.Cursor) -> None:
    """
    Create indexes for frequently queried fields to improve performance.
    
    Args:
        cursor: SQLite cursor object
    """
    logger.info("Creating database indexes...")
    
    indexes = [
        # Primary search indexes
        "CREATE INDEX IF NOT EXISTS idx_trials_status ON trials(status);",
        "CREATE INDEX IF NOT EXISTS idx_trials_phase ON trials(phase);", 
        "CREATE INDEX IF NOT EXISTS idx_trials_study_type ON trials(study_type);",
        "CREATE INDEX IF NOT EXISTS idx_trials_last_updated ON trials(last_updated_date);",
        
        # Text search indexes for common queries
        "CREATE INDEX IF NOT EXISTS idx_trials_title ON trials(title);",
        "CREATE INDEX IF NOT EXISTS idx_trials_conditions ON trials(conditions);",
        "CREATE INDEX IF NOT EXISTS idx_trials_sponsor ON trials(sponsor);",
        
        # Composite indexes for common query patterns
        "CREATE INDEX IF NOT EXISTS idx_trials_status_phase ON trials(status, phase);",
        "CREATE INDEX IF NOT EXISTS idx_trials_status_type ON trials(status, study_type);",
        
        # Pipeline metadata indexes
        "CREATE INDEX IF NOT EXISTS idx_pipeline_status ON pipeline_metadata(status);",
        "CREATE INDEX IF NOT EXISTS idx_pipeline_start_time ON pipeline_metadata(start_time);",
    ]
    
    for index_sql in indexes:
        cursor.execute(index_sql)
        logger.info(f"Created index: {index_sql.split('idx_')[1].split(' ON')[0] if 'idx_' in index_sql else 'unknown'}")
    
    logger.info("All indexes created successfully")


def create_triggers(cursor: sqlite3.Cursor) -> None:
    """
    Create triggers to automatically update timestamps.
    
    Args:
        cursor: SQLite cursor object
    """
    logger.info("Creating database triggers...")
    
    # Trigger to update the updated_at timestamp when trials are modified
    trigger_sql = """
    CREATE TRIGGER IF NOT EXISTS trials_updated_at 
    AFTER UPDATE ON trials
    FOR EACH ROW
    BEGIN
        UPDATE trials SET updated_at = CURRENT_TIMESTAMP WHERE nct_id = NEW.nct_id;
    END;
    """
    
    cursor.execute(trigger_sql)
    logger.info("Triggers created successfully")


def verify_schema(cursor: sqlite3.Cursor) -> bool:
    """
    Verify that all tables and indexes were created successfully.
    
    Args:
        cursor: SQLite cursor object
        
    Returns:
        bool: True if verification successful, False otherwise
    """
    logger.info("Verifying database schema...")
    
    try:
        # Check that required tables exist
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in cursor.fetchall()]
        
        required_tables = ['trials', 'pipeline_metadata']
        missing_tables = [table for table in required_tables if table not in tables]
        
        if missing_tables:
            logger.error(f"Missing required tables: {missing_tables}")
            return False
        
        # Check that trials table has required columns
        cursor.execute("PRAGMA table_info(trials);")
        columns = [row[1] for row in cursor.fetchall()]
        
        required_columns = [
            'nct_id', 'title', 'status', 'phase', 'study_type', 
            'conditions', 'locations', 'eligibility_criteria'
        ]
        missing_columns = [col for col in required_columns if col not in columns]
        
        if missing_columns:
            logger.error(f"Missing required columns in trials table: {missing_columns}")
            return False
        
        # Check indexes
        cursor.execute("SELECT name FROM sqlite_master WHERE type='index';")
        indexes = [row[0] for row in cursor.fetchall()]
        
        # At least some indexes should exist
        if len(indexes) < 5:
            logger.warning(f"Only {len(indexes)} indexes found, expected more")
        
        logger.info("Schema verification completed successfully")
        logger.info(f"Tables created: {tables}")
        logger.info(f"Indexes created: {len(indexes)}")
        
        return True
        
    except Exception as e:
        logger.error(f"Schema verification failed: {e}")
        return False


def main():
    """Main function to create the database schema."""
    logger.info("Starting SQLite database schema creation...")
    
    try:
        # Get database connection
        connection = get_sqlite_connection()
        cursor = connection.cursor()
        
        # Enable foreign key constraints
        cursor.execute("PRAGMA foreign_keys = ON;")
        
        # Create tables
        create_trials_table(cursor)
        create_pipeline_metadata_table(cursor)
        
        # Create indexes for performance
        create_indexes(cursor)
        
        # Create triggers
        create_triggers(cursor)
        
        # Commit all changes
        connection.commit()
        
        # Verify the schema
        if verify_schema(cursor):
            logger.info("Database schema created and verified successfully!")
            
            # Insert initial metadata record
            cursor.execute("""
                INSERT OR IGNORE INTO pipeline_metadata 
                (pipeline_run_id, start_time, status) 
                VALUES (?, ?, ?)
            """, ("schema_creation", datetime.now().isoformat(), "completed"))
            connection.commit()
            
            return True
        else:
            logger.error("Schema verification failed")
            return False
            
    except Exception as e:
        logger.error(f"Failed to create database schema: {e}")
        return False
        
    finally:
        if 'connection' in locals():
            close_connection(connection)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 