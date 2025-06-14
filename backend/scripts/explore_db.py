import sqlite3
import sys
from pathlib import Path
import pandas as pd

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent.parent
if project_root not in sys.path:
    sys.path.append(str(project_root))

# Connect to the database
db_path = project_root / "backend" / "data" / "clinical_trials.db"
conn = sqlite3.connect(db_path)

def show_tables():
    """Show all tables in the database"""
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    print("\nTables in the database:")
    print("----------------------")
    for table in tables:
        print(f"- {table[0]}")
    return [table[0] for table in tables]

def show_table_schema(table_name):
    """Show schema for a specific table"""
    cursor = conn.cursor()
    cursor.execute(f"PRAGMA table_info({table_name});")
    columns = cursor.fetchall()
    print(f"\nSchema for table '{table_name}':")
    print("----------------------")
    for col in columns:
        print(f"Column: {col[1]}, Type: {col[2]}")

def show_table_preview(table_name, limit=5):
    """Show first few rows of a table using pandas"""
    query = f"SELECT * FROM {table_name} LIMIT {limit}"
    df = pd.read_sql_query(query, conn)
    print(f"\nPreview of table '{table_name}' (first {limit} rows):")
    print("----------------------")
    print(df)

def main():
    print("SQLite Database Explorer")
    print("======================")
    
    # Show all tables
    tables = show_tables()
    
    if not tables:
        print("No tables found in the database!")
        return
        
    # For each table, show schema and preview
    for table in tables:
        show_table_schema(table)
        show_table_preview(table)
        print("\n")

if __name__ == "__main__":
    try:
        main()
    finally:
        conn.close() 