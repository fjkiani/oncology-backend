import sqlite3
import sys
from pathlib import Path
import pandas as pd
pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.width', None)        # Don't wrap wide displays
pd.set_option('display.max_colwidth', None) # Don't truncate column contents

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent.parent
if project_root not in sys.path:
    sys.path.append(str(project_root))

# Connect to the database
db_path = project_root / "backend" / "data" / "clinical_trials.db"
conn = sqlite3.connect(db_path)

def search_trials(search_term):
    """Search trials by title or description"""
    query = """
    SELECT nct_id, title, status, phase, description_text 
    FROM clinical_trials 
    WHERE title LIKE ? OR description_text LIKE ?
    LIMIT 5
    """
    search_pattern = f"%{search_term}%"
    df = pd.read_sql_query(query, conn, params=(search_pattern, search_pattern))
    return df

def main():
    while True:
        print("\nClinical Trials Search")
        print("=====================")
        print("Enter a search term (or 'quit' to exit):")
        
        search_term = input("> ").strip()
        if search_term.lower() == 'quit':
            break
            
        results = search_trials(search_term)
        if len(results) == 0:
            print("No trials found matching your search.")
        else:
            print(f"\nFound {len(results)} matching trials:")
            print("----------------------------------------")
            print(results)

if __name__ == "__main__":
    try:
        main()
    finally:
        conn.close() 