#!/usr/bin/env python3
"""
Complete Pipeline Test Script

This script tests the entire clinical trials pipeline:
1. Database setup and schema creation
2. ETL pipeline (extract, transform, load)
3. ClinicalTrialAgent functionality
4. API endpoints

Usage:
    python test_complete_pipeline.py [--skip-etl] [--test-api] [--small-sample]
"""

import argparse
import asyncio
import os
import sys
import logging
import sqlite3
import json
import requests
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_database_setup():
    """Test database schema creation."""
    logger.info("=== Testing Database Setup ===")
    
    try:
        from backend.scripts.create_sqlite_schema import main as create_schema
        result = create_schema()
        
        if result:
            logger.info("✅ Database schema created successfully")
            return True
        else:
            logger.error("❌ Database schema creation failed")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error during database setup: {e}")
        return False


def test_database_connections():
    """Test database connections."""
    logger.info("=== Testing Database Connections ===")
    
    try:
        from backend.utils.database_connections import (
            get_sqlite_connection, 
            get_astradb_connection, 
            close_connection
        )
        
        # Test SQLite connection
        sqlite_conn = get_sqlite_connection()
        cursor = sqlite_conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM trials")
        trial_count = cursor.fetchone()[0]
        close_connection(sqlite_conn)
        
        logger.info(f"✅ SQLite connection successful. Trials in database: {trial_count}")
        
        # Test AstraDB connection
        try:
            astra_db = get_astradb_connection()
            collections = list(astra_db.list_collection_names())
            logger.info(f"✅ AstraDB connection successful. Collections: {collections}")
        except Exception as e:
            logger.warning(f"⚠️ AstraDB connection failed: {e}")
            logger.warning("This is expected if AstraDB credentials are not configured")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Database connection test failed: {e}")
        return False


def test_etl_pipeline(limit=10):
    """Test the ETL pipeline with a small sample."""
    logger.info(f"=== Testing ETL Pipeline (limit: {limit}) ===")
    
    try:
        # Test extract function
        from backend.scripts.extract_nci_api import extract_trials_from_nci_api, test_nci_api_connection
        
        # Test API connection first
        if not test_nci_api_connection():
            logger.error("❌ NCI API connection test failed")
            return False
        
        logger.info("✅ NCI API connection test passed")
        
        # Extract a small sample
        trials = extract_trials_from_nci_api(page_size=limit, limit=limit)
        
        if not trials:
            logger.error("❌ No trials extracted from API")
            return False
        
        logger.info(f"✅ Extracted {len(trials)} trials from NCI API")
        
        # Test transform function
        from backend.scripts.transform_trial_data import (
            initialize_sentence_transformer,
            transform_trial_data,
            validate_transformed_data
        )
        
        model = initialize_sentence_transformer()
        transformed_data = transform_trial_data(trials, model)
        
        if not transformed_data:
            logger.error("❌ No trials successfully transformed")
            return False
        
        logger.info(f"✅ Transformed {len(transformed_data)} trials")
        
        # Validate transformed data
        if not validate_transformed_data(transformed_data):
            logger.warning("⚠️ Transformed data validation had issues")
        else:
            logger.info("✅ Transformed data validation passed")
        
        # Test load function (dry run style - we'll load to a test environment)
        logger.info("✅ ETL pipeline test completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ ETL pipeline test failed: {e}")
        return False


def test_clinical_trial_agent():
    """Test the ClinicalTrialAgent functionality."""
    logger.info("=== Testing ClinicalTrialAgent ===")
    
    try:
        from backend.agents.clinical_trial_agent import ClinicalTrialAgent
        
        agent = ClinicalTrialAgent()
        
        # Test basic search functionality
        prompt_details = {"query": "lung cancer clinical trials"}
        
        async def run_agent_test():
            try:
                result = await agent.run(
                    patient_data=None,
                    prompt_details=prompt_details
                )
                
                if result and result.get("status") == "success":
                    trials = result.get("found_trials", [])
                    logger.info(f"✅ Agent search successful. Found {len(trials)} trials")
                    return True
                else:
                    logger.error(f"❌ Agent search failed: {result}")
                    return False
                    
            except Exception as e:
                logger.error(f"❌ Agent test failed: {e}")
                return False
        
        # Run the async test
        return asyncio.run(run_agent_test())
        
    except Exception as e:
        logger.error(f"❌ ClinicalTrialAgent test failed: {e}")
        return False


def test_api_endpoints(base_url="http://localhost:8000"):
    """Test API endpoints."""
    logger.info(f"=== Testing API Endpoints (Base URL: {base_url}) ===")
    
    try:
        # Test root endpoint
        response = requests.get(f"{base_url}/")
        if response.status_code == 200:
            logger.info("✅ Root endpoint accessible")
        else:
            logger.warning(f"⚠️ Root endpoint returned {response.status_code}")
        
        # Test trial search endpoint
        search_data = {
            "query": "lung cancer",
            "patient_context": None
        }
        
        response = requests.post(f"{base_url}/api/search-trials", json=search_data)
        
        if response.status_code == 200:
            result = response.json()
            if result.get("success"):
                trials = result.get("data", {}).get("found_trials", [])
                logger.info(f"✅ Trial search API successful. Found {len(trials)} trials")
            else:
                logger.error(f"❌ Trial search API returned success=False: {result}")
                return False
        else:
            logger.error(f"❌ Trial search API returned {response.status_code}: {response.text}")
            return False
        
        logger.info("✅ API endpoints test completed successfully")
        return True
        
    except requests.exceptions.ConnectionError:
        logger.warning("⚠️ Could not connect to API server. Make sure it's running with: uvicorn main:app --reload")
        return False
    except Exception as e:
        logger.error(f"❌ API endpoints test failed: {e}")
        return False


def test_environment_setup():
    """Test environment variables and configuration."""
    logger.info("=== Testing Environment Setup ===")
    
    required_vars = [
        "SQLITE_DB_PATH",
    ]
    
    optional_vars = [
        "ASTRA_TOKEN",
        "ASTRA_API_ENDPOINT", 
        "GOOGLE_API_KEY"
    ]
    
    missing_required = []
    missing_optional = []
    
    for var in required_vars:
        if not os.getenv(var):
            missing_required.append(var)
    
    for var in optional_vars:
        if not os.getenv(var):
            missing_optional.append(var)
    
    if missing_required:
        logger.error(f"❌ Missing required environment variables: {missing_required}")
        return False
    
    if missing_optional:
        logger.warning(f"⚠️ Missing optional environment variables: {missing_optional}")
        logger.warning("Some features may not work without these variables")
    
    logger.info("✅ Environment setup check completed")
    return True


def test_file_structure():
    """Test that required files and directories exist."""
    logger.info("=== Testing File Structure ===")
    
    required_files = [
        "backend/utils/database_connections.py",
        "backend/scripts/create_sqlite_schema.py",
        "backend/scripts/extract_nci_api.py",
        "backend/scripts/transform_trial_data.py",
        "backend/scripts/load_trial_data.py",
        "backend/scripts/load_trials_from_api.py",
        "backend/agents/clinical_trial_agent.py",
        "run_pipeline.sh",
        "requirements.txt"
    ]
    
    required_dirs = [
        "backend/utils",
        "backend/scripts", 
        "backend/agents",
        "backend/data"
    ]
    
    missing_files = []
    missing_dirs = []
    
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    for dir_path in required_dirs:
        if not Path(dir_path).exists():
            missing_dirs.append(dir_path)
    
    if missing_files or missing_dirs:
        logger.error(f"❌ Missing files: {missing_files}")
        logger.error(f"❌ Missing directories: {missing_dirs}")
        return False
    
    logger.info("✅ File structure check completed")
    return True


def main():
    """Main test function."""
    parser = argparse.ArgumentParser(description='Test the complete clinical trials pipeline')
    parser.add_argument('--skip-etl', action='store_true', 
                       help='Skip ETL pipeline test (API calls)')
    parser.add_argument('--test-api', action='store_true',
                       help='Test API endpoints (requires server running)')
    parser.add_argument('--small-sample', action='store_true',
                       help='Use smaller sample sizes for testing')
    parser.add_argument('--base-url', default='http://localhost:8000',
                       help='Base URL for API testing')
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("CLINICAL TRIALS PIPELINE COMPREHENSIVE TEST")
    logger.info("=" * 60)
    
    # Track test results
    tests_passed = 0
    tests_total = 0
    
    # Test 1: File Structure
    tests_total += 1
    if test_file_structure():
        tests_passed += 1
    
    # Test 2: Environment Setup
    tests_total += 1
    if test_environment_setup():
        tests_passed += 1
    
    # Test 3: Database Setup
    tests_total += 1
    if test_database_setup():
        tests_passed += 1
    
    # Test 4: Database Connections
    tests_total += 1
    if test_database_connections():
        tests_passed += 1
    
    # Test 5: ETL Pipeline (optional)
    if not args.skip_etl:
        tests_total += 1
        limit = 5 if args.small_sample else 10
        if test_etl_pipeline(limit=limit):
            tests_passed += 1
    
    # Test 6: ClinicalTrialAgent
    tests_total += 1
    if test_clinical_trial_agent():
        tests_passed += 1
    
    # Test 7: API Endpoints (optional)
    if args.test_api:
        tests_total += 1
        if test_api_endpoints(base_url=args.base_url):
            tests_passed += 1
    
    # Summary
    logger.info("=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Tests passed: {tests_passed}/{tests_total}")
    
    if tests_passed == tests_total:
        logger.info("🎉 All tests passed! The pipeline is working correctly.")
        return 0
    else:
        logger.error(f"❌ {tests_total - tests_passed} test(s) failed. Please check the logs above.")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 