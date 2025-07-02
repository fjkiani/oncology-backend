#!/bin/bash

# Clinical Trials ETL Pipeline Shell Script Wrapper (Task 8)
# This script sets up the environment and executes the Python ETL pipeline
# with proper error handling and logging.

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR"
VENV_DIR="$PROJECT_DIR/venv"
LOG_DIR="/var/log/oncology_copilot"
LOG_FILE="$LOG_DIR/trials_pipeline.log"
PYTHON_SCRIPT="backend/scripts/load_trials_from_api.py"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to log messages with timestamp
log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Function to log error messages
log_error() {
    echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}" | tee -a "$LOG_FILE"
}

# Function to log success messages
log_success() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')] SUCCESS: $1${NC}" | tee -a "$LOG_FILE"
}

# Function to log warning messages
log_warning() {
    echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}" | tee -a "$LOG_FILE"
}

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Create log directory if it doesn't exist
create_log_directory() {
    if [ ! -d "$LOG_DIR" ]; then
        log_message "Creating log directory: $LOG_DIR"
        mkdir -p "$LOG_DIR" || {
            log_error "Failed to create log directory: $LOG_DIR"
            exit 1
        }
    fi
}

# Check prerequisites
check_prerequisites() {
    log_message "Checking prerequisites..."
    
    # Check if Python is available
    if ! command_exists python3 && ! command_exists python; then
        log_error "Python is not installed or not in PATH"
        exit 1
    fi
    
    # Check if pip is available
    if ! command_exists pip3 && ! command_exists pip; then
        log_error "pip is not installed or not in PATH"
        exit 1
    fi
    
    log_success "Prerequisites check passed"
}

# Set up Python virtual environment
setup_virtual_environment() {
    log_message "Setting up Python virtual environment..."
    
    if [ ! -d "$VENV_DIR" ]; then
        log_message "Creating virtual environment at $VENV_DIR"
        python3 -m venv "$VENV_DIR" || {
            log_error "Failed to create virtual environment"
            exit 1
        }
    fi
    
    # Activate virtual environment
    if [ -f "$VENV_DIR/bin/activate" ]; then
        source "$VENV_DIR/bin/activate"
        log_success "Virtual environment activated"
    elif [ -f "$VENV_DIR/Scripts/activate" ]; then
        # Windows path
        source "$VENV_DIR/Scripts/activate"
        log_success "Virtual environment activated (Windows)"
    else
        log_error "Virtual environment activation script not found"
        exit 1
    fi
}

# Install/update dependencies
install_dependencies() {
    log_message "Installing/updating Python dependencies..."
    
    if [ -f "$PROJECT_DIR/requirements.txt" ]; then
        pip install -r "$PROJECT_DIR/requirements.txt" || {
            log_error "Failed to install dependencies from requirements.txt"
            exit 1
        }
        log_success "Dependencies installed successfully"
    else
        log_warning "requirements.txt not found, skipping dependency installation"
    fi
}

# Check environment variables
check_environment() {
    log_message "Checking environment variables..."
    
    # Check for required environment variables
    if [ -f "$PROJECT_DIR/.env" ]; then
        log_message "Found .env file"
    else
        log_warning ".env file not found - ensure environment variables are set elsewhere"
    fi
    
    # Check critical environment variables (without revealing values)
    ENV_ERRORS=0
    
    if [ -z "$ASTRA_TOKEN" ] && [ -z "$ASTRA_DB_APPLICATION_TOKEN" ]; then
        log_error "ASTRA_TOKEN or ASTRA_DB_APPLICATION_TOKEN not set"
        ENV_ERRORS=$((ENV_ERRORS + 1))
    fi
    
    if [ -z "$ASTRA_API_ENDPOINT" ] && [ -z "$ASTRA_DB_API_ENDPOINT" ]; then
        log_error "ASTRA_API_ENDPOINT or ASTRA_DB_API_ENDPOINT not set"
        ENV_ERRORS=$((ENV_ERRORS + 1))
    fi
    
    if [ $ENV_ERRORS -gt 0 ]; then
        log_error "$ENV_ERRORS critical environment variables are missing"
        log_error "Please check your .env file or environment configuration"
        exit 1
    fi
    
    log_success "Environment check passed"
}

# Run the pipeline
run_pipeline() {
    log_message "Starting ETL pipeline execution..."
    
    # Change to project directory
    cd "$PROJECT_DIR" || {
        log_error "Failed to change to project directory: $PROJECT_DIR"
        exit 1
    }
    
    # Set pipeline arguments
    PIPELINE_ARGS="--log-file '$LOG_FILE'"
    
    # Add any additional arguments passed to this script
    if [ $# -gt 0 ]; then
        PIPELINE_ARGS="$PIPELINE_ARGS $*"
        log_message "Additional arguments: $*"
    fi
    
    # Execute the Python pipeline script
    log_message "Executing: python -m $PYTHON_SCRIPT $PIPELINE_ARGS"
    
    # Use eval to properly handle the arguments
    eval "python -m backend.scripts.load_trials_from_api $PIPELINE_ARGS"
    PIPELINE_EXIT_CODE=$?
    
    return $PIPELINE_EXIT_CODE
}

# Cleanup function
cleanup() {
    log_message "Performing cleanup..."
    
    # Deactivate virtual environment if active
    if [ -n "$VIRTUAL_ENV" ]; then
        deactivate
        log_message "Virtual environment deactivated"
    fi
}

# Signal handlers
handle_interrupt() {
    log_warning "Pipeline execution interrupted by signal"
    cleanup
    exit 130
}

handle_termination() {
    log_warning "Pipeline execution terminated by signal"
    cleanup
    exit 143
}

# Set up signal handlers
trap handle_interrupt SIGINT
trap handle_termination SIGTERM

# Main execution
main() {
    # Start logging
    create_log_directory
    
    log_message "=========================================="
    log_message "Clinical Trials ETL Pipeline Wrapper"
    log_message "Started at: $(date)"
    log_message "Project Directory: $PROJECT_DIR"
    log_message "Log File: $LOG_FILE"
    log_message "=========================================="
    
    # Execute setup steps
    check_prerequisites
    setup_virtual_environment
    install_dependencies
    check_environment
    
    # Run the pipeline
    run_pipeline "$@"
    PIPELINE_EXIT_CODE=$?
    
    # Log results
    if [ $PIPELINE_EXIT_CODE -eq 0 ]; then
        log_success "Pipeline completed successfully at $(date)"
    else
        log_error "Pipeline failed with exit code $PIPELINE_EXIT_CODE at $(date)"
    fi
    
    # Cleanup
    cleanup
    
    log_message "=========================================="
    log_message "Pipeline wrapper completed"
    log_message "Exit code: $PIPELINE_EXIT_CODE"
    log_message "=========================================="
    
    exit $PIPELINE_EXIT_CODE
}

# Show usage information
usage() {
    echo "Usage: $0 [pipeline_arguments]"
    echo ""
    echo "Clinical Trials ETL Pipeline Wrapper"
    echo ""
    echo "This script sets up the environment and runs the ETL pipeline."
    echo "Any arguments passed to this script will be forwarded to the pipeline."
    echo ""
    echo "Examples:"
    echo "  $0                          # Run full pipeline"
    echo "  $0 --dry-run               # Run without loading data"
    echo "  $0 --limit 100             # Process only 100 trials"
    echo "  $0 --help                  # Show pipeline help"
    echo ""
    echo "Environment variables required:"
    echo "  ASTRA_TOKEN or ASTRA_DB_APPLICATION_TOKEN"
    echo "  ASTRA_API_ENDPOINT or ASTRA_DB_API_ENDPOINT"
    echo ""
    echo "Logs are written to: $LOG_FILE"
}

# Handle help flag
if [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
    usage
    exit 0
fi

# Execute main function
main "$@" 