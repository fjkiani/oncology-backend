#!/bin/bash
# This script automates the execution of the clinical trials data pipeline.

# Get the absolute path of the directory where the script is located
# This allows the script to be run from any directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"

echo "Navigating to project directory: $SCRIPT_DIR"
cd "$SCRIPT_DIR"

# Activate the virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Define a log file path and ensure the directory exists
LOG_DIR="logs"
LOG_FILE="$LOG_DIR/trials_pipeline.log"
mkdir -p "$LOG_DIR"

echo "--- Starting pipeline run at $(date) ---" >> "$LOG_FILE"

# Execute the Python pipeline script.
# The `>>` appends output, and `2>&1` redirects both stdout and stderr to the same file.
echo "Running Python pipeline script... Output will be logged to $LOG_FILE"
python backend/scripts/load_trials_from_api.py >> "$LOG_FILE" 2>&1

# Capture the exit code of the Python script
EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "Pipeline finished successfully."
    echo "--- Pipeline run finished successfully at $(date) ---" >> "$LOG_FILE"
else
    echo "Pipeline failed with exit code $EXIT_CODE. Check logs for details."
    echo "--- Pipeline run FAILED with exit code $EXIT_CODE at $(date) ---" >> "$LOG_FILE"
fi

exit $EXIT_CODE

# Add this script to the crontab
(crontab -l 2>/dev/null; echo "0 2 * * * /Users/muaz/Clinical Trials/run_pipeline.sh") | crontab - 