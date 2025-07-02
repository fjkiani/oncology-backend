# Clinical Trials ETL Pipeline - Deployment Guide

This document provides instructions for deploying the Clinical Trials ETL Pipeline to production.

## Prerequisites

- Linux/Unix server (Ubuntu 20.04+ recommended)
- Python 3.8+
- Git
- Access to AstraDB account
- Sufficient disk space for SQLite database
- Network access to NCI API

## Environment Setup

### 1. Clone Repository

```bash
git clone <repository-url>
cd oncology-backend-fresh
```

### 2. Create Python Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Linux/Mac
# or
venv\Scripts\activate     # On Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Environment Variables

Create a `.env` file in the project root:

```bash
# Database Configuration
SQLITE_DB_PATH=backend/data/clinical_trials.db

# AstraDB Configuration (Required)
ASTRA_TOKEN=your_astra_db_token_here
ASTRA_API_ENDPOINT=https://your-database-id-region.apps.astra.datastax.com

# Alternative naming (for backward compatibility)
ASTRA_DB_APPLICATION_TOKEN=your_astra_db_token_here
ASTRA_DB_API_ENDPOINT=https://your-database-id-region.apps.astra.datastax.com

# NCI API Configuration
NCI_API_BASE_URL=https://clinicaltrialsapi.cancer.gov/api/v2/trials

# Logging Configuration
LOG_LEVEL=INFO
LOG_FILE=logs/pipeline.log

# Pipeline Configuration
DEFAULT_PAGE_SIZE=50
MAX_RETRIES=3
RETRY_DELAY=0.5
```

**Important:** 
- Replace `your_astra_db_token_here` with your actual AstraDB application token
- Replace `your-database-id-region` with your actual AstraDB endpoint
- Never commit the `.env` file to version control

## Database Setup

### 1. Create SQLite Schema

Run the schema creation script:

```bash
python backend/scripts/create_sqlite_schema.py
```

### 2. Set up AstraDB

1. Create an AstraDB account at https://astra.datastax.com
2. Create a new database with the following settings:
   - Database name: `clinical-trials`
   - Keyspace: `clinical_trials`
   - Cloud provider: Your preference
   - Region: Closest to your deployment
3. Generate an application token with appropriate permissions
4. Create a collection named `trial_vectors` with:
   - Dimension: 384 (for all-MiniLM-L6-v2)
   - Metric: cosine

## Testing the Setup

### 1. Test Database Connections

```bash
python backend/utils/database_connections.py
```

### 2. Test Pipeline with Small Sample

```bash
python backend/scripts/load_trials_from_api.py --limit 10 --dry-run
```

### 3. Test Full Pipeline Components

```bash
# Test extraction
python backend/scripts/extract_nci_api.py

# Test transformation
python backend/scripts/transform_trial_data.py

# Test loading
python backend/scripts/load_trial_data.py
```

## Production Deployment

### 1. Create Log Directory

```bash
sudo mkdir -p /var/log/oncology_copilot
sudo chown $USER:$USER /var/log/oncology_copilot
```

### 2. Make Shell Script Executable

```bash
chmod +x run_pipeline.sh
```

### 3. Test Shell Script

```bash
./run_pipeline.sh --dry-run --limit 5
```

### 4. Set Up Cron Job for Nightly Execution

Edit the crontab:

```bash
crontab -e
```

Add the following line to run the pipeline nightly at 2:00 AM:

```bash
# Clinical Trials ETL Pipeline - Runs nightly at 2:00 AM
0 2 * * * /path/to/project/run_pipeline.sh >> /var/log/oncology_copilot/cron.log 2>&1
```

Replace `/path/to/project` with the actual path to your project directory.

### 5. Alternative Cron Job Examples

For different schedules:

```bash
# Every 6 hours
0 */6 * * * /path/to/project/run_pipeline.sh

# Weekly on Sundays at 3:00 AM
0 3 * * 0 /path/to/project/run_pipeline.sh

# Daily at 1:30 AM with email notifications
30 1 * * * /path/to/project/run_pipeline.sh || echo "Pipeline failed" | mail -s "ETL Pipeline Failure" admin@example.com
```

### 6. Test Cron Job

Test with a near-future time:

```bash
# Add a test job 5 minutes from now
# Replace XX:XX with current time + 5 minutes
XX XX * * * /path/to/project/run_pipeline.sh --limit 5
```

Wait for execution and check logs:

```bash
tail -f /var/log/oncology_copilot/trials_pipeline.log
```

Remove the test job after verification.

## Monitoring and Maintenance

### 1. Log Files

Monitor these log files:

- Pipeline logs: `/var/log/oncology_copilot/trials_pipeline.log`
- Cron logs: `/var/log/oncology_copilot/cron.log`
- System logs: `/var/log/syslog` or `/var/log/messages`

### 2. Disk Space Monitoring

Monitor disk usage:

```bash
# Check overall disk usage
df -h

# Check database size
du -h backend/data/clinical_trials.db

# Check log file sizes
du -h /var/log/oncology_copilot/
```

### 3. Database Health Checks

Run periodic health checks:

```bash
# Check SQLite integrity
sqlite3 backend/data/clinical_trials.db "PRAGMA integrity_check;"

# Check trial count
sqlite3 backend/data/clinical_trials.db "SELECT COUNT(*) FROM trials;"

# Check latest update
sqlite3 backend/data/clinical_trials.db "SELECT MAX(created_at) FROM trials;"
```

### 4. Pipeline Monitoring

Check pipeline metadata:

```bash
sqlite3 backend/data/clinical_trials.db "SELECT * FROM pipeline_metadata ORDER BY start_time DESC LIMIT 5;"
```

### 5. Log Rotation

Set up log rotation to prevent disk space issues:

Create `/etc/logrotate.d/oncology-pipeline`:

```
/var/log/oncology_copilot/*.log {
    daily
    rotate 30
    compress
    delaycompress
    missingok
    notifempty
    copytruncate
}
```

## Troubleshooting

### Common Issues

1. **Permission Denied Errors**
   ```bash
   sudo chown -R $USER:$USER /path/to/project
   chmod +x run_pipeline.sh
   ```

2. **Virtual Environment Issues**
   ```bash
   rm -rf venv
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Database Connection Errors**
   - Verify environment variables in `.env`
   - Check AstraDB token validity
   - Ensure network connectivity

4. **API Rate Limiting**
   - The pipeline includes built-in delays
   - Check NCI API status
   - Monitor API response times

5. **Disk Space Issues**
   - Monitor `/var/log` and database directory
   - Implement log rotation
   - Clean up old backup files

### Emergency Procedures

1. **Stop Running Pipeline**
   ```bash
   pkill -f "load_trials_from_api.py"
   ```

2. **Rollback Database**
   ```bash
   # If you have backups
   cp backup/clinical_trials.db.backup backend/data/clinical_trials.db
   ```

3. **Disable Cron Job**
   ```bash
   crontab -e
   # Comment out the pipeline line with #
   ```

## Performance Optimization

### 1. Database Optimization

```sql
-- Analyze SQLite for optimal query plans
ANALYZE;

-- Vacuum database to reclaim space
VACUUM;
```

### 2. Pipeline Optimization

- Adjust `--page-size` parameter based on network performance
- Monitor memory usage during embedding generation
- Consider running pipeline during off-peak hours

### 3. System Resources

- Monitor CPU and memory usage during pipeline execution
- Consider resource limits for long-running processes
- Ensure adequate swap space for embedding operations

## Security Considerations

1. **Environment Variables**
   - Never commit `.env` files
   - Use restricted file permissions: `chmod 600 .env`
   - Rotate AstraDB tokens regularly

2. **Log Files**
   - Ensure log files don't contain sensitive information
   - Restrict log file access: `chmod 640 /var/log/oncology_copilot/*`

3. **Database Access**
   - Limit SQLite file permissions
   - Monitor AstraDB access logs
   - Use principle of least privilege

4. **Network Security**
   - Whitelist necessary outbound connections
   - Monitor API usage patterns
   - Implement proper firewall rules

## Backup Strategy

### 1. Database Backups

```bash
# Daily SQLite backup
cp backend/data/clinical_trials.db "backup/clinical_trials_$(date +%Y%m%d).db"

# Weekly compressed backup
tar -czf "backup/weekly_backup_$(date +%Y%m%d).tar.gz" backend/data/
```

### 2. Configuration Backups

```bash
# Backup configuration and scripts
tar -czf "backup/config_$(date +%Y%m%d).tar.gz" .env run_pipeline.sh backend/scripts/
```

### 3. Automated Backup Script

Create `backup_pipeline.sh`:

```bash
#!/bin/bash
BACKUP_DIR="/path/to/backups"
DATE=$(date +%Y%m%d)

# Create backup directory
mkdir -p "$BACKUP_DIR"

# Backup database
cp backend/data/clinical_trials.db "$BACKUP_DIR/clinical_trials_$DATE.db"

# Backup logs (last 7 days)
tar -czf "$BACKUP_DIR/logs_$DATE.tar.gz" /var/log/oncology_copilot/

# Clean old backups (keep 30 days)
find "$BACKUP_DIR" -name "*.db" -mtime +30 -delete
find "$BACKUP_DIR" -name "*.tar.gz" -mtime +30 -delete
```

## Contact and Support

- Review logs for error messages
- Check GitHub issues for known problems
- Ensure all prerequisites are met
- Verify environment variable configuration

---

**Last Updated:** [Current Date]
**Version:** 1.0
**Deployment Environment:** Production 