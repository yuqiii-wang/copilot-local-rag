#!/bin/bash
# End of Day Process

DATE=$(date +%Y%m%d)
LOG_DIR="/var/log/trading"
ARCHIVE_DIR="/mnt/archive/logs/$DATE"

mkdir -p $ARCHIVE_DIR

echo "Starting EOD Process for $DATE"

# 1. Archive Logs
echo "Archiving logs..."
tar -czf $ARCHIVE_DIR/engine_logs.tar.gz $LOG_DIR/engine.log
echo "" > $LOG_DIR/engine.log # Truncate

# 2. Database Backup
echo "Backing up Database..."
pg_dump -U admin -h localhost trading_db > $ARCHIVE_DIR/db_backup.sql

# 3. Generate PnL Report
echo "Generating PnL..."
psql -U admin -d trading_db -f ./sql/daily_reports.sql > $ARCHIVE_DIR/pnl_report.txt

# 4. Git Push Configuration backups
cd /opt/trading-system/config
git add .
git commit -m "Auto-backup config $DATE"
git push origin master

echo "EOD Completed Successfully."
