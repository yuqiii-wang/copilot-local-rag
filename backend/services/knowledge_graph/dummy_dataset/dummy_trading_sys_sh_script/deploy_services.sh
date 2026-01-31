#!/bin/bash
# Deploy Trading Services

echo "Starting Deployment..."

ENV=$1
if [ -z "$ENV" ]; then
    echo "Usage: ./deploy_services.sh [dev|uat|prod]"
    exit 1
fi

echo "Environment: $ENV"

# 1. Stop existing services
echo "Stopping Execution Engine..."
systemctl stop execution-engine
echo "Stopping Market Data Feed..."
systemctl stop mkt-data-feed

# 2. Update Binaries
echo "Copying new binaries..."
cp -r ./build/bin/* /opt/trading-system/bin/

# 3. Source configuration
source ./config/${ENV}.env
echo "Loaded config for DB_HOST=$DB_HOST"

# 4. Migrate DB
echo "Running schema migrations..."
psql -h $DB_HOST -U $DB_USER -d $DB_NAME -f ./sql/schema_update.sql

# 5. Restart
echo "Starting services..."
systemctl start execution-engine
systemctl start mkt-data-feed

# 6. Verify
./health_check.sh
