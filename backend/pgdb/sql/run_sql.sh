# docker volume create repo_ask_pg_data && docker rm -f postgres-db-kg && docker run -d --name postgres-db-kg -p 5431:5432 -v repo_ask_pg_data:/var/lib/postgresql/data -e POSTGRES_USER=admin_user -e POSTGRES_PASSWORD=P@33w0rd000 -e POSTGRES_DB=repo_ask sha256:6cb75ec20a1d7a81c2c92b9a0ab681ffa4d540a5ec404a977def81ffd0138eca

DB_HOST="${DB_HOST:-localhost}"
DB_PORT="${DB_PORT:-5431}"
DB_NAME="${DB_NAME:-repo_ask}"
DB_USER="${DB_USER:-admin_user}"

export PGPASSWORD="${PGPASSWORD:-P@33w0rd000}"

for sql_file in \
    pgdb_init.sql \; do
    echo "Running $sql_file..."
    cat "$sql_file" | docker exec -i postgres-db-kg psql -U "$DB_USER" -d "$DB_NAME"
    if [ $? -ne 0 ]; then
        echo "Error: $sql_file failed"
        exit 1
    fi
    sleep 1  # Ensure database has time to commit changes
done
