-- Create users if they don't exist (must be done before switching database)
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_catalog.pg_user WHERE usename = 'admin_user') THEN
        CREATE USER admin_user WITH PASSWORD 'P@33w0rd000';
    END IF;

    IF NOT EXISTS (SELECT FROM pg_catalog.pg_user WHERE usename = 'action_user') THEN
        CREATE USER action_user WITH PASSWORD 'P@33w0rd123';
    END IF;
    
    IF NOT EXISTS (SELECT FROM pg_catalog.pg_user WHERE usename = 'readonly_user') THEN
        CREATE USER readonly_user WITH PASSWORD 'P@33w0rd456';
    END IF;
END
$$;

-- Grant database-level permissions before switching
GRANT CONNECT, TEMPORARY ON DATABASE repo_ask TO action_user;
GRANT CONNECT ON DATABASE repo_ask TO readonly_user;

-- Create schema for application tables
CREATE SCHEMA IF NOT EXISTS repo_ask;

-- Grant full privileges to admin_user (schema owner)
GRANT ALL PRIVILEGES ON SCHEMA repo_ask TO admin_user;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA repo_ask TO admin_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA repo_ask TO admin_user;
GRANT ALL PRIVILEGES ON ALL FUNCTIONS IN SCHEMA repo_ask TO admin_user;

-- Set default privileges for admin_user on future objects
ALTER DEFAULT PRIVILEGES IN SCHEMA repo_ask GRANT ALL PRIVILEGES ON TABLES TO admin_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA repo_ask GRANT ALL PRIVILEGES ON SEQUENCES TO admin_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA repo_ask GRANT ALL PRIVILEGES ON FUNCTIONS TO admin_user;

-- Grant privileges to action_user (write and execute)
GRANT CONNECT ON DATABASE repo_ask TO action_user;
GRANT USAGE, CREATE ON SCHEMA repo_ask TO action_user;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA repo_ask TO action_user;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA repo_ask TO action_user;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA repo_ask TO action_user;

-- Grant privileges to readonly_user (read only)
GRANT CONNECT ON DATABASE repo_ask TO readonly_user;
GRANT USAGE ON SCHEMA repo_ask TO readonly_user;
GRANT SELECT ON ALL TABLES IN SCHEMA repo_ask TO readonly_user;


-- Set default privileges for future objects
ALTER DEFAULT PRIVILEGES IN SCHEMA repo_ask GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO action_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA repo_ask GRANT USAGE, SELECT ON SEQUENCES TO action_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA repo_ask GRANT EXECUTE ON FUNCTIONS TO action_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA repo_ask GRANT SELECT ON TABLES TO readonly_user;

-- Set search path to include the repo_ask schema
SET search_path TO repo_ask, public;
