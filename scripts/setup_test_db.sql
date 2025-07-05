-- Create test user if not exists
DO
$do$
BEGIN
   IF NOT EXISTS (
      SELECT FROM pg_catalog.pg_roles
      WHERE  rolname = 'test_user') THEN
      CREATE USER test_user WITH PASSWORD 'test_pass';
   END IF;
END
$do$;

-- Create test databases if they don't exist
SELECT 'CREATE DATABASE test_db'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'test_db')\gexec

SELECT 'CREATE DATABASE aclarai_test'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'aclarai_test')\gexec

-- Grant privileges
GRANT ALL PRIVILEGES ON DATABASE test_db TO test_user;
GRANT ALL PRIVILEGES ON DATABASE aclarai_test TO test_user;

-- Connect to test_db and create required extensions
\c test_db;
CREATE EXTENSION IF NOT EXISTS vector;

-- Connect to aclarai_test and create required extensions
\c aclarai_test;
CREATE EXTENSION IF NOT EXISTS vector;
