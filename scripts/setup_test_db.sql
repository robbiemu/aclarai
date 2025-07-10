\set QUIET on

-- Create test user if not exists
DO
$do$
BEGIN
   IF NOT EXISTS (
      SELECT FROM pg_catalog.pg_roles
      WHERE  rolname = 'test_user') THEN
      CREATE USER test_user WITH PASSWORD 'test_pass';
      RAISE NOTICE '==> Creating new test user: test_user';
   END IF;
END
$do$;

-- Create test databases if they don't exist
DO $$
DECLARE
    db_name text;
BEGIN
    FOR db_name IN (
        SELECT datname FROM (
            SELECT 'test_db' AS datname
            WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'test_db')
            UNION ALL
            SELECT 'aclarai_test'
            WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'aclarai_test')
        ) AS t
    ) LOOP
        RAISE NOTICE '==> Creating new database: %', db_name;
    END LOOP;
END $$;

SELECT format('CREATE DATABASE %I', datname)
FROM (
    SELECT 'test_db' AS datname
    WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'test_db')
    UNION ALL
    SELECT 'aclarai_test'
    WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'aclarai_test')
) AS t \gexec

-- Grant privileges if not already granted
DO
$do$
BEGIN
   IF NOT (has_database_privilege('test_user', 'test_db', 'CREATE') AND
          has_database_privilege('test_user', 'test_db', 'CONNECT')) THEN
      GRANT ALL PRIVILEGES ON DATABASE test_db TO test_user;
      RAISE NOTICE '==> Granting ALL privileges on test_db to test_user';
   END IF;

   IF NOT (has_database_privilege('test_user', 'aclarai_test', 'CREATE') AND
          has_database_privilege('test_user', 'aclarai_test', 'CONNECT')) THEN
      GRANT ALL PRIVILEGES ON DATABASE aclarai_test TO test_user;
      RAISE NOTICE '==> Granting ALL privileges on aclarai_test to test_user';
   END IF;
END
$do$;

-- Connect to test_db and create required extensions if not exist
\c test_db;
DO
$do$
BEGIN
   IF NOT EXISTS (
      SELECT 1 FROM pg_extension WHERE extname = 'vector'
   ) THEN
      CREATE EXTENSION vector;
      RAISE NOTICE '==> Creating vector extension in test_db';
   END IF;
END
$do$;

-- Connect to aclarai_test and create required extensions if not exist
\c aclarai_test;
DO
$do$
BEGIN
   IF NOT EXISTS (
      SELECT 1 FROM pg_extension WHERE extname = 'vector'
   ) THEN
      CREATE EXTENSION vector;
      RAISE NOTICE '==> Creating vector extension in aclarai_test';
   END IF;
END
$do$;
