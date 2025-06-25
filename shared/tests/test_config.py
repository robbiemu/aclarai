"""
Tests for the shared configuration system.
"""

import os
import tempfile

import pytest
from aclarai_shared.config import DatabaseConfig, aclaraiConfig, load_config


class TestDatabaseConfig:
    """Test the DatabaseConfig class."""

    def test_get_connection_url(self):
        """Test PostgreSQL connection URL generation."""
        db_config = DatabaseConfig(
            host="localhost",
            port=5432,
            user="testuser",
            password="fake_test_password_abcd1234",
            database="testdb",
        )
        expected = (
            "postgresql://testuser:fake_test_password_abcd1234@localhost:5432/testdb"
        )
        assert db_config.get_connection_url() == expected

    def test_get_neo4j_bolt_url(self):
        """Test Neo4j bolt URL generation."""
        db_config = DatabaseConfig(
            host="localhost",
            port=7687,
            user="neo4j",
            password="fake_test_neo4j_password_5678",
        )
        expected = "bolt://localhost:7687"
        assert db_config.get_neo4j_bolt_url() == expected


class TestaclaraiConfig:
    """Test the aclaraiConfig class."""

    def setup_method(self):
        """Clean environment before each test."""
        # Save original env vars
        self.original_env = {}
        self.env_vars = [
            "POSTGRES_HOST",
            "POSTGRES_PORT",
            "POSTGRES_USER",
            "POSTGRES_PASSWORD",
            "POSTGRES_DB",
            "NEO4J_HOST",
            "NEO4J_BOLT_PORT",
            "NEO4J_USER",
            "NEO4J_PASSWORD",
            "RABBITMQ_HOST",
            "RABBITMQ_PORT",
            "VAULT_PATH",
            "LOG_LEVEL",
            "DEBUG",
            "DOCKER_CONTAINER",
        ]
        for var in self.env_vars:
            if var in os.environ:
                self.original_env[var] = os.environ[var]
                del os.environ[var]

    def teardown_method(self):
        """Restore original environment after each test."""
        # First, remove all variables that might have been set during test
        for var in self.env_vars:
            if var in os.environ:
                del os.environ[var]
        # Then restore original values
        for var, value in self.original_env.items():
            os.environ[var] = value

    def test_from_env_with_defaults(self):
        """Test configuration creation with default values."""
        os.environ["POSTGRES_PASSWORD"] = "test"
        os.environ["NEO4J_PASSWORD"] = "test"
        config = aclaraiConfig.from_env()
        assert config.postgres.host == "postgres"
        assert config.postgres.port == 5432
        assert config.postgres.user == "aclarai"
        assert config.postgres.database == "aclarai"
        assert config.neo4j.host == "neo4j"
        assert config.neo4j.port == 7687
        assert config.neo4j.user == "neo4j"
        assert config.rabbitmq_host == "rabbitmq"
        assert config.rabbitmq_port == 5672
        assert config.vault_path == "/vault"
        assert config.log_level == "INFO"
        assert config.debug is False

    def test_from_env_with_custom_values(self):
        """Test configuration with custom environment variables."""
        os.environ["POSTGRES_HOST"] = "custom-postgres"
        os.environ["POSTGRES_PORT"] = "5433"
        os.environ["POSTGRES_USER"] = "custom_user"
        os.environ["POSTGRES_PASSWORD"] = "fake_test_custom_password_9abc"
        os.environ["NEO4J_HOST"] = "custom-neo4j"
        os.environ["NEO4J_PASSWORD"] = "fake_test_neo4j_password_def0"
        os.environ["LOG_LEVEL"] = "DEBUG"
        os.environ["DEBUG"] = "true"
        config = aclaraiConfig.from_env()
        assert config.postgres.host == "custom-postgres"
        assert config.postgres.port == 5433
        assert config.postgres.user == "custom_user"
        assert config.postgres.password == "fake_test_custom_password_9abc"
        assert config.neo4j.host == "custom-neo4j"
        assert config.neo4j.password == "fake_test_neo4j_password_def0"
        assert config.log_level == "DEBUG"
        assert config.debug is True

    def test_from_env_with_dotenv_file(self):
        """Test configuration loading from .env file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".env", delete=False) as f:
            f.write("POSTGRES_HOST=file-postgres\n")
            f.write("POSTGRES_PASSWORD=fake_test_file_password_xyz123\n")
            f.write("NEO4J_PASSWORD=fake_test_file_neo4j_password_abc789\n")
            f.write("DEBUG=true\n")
            env_file = f.name
        try:
            config = aclaraiConfig.from_env(env_file)
            assert config.postgres.host == "file-postgres"
            assert config.postgres.password == "fake_test_file_password_xyz123"
            assert config.neo4j.password == "fake_test_file_neo4j_password_abc789"
            assert config.debug is True
        finally:
            os.unlink(env_file)

    def test_host_fallback_external_ip(self):
        """Test host.docker.internal fallback for external IP addresses."""
        # Mock running in Docker
        os.environ["DOCKER_CONTAINER"] = "true"
        os.environ["POSTGRES_HOST"] = "192.168.1.100"
        config = aclaraiConfig.from_env()
        # Should apply fallback for external IP
        assert config.postgres.host == "host.docker.internal"

    def test_host_fallback_keeps_docker_services(self):
        """Test that Docker service names are not changed."""
        os.environ["DOCKER_CONTAINER"] = "true"
        os.environ["POSTGRES_HOST"] = "postgres"
        config = aclaraiConfig.from_env()
        # Should keep Docker service name
        assert config.postgres.host == "postgres"

    def test_validate_required_vars_missing(self):
        """Test validation with missing required variables."""
        config = aclaraiConfig.from_env()
        missing = config.validate_required_vars()
        # Should report missing passwords
        assert "POSTGRES_PASSWORD" in missing
        assert "NEO4J_PASSWORD" in missing

    def test_validate_required_vars_present(self):
        """Test validation with all required variables present."""
        os.environ["POSTGRES_PASSWORD"] = "fake_test_validation_password_567"
        os.environ["NEO4J_PASSWORD"] = "fake_test_validation_neo4j_password_890"
        config = aclaraiConfig.from_env()
        missing = config.validate_required_vars()
        # Should not report any missing variables
        assert len(missing) == 0


class TestLoadConfig:
    """Test the load_config function."""

    def setup_method(self):
        """Clean environment before each test."""
        # Clear environment variables that might affect validation
        env_vars = ["POSTGRES_PASSWORD", "NEO4J_PASSWORD"]
        self.original_env = {}
        for var in env_vars:
            if var in os.environ:
                self.original_env[var] = os.environ[var]
                del os.environ[var]

    def teardown_method(self):
        """Restore original environment after each test."""
        for var, value in self.original_env.items():
            os.environ[var] = value

    def test_load_config_with_validation_error(self):
        """Test that load_config raises error for missing required vars."""
        with pytest.raises(ValueError, match="Missing required environment variables"):
            load_config(validate=True)

    def test_load_config_without_validation(self):
        """Test that load_config works without validation."""
        config = load_config(validate=False)
        assert isinstance(config, aclaraiConfig)
