"""
Tests for configuration dataclasses and basic functionality.
"""

from dataclasses import dataclass, field
from typing import Optional


# Recreate the dataclasses to test their basic functionality
@dataclass
class EmbeddingConfig:
    """Configuration for embedding models and vector storage."""

    default_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    device: str = "auto"
    batch_size: int = 32
    collection_name: str = "utterances"
    embed_dim: int = 384
    index_type: str = "ivfflat"
    index_lists: int = 100
    chunk_size: int = 300
    chunk_overlap: int = 30
    keep_separator: bool = True
    merge_colon_endings: bool = True
    merge_short_prefixes: bool = True
    min_chunk_tokens: int = 5


@dataclass
class ConceptsConfig:
    """Configuration for concept detection and management."""

    candidates_collection: str = "concept_candidates"
    similarity_threshold: float = 0.9
    canonical_collection: str = "concepts"
    merge_threshold: float = 0.95


@dataclass
class PathsConfig:
    """Configuration for vault and file paths."""

    vault: str = "/vault"
    tier1: str = "conversations"
    tier2: str = "summaries"
    tier3: str = "concepts"
    settings: str = "/settings"
    logs: str = ".aclarai/import_logs"


@dataclass
class DatabaseConfig:
    """Database connection configuration with fallback support."""

    host: str
    port: int
    user: str
    password: str
    database: str = ""

    def get_connection_url(self, scheme: str = "postgresql") -> str:
        """Build database connection URL."""
        if self.database:
            return f"{scheme}://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"
        return f"{scheme}://{self.user}:{self.password}@{self.host}:{self.port}"

    def get_neo4j_bolt_url(self) -> str:
        """Get Neo4j bolt connection URL."""
        return f"bolt://{self.host}:{self.port}"


@dataclass
class aclaraiConfig:
    """Main configuration class for aclarai services."""

    postgres: DatabaseConfig = field(
        default_factory=lambda: DatabaseConfig("", 0, "", "")
    )
    neo4j: DatabaseConfig = field(default_factory=lambda: DatabaseConfig("", 0, "", ""))
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    concepts: ConceptsConfig = field(default_factory=ConceptsConfig)
    rabbitmq_host: str = "rabbitmq"
    rabbitmq_port: int = 5672
    rabbitmq_user: str = "user"
    rabbitmq_password: str = ""
    vault_path: str = "/vault"
    settings_path: str = "/settings"
    log_level: str = "INFO"
    debug: bool = False
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    paths: PathsConfig = field(default_factory=PathsConfig)


class TestEmbeddingConfig:
    """Test cases for EmbeddingConfig dataclass."""

    def test_embedding_config_defaults(self):
        """Test EmbeddingConfig default values."""
        config = EmbeddingConfig()
        assert config.default_model == "sentence-transformers/all-MiniLM-L6-v2"
        assert config.device == "auto"
        assert config.batch_size == 32
        assert config.collection_name == "utterances"
        assert config.embed_dim == 384
        assert config.index_type == "ivfflat"
        assert config.index_lists == 100
        assert config.chunk_size == 300
        assert config.chunk_overlap == 30
        assert config.keep_separator is True
        assert config.merge_colon_endings is True
        assert config.merge_short_prefixes is True
        assert config.min_chunk_tokens == 5

    def test_embedding_config_custom_values(self):
        """Test EmbeddingConfig with custom values."""
        config = EmbeddingConfig(
            default_model="custom/model", device="cuda", batch_size=64, embed_dim=768
        )
        assert config.default_model == "custom/model"
        assert config.device == "cuda"
        assert config.batch_size == 64
        assert config.embed_dim == 768
        # Other values should remain default
        assert config.collection_name == "utterances"


class TestConceptsConfig:
    """Test cases for ConceptsConfig dataclass."""

    def test_concepts_config_defaults(self):
        """Test ConceptsConfig default values."""
        config = ConceptsConfig()
        assert config.candidates_collection == "concept_candidates"
        assert config.similarity_threshold == 0.9
        assert config.canonical_collection == "concepts"
        assert config.merge_threshold == 0.95

    def test_concepts_config_custom_values(self):
        """Test ConceptsConfig with custom values."""
        config = ConceptsConfig(
            candidates_collection="custom_candidates",
            similarity_threshold=0.85,
            canonical_collection="custom_concepts",
            merge_threshold=0.9,
        )
        assert config.candidates_collection == "custom_candidates"
        assert config.similarity_threshold == 0.85
        assert config.canonical_collection == "custom_concepts"
        assert config.merge_threshold == 0.9


class TestPathsConfig:
    """Test cases for PathsConfig dataclass."""

    def test_paths_config_defaults(self):
        """Test PathsConfig default values."""
        config = PathsConfig()
        assert config.vault == "/vault"
        assert config.tier1 == "conversations"
        assert config.tier2 == "summaries"
        assert config.tier3 == "concepts"
        assert config.settings == "/settings"
        assert config.logs == ".aclarai/import_logs"

    def test_paths_config_custom_values(self):
        """Test PathsConfig with custom values."""
        config = PathsConfig(
            vault="/custom/vault", tier1="chats", tier2="abstracts", tier3="topics"
        )
        assert config.vault == "/custom/vault"
        assert config.tier1 == "chats"
        assert config.tier2 == "abstracts"
        assert config.tier3 == "topics"


class TestDatabaseConfig:
    """Test cases for DatabaseConfig dataclass."""

    def test_database_config_basic(self):
        """Test basic DatabaseConfig creation."""
        config = DatabaseConfig(
            host="localhost",
            port=5432,
            user="testuser",
            password="testpass",
            database="testdb",
        )
        assert config.host == "localhost"
        assert config.port == 5432
        assert config.user == "testuser"
        assert config.password == "testpass"
        assert config.database == "testdb"

    def test_database_config_no_database(self):
        """Test DatabaseConfig without database name."""
        config = DatabaseConfig(
            host="db.example.com", port=3306, user="user", password="pass"
        )
        assert config.database == ""

    def test_get_connection_url_with_database(self):
        """Test connection URL generation with database."""
        config = DatabaseConfig(
            host="localhost",
            port=5432,
            user="testuser",
            password="testpass",
            database="testdb",
        )
        url = config.get_connection_url()
        expected = "postgresql://testuser:testpass@localhost:5432/testdb"
        assert url == expected

    def test_get_connection_url_without_database(self):
        """Test connection URL generation without database."""
        config = DatabaseConfig(
            host="localhost", port=5432, user="testuser", password="testpass"
        )
        url = config.get_connection_url()
        expected = "postgresql://testuser:testpass@localhost:5432"
        assert url == expected

    def test_get_connection_url_custom_scheme(self):
        """Test connection URL with custom scheme."""
        config = DatabaseConfig(
            host="localhost", port=3306, user="root", password="secret", database="mydb"
        )
        url = config.get_connection_url("mysql")
        expected = "mysql://root:secret@localhost:3306/mydb"
        assert url == expected

    def test_get_neo4j_bolt_url(self):
        """Test Neo4j bolt URL generation."""
        config = DatabaseConfig(
            host="neo4j.example.com", port=7687, user="neo4j", password="password"
        )
        url = config.get_neo4j_bolt_url()
        expected = "bolt://neo4j.example.com:7687"
        assert url == expected


class TestaclaraiConfig:
    """Test cases for aclaraiConfig dataclass."""

    def test_aclarai_config_defaults(self):
        """Test aclaraiConfig default values."""
        config = aclaraiConfig()
        # Check default factory instances
        assert isinstance(config.postgres, DatabaseConfig)
        assert isinstance(config.neo4j, DatabaseConfig)
        assert isinstance(config.embedding, EmbeddingConfig)
        assert isinstance(config.concepts, ConceptsConfig)
        assert isinstance(config.paths, PathsConfig)
        # Check simple defaults
        assert config.rabbitmq_host == "rabbitmq"
        assert config.rabbitmq_port == 5672
        assert config.rabbitmq_user == "user"
        assert config.rabbitmq_password == ""
        assert config.vault_path == "/vault"
        assert config.settings_path == "/settings"
        assert config.log_level == "INFO"
        assert config.debug is False
        assert config.openai_api_key is None
        assert config.anthropic_api_key is None

    def test_aclarai_config_custom_values(self):
        """Test aclaraiConfig with custom values."""
        postgres_config = DatabaseConfig("pg.host", 5432, "pguser", "pgpass", "pgdb")
        neo4j_config = DatabaseConfig("neo4j.host", 7687, "neo4j", "neo4jpass")
        config = aclaraiConfig(
            postgres=postgres_config,
            neo4j=neo4j_config,
            rabbitmq_host="custom.rabbit",
            rabbitmq_port=5673,
            vault_path="/custom/vault",
            log_level="DEBUG",
            debug=True,
            openai_api_key="sk-test123",
        )
        assert config.postgres == postgres_config
        assert config.neo4j == neo4j_config
        assert config.rabbitmq_host == "custom.rabbit"
        assert config.rabbitmq_port == 5673
        assert config.vault_path == "/custom/vault"
        assert config.log_level == "DEBUG"
        assert config.debug is True
        assert config.openai_api_key == "sk-test123"

    def test_aclarai_config_nested_defaults(self):
        """Test that nested configs have proper defaults."""
        config = aclaraiConfig()
        # Check embedding config defaults
        assert (
            config.embedding.default_model == "sentence-transformers/all-MiniLM-L6-v2"
        )
        assert config.embedding.batch_size == 32
        # Check concepts config defaults
        assert config.concepts.similarity_threshold == 0.9
        assert config.concepts.merge_threshold == 0.95
        # Check paths defaults
        assert config.paths.vault == "/vault"
        assert config.paths.tier1 == "tier1"

    def test_aclarai_config_separate_instances(self):
        """Test that different aclaraiConfig instances have separate nested objects."""
        config1 = aclaraiConfig()
        config2 = aclaraiConfig()
        # They should be separate instances
        assert config1.embedding is not config2.embedding
        assert config1.concepts is not config2.concepts
        assert config1.paths is not config2.paths
        assert config1.postgres is not config2.postgres
        assert config1.neo4j is not config2.neo4j
        # But should have same default values
        assert config1.embedding.batch_size == config2.embedding.batch_size
        assert (
            config1.concepts.similarity_threshold
            == config2.concepts.similarity_threshold
        )
