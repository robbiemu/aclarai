"""
Shared configuration system for aclarai services.
This module provides environment variable injection from .env files with
fallback logic for external database connections using host.docker.internal.
"""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Any, Dict, List, Optional, Union

import yaml

StrPath = Union[str, "os.PathLike[str]"]


try:
    from dotenv import load_dotenv
except ImportError:
    # Fallback if python-dotenv is not available
    def load_dotenv(
        dotenv_path: Optional[StrPath] = None,  # noqa: ARG001
        stream: Optional[IO[str]] = None,  # noqa: ARG001
        verbose: bool = False,  # noqa: ARG001
        override: bool = False,  # noqa: ARG001
        interpolate: bool = True,  # noqa: ARG001
        encoding: Optional[str] = "utf-8",  # noqa: ARG001
    ) -> bool:  # noqa: F841
        return False


logger = logging.getLogger(__name__)


# In shared/aclarai_shared/config.py


@dataclass
class ProcessingConfig:
    """Configuration for data processing parameters."""

    temperature: float = 0.1
    max_tokens: int = 1000
    timeout_seconds: int = 30
    claimify: Dict[str, Any] = field(default_factory=dict)
    batch_sizes: Dict[str, Any] = field(default_factory=dict)
    retries: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EmbeddingConfig:
    """Configuration for embedding models and vector storage."""

    # Model settings
    default_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    device: str = "auto"
    batch_size: int = 32
    # PGVector settings
    collection_name: str = "utterances"
    embed_dim: int = 384
    index_type: str = "ivfflat"
    index_lists: int = 100
    # Chunking settings
    chunk_size: int = 300
    chunk_overlap: int = 30
    keep_separator: bool = True
    merge_colon_endings: bool = True
    merge_short_prefixes: bool = True
    min_chunk_tokens: int = 5


@dataclass
class ConceptsConfig:
    """Configuration for concept detection and management."""

    # Concept candidates settings
    candidates_collection: str = "concept_candidates"
    similarity_threshold: float = 0.9
    # Canonical concepts settings
    canonical_collection: str = "concepts"
    merge_threshold: float = 0.95


@dataclass
class NounPhraseExtractionConfig:
    """Configuration for noun phrase extraction from Claims and Summaries."""

    # spaCy model configuration
    spacy_model: str = "en_core_web_sm"
    # Normalization settings
    min_phrase_length: int = 2
    filter_digits_only: bool = True
    # Vector storage settings for concept_candidates
    concept_candidates_collection: str = "concept_candidates"
    status_field: str = "status"
    default_status: str = "pending"


@dataclass
class ConceptSummariesConfig:
    """Configuration for the Concept Summary Agent."""

    model: str = "gpt-4"
    max_examples: int = 5
    skip_if_no_claims: bool = True
    include_see_also: bool = True


@dataclass
class SubjectSummariesConfig:
    """Configuration for the Subject Summary Agent and concept clustering."""

    model: str = "gpt-3.5-turbo"
    similarity_threshold: float = 0.92
    min_concepts: int = 3
    max_concepts: int = 15
    allow_web_search: bool = True
    skip_if_incoherent: bool = False


@dataclass
class ThresholdConfig:
    """Configuration for various similarity and processing thresholds."""

    # Cosine similarity threshold for merging candidates
    concept_merge: float = 0.90
    # Minimum link strength to create graph edge
    claim_link_strength: float = 0.60
    # Cosine similarity for grouping utterances for Tier 2 summaries
    summary_grouping_similarity: float = 0.80
    # Quality threshold for claim promotion and linking (geometric mean of evaluation scores)
    claim_quality: float = 0.70


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
        """Build database connection URL.
        
        Returns a connection URL in the format:
        postgresql://user:password@host:port/database
        
        If POSTGRES_URL environment variable is set, it takes precedence over
        individual connection parameters.
        """
        # If POSTGRES_URL is set in environment, use it directly
        postgres_url = os.getenv("POSTGRES_URL")
        if postgres_url and scheme == "postgresql":
            return postgres_url
            
        # Otherwise build from components
        if self.database:
            return f"{scheme}://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"
        return f"{scheme}://{self.user}:{self.password}@{self.host}:{self.port}"

    def get_neo4j_bolt_url(self) -> str:
        """Get Neo4j bolt connection URL."""
        return f"bolt://{self.host}:{self.port}"


@dataclass
class VaultWatcherConfig:
    """Configuration for vault watcher service."""

    batch_interval: float = 2.0
    max_batch_size: int = 50
    queue_name: str = "aclarai_dirty_blocks"
    exchange: str = ""
    routing_key: str = "aclarai_dirty_blocks"


@dataclass
class JobConfig:
    """Configuration for a scheduled job."""

    enabled: bool = True
    manual_only: bool = False
    cron: str = "0 3 * * *"
    description: str = ""


@dataclass
class TopConceptsJobConfig(JobConfig):
    """Configuration for the Top Concepts job."""

    metric: str = "pagerank"  # pagerank | degree
    count: Optional[int] = 25  # number of top concepts (exclusive with percent)
    percent: Optional[float] = None  # use top N% instead of fixed count
    target_file: str = "Top Concepts.md"  # target file name in vault


@dataclass
class TrendingTopicsJobConfig(JobConfig):
    """Configuration for the Trending Topics job."""

    window_days: int = 7  # How far back to look for change
    count: Optional[int] = None  # number of trending concepts (exclusive with percent)
    percent: Optional[float] = 5  # use top N% instead of fixed count
    min_mentions: int = 2  # minimum mentions required to be considered
    target_file: str = (
        "Trending Topics - {date}.md"  # target file name with date placeholder
    )


@dataclass
class ConceptClusteringJobConfig(JobConfig):
    """Configuration for the Concept Clustering job."""

    similarity_threshold: float = 0.92  # Similarity threshold for clustering
    min_concepts: int = 3  # Minimum concepts per cluster
    max_concepts: int = 15  # Maximum concepts per cluster
    algorithm: str = "dbscan"  # Clustering algorithm: "dbscan", "hierarchical"
    cache_ttl: int = 3600  # Cache TTL in seconds (1 hour)
    use_persistent_cache: bool = True  # Whether to use persistent cache


@dataclass
class ConceptSubjectLinkingJobConfig(JobConfig):
    """Configuration for the Concept Subject Linking job."""

    create_neo4j_edges: bool = (
        False  # Whether to create (:Concept)-[:PART_OF]->(:Subject) edges
    )
    batch_size: int = 50  # Number of concepts to process in one batch
    footer_section_title: str = "Part of Subjects"  # Title for footer section


@dataclass
class SchedulerJobsConfig:
    """Configuration for all scheduled jobs."""

    concept_embedding_refresh: JobConfig = field(
        default_factory=lambda: JobConfig(
            enabled=True,
            manual_only=False,
            cron="0 3 * * *",
            description="Refresh concept embeddings from Tier 3 pages",
        )
    )
    vault_sync: JobConfig = field(
        default_factory=lambda: JobConfig(
            enabled=True,
            manual_only=False,
            cron="*/30 * * * *",
            description="Sync vault files with knowledge graph",
        )
    )
    top_concepts: TopConceptsJobConfig = field(
        default_factory=lambda: TopConceptsJobConfig(
            enabled=True,
            manual_only=False,
            cron="0 4 * * *",  # 4 AM daily
            description="Generate Top Concepts.md from PageRank analysis",
            metric="pagerank",
            count=25,
            percent=None,
            target_file="Top Concepts.md",
        )
    )
    trending_topics: TrendingTopicsJobConfig = field(
        default_factory=lambda: TrendingTopicsJobConfig(
            enabled=True,
            manual_only=False,
            cron="0 5 * * *",  # 5 AM daily
            description="Generate Trending Topics - <date>.md from concept mention deltas",
            window_days=7,
            count=None,
            percent=5,
            min_mentions=2,
            target_file="Trending Topics - {date}.md",
        )
    )
    concept_highlight_refresh: JobConfig = field(
        default_factory=lambda: JobConfig(
            enabled=True,
            manual_only=False,
            cron="0 6 * * *",  # 6 AM daily
            description="Generate both Top Concepts and Trending Topics highlight files",
        )
    )
    concept_summary_refresh: JobConfig = field(
        default_factory=lambda: JobConfig(
            enabled=True,
            manual_only=False,
            cron="0 7 * * *",  # 7 AM daily
            description="Generate concept summary pages for all canonical concepts",
        )
    )
    concept_clustering: ConceptClusteringJobConfig = field(
        default_factory=lambda: ConceptClusteringJobConfig(
            enabled=True,
            manual_only=False,
            cron="0 2 * * *",  # 2 AM daily
            description="Group related concepts into thematic clusters",
            similarity_threshold=0.92,
            min_concepts=3,
            max_concepts=15,
            algorithm="dbscan",
            cache_ttl=3600,
            use_persistent_cache=True,
        )
    )
    subject_summary_refresh: JobConfig = field(
        default_factory=lambda: JobConfig(
            enabled=True,
            manual_only=False,
            cron="0 6 * * *",  # 6 AM daily (after clustering at 2 AM)
            description="Generate [[Subject:XYZ]] pages from concept clusters",
        )
    )
    concept_subject_linking: ConceptSubjectLinkingJobConfig = field(
        default_factory=lambda: ConceptSubjectLinkingJobConfig(
            enabled=True,
            manual_only=False,
            cron="0 8 * * *",  # 8 AM daily (after subject summary at 6 AM)
            description="Link concepts to their subjects with footer links",
            create_neo4j_edges=False,
            batch_size=50,
            footer_section_title="Part of Subjects",
        )
    )


@dataclass
class SchedulerConfig:
    """Configuration for scheduler service."""

    jobs: SchedulerJobsConfig = field(default_factory=SchedulerJobsConfig)


@dataclass
class LLMConfig:
    """Configuration for LLM providers and models."""

    provider: str = "openai"
    model: str = "gpt-3.5-turbo"
    api_key: Optional[str] = None
    model_params: Dict[str, Any] = field(default_factory=lambda: {})


@dataclass
class aclaraiConfig:
    """Main configuration class for aclarai services."""

    # LLM configuration
    llm: LLMConfig = field(default_factory=LLMConfig)

    # Database configurations
    postgres: DatabaseConfig = field(
        default_factory=lambda: DatabaseConfig("", 0, "", "")
    )
    neo4j: DatabaseConfig = field(default_factory=lambda: DatabaseConfig("", 0, "", ""))
    # New configuration sections
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    concepts: ConceptsConfig = field(default_factory=ConceptsConfig)
    noun_phrase_extraction: NounPhraseExtractionConfig = field(
        default_factory=NounPhraseExtractionConfig
    )
    concept_summaries: ConceptSummariesConfig = field(
        default_factory=ConceptSummariesConfig
    )
    subject_summaries: SubjectSummariesConfig = field(
        default_factory=SubjectSummariesConfig
    )
    threshold: ThresholdConfig = field(default_factory=ThresholdConfig)
    vault_watcher: VaultWatcherConfig = field(default_factory=VaultWatcherConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    # Message broker configuration
    rabbitmq_host: str = "rabbitmq"
    rabbitmq_port: int = 5672
    rabbitmq_user: str = "user"
    rabbitmq_password: str = ""
    # Service configuration
    vault_path: str = "/vault"
    settings_path: str = "/settings"
    log_level: str = "INFO"
    debug: bool = False
    # AI/ML configuration
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    # Vault structure configuration
    paths: PathsConfig = field(default_factory=PathsConfig)
    # Feature flags
    features: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_env(
        cls, env_file: Optional[str] = None, config_file: Optional[str] = None
    ) -> "aclaraiConfig":
        """
        Create configuration from environment variables and YAML config file.
        Args:
            env_file: Path to .env file (optional, defaults to searching for .env)
            config_file: Path to YAML config file (optional, defaults to settings/aclarai.config.yaml)
        """
        # Load YAML configuration first
        yaml_config = cls._load_yaml_config(config_file)
        # Load .env file if specified or found
        if env_file:
            load_dotenv(env_file)
        else:
            # Try to find .env file in current directory or parent directories
            current_path = Path.cwd()
            for path in [current_path] + list(current_path.parents):
                env_path = path / ".env"
                if env_path.exists():
                    logger.info(f"Loading environment variables from {env_path}")
                    load_dotenv(env_path)
                    break
        # PostgreSQL configuration with fallback
        postgres_config = yaml_config.get("databases", {}).get("postgres", {})
        postgres_host = os.getenv(
            "POSTGRES_HOST", postgres_config.get("host", "postgres")
        )
        postgres_host = cls._apply_host_fallback(postgres_host)
        postgres = DatabaseConfig(
            host=postgres_host,
            port=int(os.getenv("POSTGRES_PORT", postgres_config.get("port", "5432"))),
            user=os.getenv("POSTGRES_USER", "aclarai"),
            password=os.getenv("POSTGRES_PASSWORD", ""),
            database=os.getenv(
                "POSTGRES_DB", postgres_config.get("database", "aclarai")
            ),
        )
        # Neo4j configuration with fallback
        neo4j_config = yaml_config.get("databases", {}).get("neo4j", {})
        neo4j_host = os.getenv("NEO4J_HOST", neo4j_config.get("host", "neo4j"))
        neo4j_host = cls._apply_host_fallback(neo4j_host)
        neo4j = DatabaseConfig(
            host=neo4j_host,
            port=int(os.getenv("NEO4J_BOLT_PORT", neo4j_config.get("port", "7687"))),
            user=os.getenv("NEO4J_USER", "neo4j"),
            password=os.getenv("NEO4J_PASSWORD", ""),
        )
        # Load embedding configuration from YAML
        embedding_config = yaml_config.get("embedding", {})
        embedding = EmbeddingConfig(
            default_model=embedding_config.get("models", {}).get(
                "default", "sentence-transformers/all-MiniLM-L6-v2"
            ),
            device=embedding_config.get("device", "auto"),
            batch_size=embedding_config.get("batch_size", 32),
            collection_name=embedding_config.get("pgvector", {}).get(
                "collection_name", "utterances"
            ),
            embed_dim=embedding_config.get("pgvector", {}).get("embed_dim", 384),
            index_type=embedding_config.get("pgvector", {}).get(
                "index_type", "ivfflat"
            ),
            index_lists=embedding_config.get("pgvector", {}).get("index_lists", 100),
            chunk_size=embedding_config.get("chunking", {}).get("chunk_size", 300),
            chunk_overlap=embedding_config.get("chunking", {}).get("chunk_overlap", 30),
            keep_separator=embedding_config.get("chunking", {}).get(
                "keep_separator", True
            ),
            merge_colon_endings=embedding_config.get("chunking", {}).get(
                "merge_colon_endings", True
            ),
            merge_short_prefixes=embedding_config.get("chunking", {}).get(
                "merge_short_prefixes", True
            ),
            min_chunk_tokens=embedding_config.get("chunking", {}).get(
                "min_chunk_tokens", 5
            ),
        )
        # Load processing configuration from YAML
        processing_config_data = yaml_config.get("processing", {})
        processing = ProcessingConfig(
            temperature=processing_config_data.get("temperature", 0.1),
            max_tokens=processing_config_data.get("max_tokens", 1000),
            timeout_seconds=processing_config_data.get("timeout_seconds", 30),
            claimify=processing_config_data.get("claimify", {}),
            batch_sizes=processing_config_data.get("batch_sizes", {}),
            retries=processing_config_data.get("retries", {}),
        )
        # Load concepts configuration from YAML
        concepts_config = yaml_config.get("concepts", {})
        concepts = ConceptsConfig(
            candidates_collection=concepts_config.get("candidates", {}).get(
                "collection_name", "concept_candidates"
            ),
            similarity_threshold=concepts_config.get("candidates", {}).get(
                "similarity_threshold", 0.9
            ),
            canonical_collection=concepts_config.get("canonical", {}).get(
                "collection_name", "concepts"
            ),
            merge_threshold=concepts_config.get("canonical", {}).get(
                "similarity_threshold", 0.95
            ),
        )
        # Load noun phrase extraction configuration from YAML
        noun_phrase_config = yaml_config.get("noun_phrase_extraction", {})
        noun_phrase_extraction = NounPhraseExtractionConfig(
            spacy_model=noun_phrase_config.get("spacy_model", "en_core_web_sm"),
            min_phrase_length=noun_phrase_config.get("min_phrase_length", 2),
            filter_digits_only=noun_phrase_config.get("filter_digits_only", True),
            concept_candidates_collection=noun_phrase_config.get(
                "concept_candidates", {}
            ).get("collection_name", "concept_candidates"),
            status_field=noun_phrase_config.get("concept_candidates", {}).get(
                "status_field", "status"
            ),
            default_status=noun_phrase_config.get("concept_candidates", {}).get(
                "default_status", "pending"
            ),
        )
        # Load concept summaries configuration from YAML
        concept_summaries_config = yaml_config.get("concept_summaries", {})
        concept_summaries = ConceptSummariesConfig(
            model=concept_summaries_config.get("model", "gpt-4"),
            max_examples=concept_summaries_config.get("max_examples", 5),
            skip_if_no_claims=concept_summaries_config.get("skip_if_no_claims", True),
            include_see_also=concept_summaries_config.get("include_see_also", True),
        )
        # Load threshold configuration from YAML
        threshold_config = yaml_config.get("threshold", {})
        threshold = ThresholdConfig(
            concept_merge=threshold_config.get("concept_merge", 0.90),
            claim_link_strength=threshold_config.get("claim_link_strength", 0.60),
            summary_grouping_similarity=threshold_config.get(
                "summary_grouping_similarity", 0.80
            ),
            claim_quality=threshold_config.get("claim_quality", 0.70),
        )
        # Load paths configuration from YAML
        paths_config = yaml_config.get("paths", {})
        vault_path = os.getenv("VAULT_PATH", paths_config.get("vault", "/vault"))
        settings_path = os.getenv(
            "SETTINGS_PATH", paths_config.get("settings", "/settings")
        )
        # Vault paths configuration (from main branch)
        paths = PathsConfig(
            vault=vault_path,
            settings=settings_path,
            tier1=os.getenv("VAULT_TIER1_PATH", paths_config.get("tier1", "tier1")),
            tier2=os.getenv("VAULT_SUMMARIES_PATH", paths_config.get("summaries", ".")),
            tier3=os.getenv("VAULT_CONCEPTS_PATH", paths_config.get("concepts", ".")),
            logs=os.getenv(
                "VAULT_LOGS_PATH", paths_config.get("logs", ".aclarai/import_logs")
            ),
        )
        # Load features configuration from YAML
        features_config = yaml_config.get("features", {})
        # Load vault watcher configuration from YAML
        vault_watcher_config = yaml_config.get("vault_watcher", {})
        vault_watcher = VaultWatcherConfig(
            batch_interval=vault_watcher_config.get("batch_interval", 2.0),
            max_batch_size=vault_watcher_config.get("max_batch_size", 50),
            queue_name=vault_watcher_config.get("rabbitmq", {}).get(
                "queue_name", "aclarai_dirty_blocks"
            ),
            exchange=vault_watcher_config.get("rabbitmq", {}).get("exchange", ""),
            routing_key=vault_watcher_config.get("rabbitmq", {}).get(
                "routing_key", "aclarai_dirty_blocks"
            ),
        )
        # Load scheduler configuration from YAML
        scheduler_config = yaml_config.get("scheduler", {})
        jobs_config = scheduler_config.get("jobs", {})

        # Load concept embedding refresh job config
        concept_refresh_config = jobs_config.get("concept_embedding_refresh", {})
        concept_embedding_refresh = JobConfig(
            enabled=concept_refresh_config.get("enabled", True),
            manual_only=concept_refresh_config.get("manual_only", False),
            cron=concept_refresh_config.get("cron", "0 3 * * *"),
            description=concept_refresh_config.get(
                "description", "Refresh concept embeddings from Tier 3 pages"
            ),
        )

        # Load vault sync job config
        vault_sync_config = jobs_config.get("vault_sync", {})
        vault_sync = JobConfig(
            enabled=vault_sync_config.get("enabled", True),
            manual_only=vault_sync_config.get("manual_only", False),
            cron=vault_sync_config.get("cron", "*/30 * * * *"),
            description=vault_sync_config.get(
                "description", "Sync vault files with knowledge graph"
            ),
        )

        scheduler = SchedulerConfig(
            jobs=SchedulerJobsConfig(
                concept_embedding_refresh=concept_embedding_refresh,
                vault_sync=vault_sync,
            )
        )
        return cls(
            postgres=postgres,
            neo4j=neo4j,
            embedding=embedding,
            processing=processing,
            concepts=concepts,
            noun_phrase_extraction=noun_phrase_extraction,
            concept_summaries=concept_summaries,
            threshold=threshold,
            vault_watcher=vault_watcher,
            scheduler=scheduler,
            rabbitmq_host=os.getenv("RABBITMQ_HOST", "rabbitmq"),
            rabbitmq_port=int(os.getenv("RABBITMQ_PORT", "5672")),
            rabbitmq_user=os.getenv("RABBITMQ_USER", "user"),
            rabbitmq_password=os.getenv("RABBITMQ_PASSWORD", ""),
            vault_path=vault_path,  # For backward compatibility
            settings_path=settings_path,  # For backward compatibility
            log_level=os.getenv(
                "LOG_LEVEL", yaml_config.get("logging", {}).get("level", "INFO")
            ),
            debug=os.getenv("DEBUG", "false").lower() == "true",
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
            paths=paths,
            features=features_config,
        )

    @classmethod
    def _load_yaml_config(cls, config_file: Optional[str] = None) -> Dict[str, Any]:
        """Load configuration from YAML file with default config merge."""
        if yaml is None:
            logger.warning("PyYAML not available, skipping YAML config loading")
            return {}
        # Load default configuration first
        default_config = cls._load_default_config()
        # Find user configuration file
        user_config_file = config_file
        if user_config_file is None:
            user_config_file = cls._find_user_config_file()
        # Load user configuration if it exists
        user_config: Dict[str, Any] = {}
        if user_config_file and Path(user_config_file).exists():
            logger.info(f"Loading YAML configuration from {user_config_file}")
            try:
                with open(user_config_file, "r") as f:
                    user_config = yaml.safe_load(f) or {}
            except Exception as e:
                logger.error(f"Failed to load YAML config from {user_config_file}: {e}")
                user_config = {}
        else:
            logger.info("No user YAML configuration file found, using defaults only")
        # Deep merge user config over default config
        merged_config = cls._deep_merge_configs(default_config, user_config)
        return merged_config

    @classmethod
    def _load_default_config(cls) -> Dict[str, Any]:
        """Load the default configuration file."""
        if yaml is None:
            return {}
        # Look for aclarai.config.default.yaml in shared package directory
        current_path = Path.cwd()
        search_paths = []
        # Priority 1: same directory as this module (shared package)
        module_path = Path(__file__).parent
        search_paths.append(module_path / "aclarai.config.default.yaml")
        # Priority 2: in shared/ directory relative to current working directory
        for path in [current_path] + list(current_path.parents):
            search_paths.append(
                path / "shared" / "aclarai_shared" / "aclarai.config.default.yaml"
            )
        for config_path in search_paths:
            if config_path.exists():
                logger.debug(f"Loading default configuration from {config_path}")
                try:
                    with open(config_path, "r") as f:
                        return yaml.safe_load(f) or {}
                except Exception as e:
                    logger.error(
                        f"Failed to load default config from {config_path}: {e}"
                    )
                    continue
        logger.warning("No default configuration file found, using hardcoded defaults")
        return {}

    @classmethod
    def _find_user_config_file(cls) -> Optional[str]:
        """Find the user configuration file."""
        current_path = Path.cwd()
        search_paths = []
        # Priority 1: settings directory in current and parent directories
        for path in [current_path] + list(current_path.parents):
            search_paths.append(path / "settings" / "aclarai.config.yaml")
        # Priority 2: root level in current and parent directories
        for path in [current_path] + list(current_path.parents):
            search_paths.append(path / "aclarai.config.yaml")
        for config_path in search_paths:
            if config_path.exists():
                return str(config_path)
        return None

    @staticmethod
    def _deep_merge_configs(
        default: Dict[str, Any], user: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Deep merge user configuration over default configuration.
        Args:
            default: Default configuration dictionary
            user: User configuration dictionary
        Returns:
            Merged configuration dictionary
        """
        import copy

        result = copy.deepcopy(default)

        def _merge_recursive(base_dict: Dict[str, Any], override_dict: Dict[str, Any]):
            for key, value in override_dict.items():
                if (
                    key in base_dict
                    and isinstance(base_dict[key], dict)
                    and isinstance(value, dict)
                ):
                    _merge_recursive(base_dict[key], value)
                else:
                    base_dict[key] = value

        _merge_recursive(result, user)
        return result

    @staticmethod
    def _apply_host_fallback(host: str) -> str:
        """
        Apply host.docker.internal fallback for external database connections.
        If the host appears to be external (not a Docker service name),
        and we're running in a Docker container, use host.docker.internal.
        """
        # If explicitly set to host.docker.internal, keep it
        if host == "host.docker.internal":
            return host
        # Check if we're running in Docker by looking for typical indicators
        running_in_docker = (
            os.path.exists("/.dockerenv") or os.getenv("DOCKER_CONTAINER") == "true"
        )
        # List of common Docker service names (internal services)
        docker_services = {
            "postgres",
            "neo4j",
            "rabbitmq",
            "aclarai-core",
            "vault-watcher",
            "scheduler",
        }
        # If running in Docker and host is not localhost/127.0.0.1, not a known service and if it's an IP address or external hostname
        if (
            running_in_docker
            and host not in docker_services
            and host not in ("localhost", "127.0.0.1")
            and aclaraiConfig._is_external_host(host)
        ):
            return "host.docker.internal"
        return host

    @staticmethod
    def _is_external_host(host: str) -> bool:
        """Check if a host appears to be external (IP address or FQDN)."""
        import re

        # Check if it's an IP address pattern
        ip_pattern = re.compile(r"^(\d{1,3}\.){3}\d{1,3}$")
        if ip_pattern.match(host):
            return True
        # Check if it contains dots (likely FQDN)
        return bool("." in host and not host.startswith("localhost"))

    def validate_required_vars(
        self, required_vars: Optional[List[str]] = None
    ) -> List[str]:
        """
        Validate that required environment variables are set.
        Args:
            required_vars: List of required variable names (optional)
        Returns:
            List of missing variables
        """
        if required_vars is None:
            required_vars = ["POSTGRES_PASSWORD", "NEO4J_PASSWORD"]
        missing_vars = []
        # Check environment variables directly
        for var in required_vars:
            value = os.getenv(var)
            if not value:
                missing_vars.append(var)
        # Only check config if no direct env vars were specified
        if not required_vars or len(missing_vars) == len(required_vars):
            # Additional validation for database connections via config
            if not self.postgres.password and "POSTGRES_PASSWORD" not in missing_vars:
                missing_vars.append("POSTGRES_PASSWORD (via config)")
            if not self.neo4j.password and "NEO4J_PASSWORD" not in missing_vars:
                missing_vars.append("NEO4J_PASSWORD (via config)")
        return list(set(missing_vars))  # Remove duplicates

    def setup_logging(self):
        """Configure logging based on the log level setting."""
        log_level = getattr(logging, self.log_level.upper(), logging.INFO)
        logging.basicConfig(
            level=log_level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        if self.debug:
            logging.getLogger().setLevel(logging.DEBUG)


def load_config(
    env_file: Optional[str] = None,
    config_file: Optional[str] = None,
    validate: bool = True,
    required_vars: Optional[List[str]] = None,
) -> aclaraiConfig:
    """
    Load aclarai configuration with validation.
    Args:
        env_file: Path to .env file (optional)
        config_file: Path to YAML config file (optional)
        validate: Whether to validate required variables
        required_vars: List of required variables to validate
    Returns:
        aclaraiConfig instance
    Raises:
        ValueError: If required variables are missing
    """
    config = aclaraiConfig.from_env(env_file, config_file)
    # Validate required environment variables for security
    if validate:
        # Check other required vars (this will include database passwords)
        missing_vars = config.validate_required_vars(required_vars)
        if missing_vars:
            # Remove duplicates and sort for consistent output
            missing_vars = sorted(set(missing_vars))
            error_msg = (
                f"Missing required environment variables: {', '.join(missing_vars)}. "
                f"Please check your .env file or environment configuration."
            )
            logger.error(error_msg)
            raise ValueError(error_msg)
    config.setup_logging()
    return config
