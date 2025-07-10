# shared/aclarai_shared/config.py

"""
Shared configuration system for aclarai services.
This module provides environment variable injection from .env files with
fallback logic for external database connections using host.docker.internal.
"""

import logging
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import IO, Any, Dict, List, Optional, Type, TypeVar, Union

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


@dataclass
class ClaimifyModelConfig:
    """Configuration for Claimify-specific models."""

    default: str = "gpt-3.5-turbo"
    selection: Optional[str] = None
    disambiguation: Optional[str] = None
    decomposition: Optional[str] = None
    entailment: Optional[str] = None


@dataclass
class ModelConfig:
    """Configuration for all agent and LLM models."""

    claimify: ClaimifyModelConfig = field(default_factory=ClaimifyModelConfig)
    concept_linker: str = "gpt-3.5-turbo"
    concept_summary: str = "gpt-4"
    subject_summary: str = "gpt-3.5-turbo"
    trending_concepts_agent: str = "gpt-4"
    fallback_plugin: str = "gpt-3.5-turbo"


@dataclass
class ProcessingConfig:
    """Configuration for data processing parameters."""

    temperature: float = 0.1
    max_tokens: int = 1000
    timeout_seconds: int = 30
    claimify: Dict[str, Any] = field(default_factory=dict)
    batch_sizes: Dict[str, Any] = field(default_factory=dict)
    retries: Dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value by key with fallback to default."""
        return getattr(self, key, default)

    def __getitem__(self, key: str) -> Any:
        """Allow dictionary-style access to configuration values."""
        return getattr(self, key)

    def __contains__(self, key: str) -> bool:
        """Check if a configuration key exists."""
        return hasattr(self, key)


@dataclass
class EmbeddingConfig:
    """Configuration for embedding models and vector storage."""

    # Model settings
    default_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    utterance: Optional[str] = None
    concept: Optional[str] = None
    summary: Optional[str] = None
    fallback: Optional[str] = None
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

    concept_merge: float = 0.90
    claim_link_strength: float = 0.60
    summary_grouping_similarity: float = 0.80
    claim_quality: float = 0.70


@dataclass
class WindowConfig:
    """Configuration for context windows."""

    p: int = 3
    f: int = 1


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
        postgres_url = os.getenv("POSTGRES_URL")
        if postgres_url and scheme == "postgresql":
            return postgres_url
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


T = TypeVar("T", bound=JobConfig)


@dataclass
class TopConceptsJobConfig(JobConfig):
    """Configuration for the Top Concepts job."""

    metric: str = "pagerank"
    count: Optional[int] = 25
    percent: Optional[float] = None
    target_file: str = "Top Concepts.md"


@dataclass
class TrendingTopicsJobConfig(JobConfig):
    """Configuration for the Trending Topics job."""

    window_days: int = 7
    count: Optional[int] = None
    percent: Optional[float] = 5
    min_mentions: int = 2
    target_file: str = "Trending Topics - {date}.md"


@dataclass
class ConceptClusteringJobConfig(JobConfig):
    """Configuration for the Concept Clustering job."""

    similarity_threshold: float = 0.92
    min_concepts: int = 3
    max_concepts: int = 15
    algorithm: str = "dbscan"
    cache_ttl: int = 3600
    use_persistent_cache: bool = True


@dataclass
class ConceptSubjectLinkingJobConfig(JobConfig):
    """Configuration for the Concept Subject Linking job."""

    create_neo4j_edges: bool = False
    batch_size: int = 50
    footer_section_title: str = "Part of Subjects"


@dataclass
class SchedulerJobsConfig:
    """Configuration for all scheduled jobs."""

    concept_embedding_refresh: JobConfig = field(default_factory=JobConfig)
    vault_sync: JobConfig = field(default_factory=JobConfig)
    top_concepts: TopConceptsJobConfig = field(default_factory=TopConceptsJobConfig)
    trending_topics: TrendingTopicsJobConfig = field(
        default_factory=TrendingTopicsJobConfig
    )
    concept_highlight_refresh: JobConfig = field(default_factory=JobConfig)
    concept_summary_refresh: JobConfig = field(default_factory=JobConfig)
    concept_clustering: ConceptClusteringJobConfig = field(
        default_factory=ConceptClusteringJobConfig
    )
    subject_summary_refresh: JobConfig = field(default_factory=JobConfig)
    concept_subject_linking: ConceptSubjectLinkingJobConfig = field(
        default_factory=ConceptSubjectLinkingJobConfig
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
class ServiceDiscoveryConfig:
    """Configuration for service discovery and connection preferences."""

    prefer_docker_services: bool = True


@dataclass
class aclaraiConfig:
    """Main configuration class for aclarai services."""

    service_discovery: ServiceDiscoveryConfig = field(default_factory=ServiceDiscoveryConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    threshold: ThresholdConfig = field(default_factory=ThresholdConfig)
    window: Dict[str, WindowConfig] = field(default_factory=dict)
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
    postgres: DatabaseConfig = field(
        default_factory=lambda: DatabaseConfig("", 0, "", "")
    )
    neo4j: DatabaseConfig = field(default_factory=lambda: DatabaseConfig("", 0, "", ""))
    vault_watcher: VaultWatcherConfig = field(default_factory=VaultWatcherConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    rabbitmq_host: str = "rabbitmq"
    rabbitmq_port: int = 5672
    rabbitmq_user: str = "user"
    rabbitmq_password: str = ""
    paths: PathsConfig = field(default_factory=PathsConfig)
    log_level: str = "INFO"
    debug: bool = False
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    features: Dict[str, Any] = field(default_factory=dict)
    vault_path: str = "/vault"
    settings_path: str = "/settings"

    @classmethod
    def from_env(
        cls, env_file: Optional[str] = None, config_file: Optional[str] = None
    ) -> "aclaraiConfig":
        """Create configuration from environment variables and YAML config file."""
        yaml_config = cls._load_yaml_config(config_file)
        if env_file:
            load_dotenv(env_file)
        else:
            current_path = Path.cwd()
            for path in [current_path] + list(current_path.parents):
                env_path = path / ".env"
                if env_path.exists():
                    logger.info(f"Loading environment variables from {env_path}")
                    load_dotenv(env_path)
                    break

        def filter_none(data: Dict) -> Dict:
            return {k: v for k, v in data.items() if v is not None}

        # Load service discovery config first to apply it to database config
        service_discovery_data = yaml_config.get("service_discovery", {})
        prefer_docker_services = service_discovery_data.get("prefer_docker_services", True)

        # --- Database Config ---
        postgres_data = yaml_config.get("databases", {}).get("postgres", {})
        postgres_host = os.getenv(
            "POSTGRES_HOST", postgres_data.get("host", "postgres")
        )
        postgres_host = cls._apply_service_host_fallback(postgres_host, "postgres", prefer_docker_services)
        postgres = DatabaseConfig(
            host=postgres_host,
            port=int(os.getenv("POSTGRES_PORT", postgres_data.get("port", 5432))),
            user=os.getenv("POSTGRES_USER", "aclarai"),
            password=os.getenv("POSTGRES_PASSWORD", ""),
            database=os.getenv("POSTGRES_DB", postgres_data.get("database", "aclarai")),
        )
        neo4j_data = yaml_config.get("databases", {}).get("neo4j", {})
        neo4j_host = os.getenv("NEO4J_HOST", neo4j_data.get("host", "neo4j"))
        neo4j_host = cls._apply_service_host_fallback(neo4j_host, "neo4j", prefer_docker_services)
        neo4j = DatabaseConfig(
            host=neo4j_host,
            port=int(os.getenv("NEO4J_BOLT_PORT", neo4j_data.get("port", 7687))),
            user=os.getenv("NEO4J_USER", "neo4j"),
            password=os.getenv("NEO4J_PASSWORD", ""),
        )

        # --- Service Discovery Config ---
        service_discovery = ServiceDiscoveryConfig(**filter_none(service_discovery_data))

        # --- LLM & Model Config ---
        llm_data = yaml_config.get("llm", {})
        llm = LLMConfig(**filter_none(llm_data))

        model_data = yaml_config.get("model", {})
        claimify_data = model_data.get("claimify", {})
        claimify_default = claimify_data.get("default", ClaimifyModelConfig.default)
        claimify_config = ClaimifyModelConfig(
            default=claimify_default,
            selection=claimify_data.get("selection") or claimify_default,
            disambiguation=claimify_data.get("disambiguation") or claimify_default,
            decomposition=claimify_data.get("decomposition") or claimify_default,
            entailment=claimify_data.get("entailment") or claimify_default,
        )
        model = ModelConfig(
            claimify=claimify_config,
            **filter_none({k: v for k, v in model_data.items() if k != "claimify"}),
        )

        # --- Embedding Config ---
        embedding_data = yaml_config.get("embedding", {})
        pgvector_data = embedding_data.get("pgvector", {})
        chunking_data = embedding_data.get("chunking", {})
        # Use the legacy 'models.default' if present, otherwise use dataclass default
        default_embedding = embedding_data.get("models", {}).get(
            "default", EmbeddingConfig.default_model
        )
        embedding = EmbeddingConfig(
            default_model=default_embedding,
            utterance=embedding_data.get("utterance") or default_embedding,
            concept=embedding_data.get("concept") or default_embedding,
            summary=embedding_data.get("summary") or default_embedding,
            fallback=embedding_data.get("fallback") or default_embedding,
            device=embedding_data.get("device"),
            batch_size=embedding_data.get("batch_size"),
            collection_name=pgvector_data.get("collection_name"),
            embed_dim=pgvector_data.get("embed_dim"),
            index_type=pgvector_data.get("index_type"),
            index_lists=pgvector_data.get("index_lists"),
            chunk_size=chunking_data.get("chunk_size"),
            chunk_overlap=chunking_data.get("chunk_overlap"),
            keep_separator=chunking_data.get("keep_separator"),
            merge_colon_endings=chunking_data.get("merge_colon_endings"),
            merge_short_prefixes=chunking_data.get("merge_short_prefixes"),
            min_chunk_tokens=chunking_data.get("min_chunk_tokens"),
        )
        embedding = EmbeddingConfig(**filter_none(asdict(embedding)))

        # --- Concepts Config (Handles nested structure) ---
        concepts_data = yaml_config.get("concepts", {})
        candidates_data = concepts_data.get("candidates", {})
        canonical_data = concepts_data.get("canonical", {})
        concepts = ConceptsConfig(
            candidates_collection=candidates_data.get("collection_name"),
            similarity_threshold=candidates_data.get("similarity_threshold"),
            canonical_collection=canonical_data.get("collection_name"),
            merge_threshold=canonical_data.get("similarity_threshold"),
        )
        concepts = ConceptsConfig(**filter_none(asdict(concepts)))

        # --- Noun Phrase Extraction (Handles nested structure) ---
        npe_data = yaml_config.get("noun_phrase_extraction", {})
        npe_candidates_data = npe_data.get("concept_candidates", {})
        noun_phrase_extraction = NounPhraseExtractionConfig(
            spacy_model=npe_data.get("spacy_model"),
            min_phrase_length=npe_data.get("min_phrase_length"),
            filter_digits_only=npe_data.get("filter_digits_only"),
            concept_candidates_collection=npe_candidates_data.get("collection_name"),
            status_field=npe_candidates_data.get("status_field"),
            default_status=npe_candidates_data.get("default_status"),
        )
        noun_phrase_extraction = NounPhraseExtractionConfig(
            **filter_none(asdict(noun_phrase_extraction))
        )

        # --- Other Config Sections ---
        processing = ProcessingConfig(**filter_none(yaml_config.get("processing", {})))
        threshold = ThresholdConfig(**filter_none(yaml_config.get("threshold", {})))
        concept_summaries = ConceptSummariesConfig(
            **filter_none(yaml_config.get("concept_summaries", {}))
        )
        subject_summaries = SubjectSummariesConfig(
            **filter_none(yaml_config.get("subject_summaries", {}))
        )

        vault_watcher_data = yaml_config.get("vault_watcher", {})
        vault_watcher_rabbitmq_data = vault_watcher_data.get("rabbitmq", {})
        vault_watcher = VaultWatcherConfig(
            batch_interval=vault_watcher_data.get("batch_interval"),
            max_batch_size=vault_watcher_data.get("max_batch_size"),
            queue_name=vault_watcher_rabbitmq_data.get("queue_name"),
            exchange=vault_watcher_rabbitmq_data.get("exchange"),
            routing_key=vault_watcher_rabbitmq_data.get("routing_key"),
        )
        vault_watcher = VaultWatcherConfig(**filter_none(asdict(vault_watcher)))

        window_data = yaml_config.get("window", {})
        window = {
            "claimify": WindowConfig(**filter_none(window_data.get("claimify", {})))
        }

        # --- Paths Config ---
        paths_data = yaml_config.get("paths", {})
        vault_path = os.getenv("VAULT_PATH", paths_data.get("vault", "/vault"))
        settings_path = os.getenv(
            "SETTINGS_PATH", paths_data.get("settings", "/settings")
        )
        paths = PathsConfig(
            vault=vault_path,
            settings=settings_path,
            tier1=os.getenv("VAULT_TIER1_PATH", paths_data.get("tier1")),
            tier2=os.getenv("VAULT_SUMMARIES_PATH", paths_data.get("tier2")),
            tier3=os.getenv("VAULT_CONCEPTS_PATH", paths_data.get("tier3")),
            logs=os.getenv("VAULT_LOGS_PATH", paths_data.get("logs")),
        )
        paths = PathsConfig(**filter_none(asdict(paths)))

        # --- Scheduler Config ---
        scheduler_data = yaml_config.get("scheduler", {})
        jobs_data = scheduler_data.get("jobs", {})

        def load_job_config(job_key: str, config_cls: Type[T]) -> T:
            job_data = jobs_data.get(job_key, {})
            return config_cls(**filter_none(job_data))

        scheduler_jobs = SchedulerJobsConfig(
            concept_embedding_refresh=load_job_config(
                "concept_embedding_refresh", JobConfig
            ),
            vault_sync=load_job_config("vault_sync", JobConfig),
            top_concepts=load_job_config("top_concepts", TopConceptsJobConfig),
            trending_topics=load_job_config("trending_topics", TrendingTopicsJobConfig),
            concept_highlight_refresh=load_job_config(
                "concept_highlight_refresh", JobConfig
            ),
            concept_summary_refresh=load_job_config(
                "concept_summary_refresh", JobConfig
            ),
            concept_clustering=load_job_config(
                "concept_clustering", ConceptClusteringJobConfig
            ),
            subject_summary_refresh=load_job_config(
                "subject_summary_refresh", JobConfig
            ),
            concept_subject_linking=load_job_config(
                "concept_subject_linking", ConceptSubjectLinkingJobConfig
            ),
        )
        scheduler = SchedulerConfig(jobs=scheduler_jobs)

        return cls(
            service_discovery=service_discovery,
            llm=llm,
            model=model,
            processing=processing,
            embedding=embedding,
            threshold=threshold,
            window=window,
            concepts=concepts,
            noun_phrase_extraction=noun_phrase_extraction,
            concept_summaries=concept_summaries,
            subject_summaries=subject_summaries,
            postgres=postgres,
            neo4j=neo4j,
            vault_watcher=vault_watcher,
            scheduler=scheduler,
            rabbitmq_host=cls._apply_service_host_fallback(
                os.getenv("RABBITMQ_HOST", "rabbitmq"), "rabbitmq", prefer_docker_services
            ),
            rabbitmq_port=int(os.getenv("RABBITMQ_PORT", 5672)),
            rabbitmq_user=os.getenv("RABBITMQ_USER", "user"),
            rabbitmq_password=os.getenv("RABBITMQ_PASSWORD", ""),
            paths=paths,
            log_level=os.getenv(
                "LOG_LEVEL", yaml_config.get("logging", {}).get("level", "INFO")
            ),
            debug=os.getenv("DEBUG", "false").lower() == "true",
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
            features=yaml_config.get("features", {}),
            vault_path=vault_path,
            settings_path=settings_path,
        )

    @classmethod
    def _load_yaml_config(cls, config_file: Optional[str] = None) -> Dict[str, Any]:
        """Load configuration from YAML file with default config merge."""
        if yaml is None:
            logger.warning("PyYAML not available, skipping YAML config loading")
            return {}
        default_config = cls._load_default_config()
        user_config_file = config_file or cls._find_user_config_file()
        user_config: Dict[str, Any] = {}
        if user_config_file and Path(user_config_file).exists():
            logger.info(f"Loading YAML configuration from {user_config_file}")
            try:
                with open(user_config_file, "r") as f:
                    user_config = yaml.safe_load(f) or {}
            except Exception as e:
                logger.error(f"Failed to load YAML config from {user_config_file}: {e}")
        else:
            logger.info("No user YAML configuration file found, using defaults only")
        return cls._deep_merge_configs(default_config, user_config)

    @classmethod
    def _load_default_config(cls) -> Dict[str, Any]:
        """Load the default configuration file."""
        if yaml is None:
            return {}
        module_path = Path(__file__).parent
        search_paths = [
            module_path / "aclarai.config.default.yaml",
            Path.cwd() / "shared" / "aclarai_shared" / "aclarai.config.default.yaml",
        ]
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
        logger.warning("No default configuration file found, using hardcoded defaults")
        return {}

    @classmethod
    def _find_user_config_file(cls) -> Optional[str]:
        """Find the user configuration file."""
        current_path = Path.cwd()
        search_paths = [
            path / "settings" / "aclarai.config.yaml"
            for path in [current_path] + list(current_path.parents)
        ] + [
            path / "aclarai.config.yaml"
            for path in [current_path] + list(current_path.parents)
        ]
        for config_path in search_paths:
            if config_path.exists():
                return str(config_path)
        return None

    @staticmethod
    def _deep_merge_configs(
        default: Dict[str, Any], user: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Deep merge user configuration over default configuration."""
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
    def _apply_service_host_fallback(host: str, service_name: str, prefer_docker_services: bool = True) -> str:
        """Apply Docker service name fallback for a specific service when running in containers."""
        if host == "host.docker.internal":
            return host
        
        running_in_docker = (
            os.path.exists("/.dockerenv") or os.getenv("DOCKER_CONTAINER") == "true"
        )
        
        if not running_in_docker:
            return host
            
        # If prefer_docker_services is False, respect user configuration
        if not prefer_docker_services:
            if aclaraiConfig._is_external_host(host):
                return "host.docker.internal"
            return host
            
        # Handle localhost/127.0.0.1 mapping to specific service names
        localhost_aliases = {"localhost", "127.0.0.1"}
        if host in localhost_aliases:
            return service_name
        
        docker_services = {
            "postgres",
            "neo4j", 
            "rabbitmq",
            "aclarai-core",
            "vault-watcher",
            "scheduler",
        }
        
        if host in docker_services:
            return host
            
        if aclaraiConfig._is_external_host(host):
            return "host.docker.internal"
            
        return host

    @staticmethod
    def _apply_host_fallback(host: str, prefer_docker_services: bool = True) -> str:
        """Apply Docker service name fallback when running in containers."""
        if host == "host.docker.internal":
            return host
        
        running_in_docker = (
            os.path.exists("/.dockerenv") or os.getenv("DOCKER_CONTAINER") == "true"
        )
        
        if not running_in_docker:
            return host
            
        # If prefer_docker_services is False, respect user configuration
        if not prefer_docker_services:
            if aclaraiConfig._is_external_host(host):
                return "host.docker.internal"
            return host
            
        # Handle localhost/127.0.0.1 mapping - return as-is for now
        # The service-specific mapping will be handled by individual calls
        localhost_aliases = {"localhost", "127.0.0.1"}
        if host in localhost_aliases:
            # Return localhost for further processing by caller
            return host
        
        docker_services = {
            "postgres",
            "neo4j", 
            "rabbitmq",
            "aclarai-core",
            "vault-watcher",
            "scheduler",
        }
        
        if host in docker_services:
            return host
            
        if aclaraiConfig._is_external_host(host):
            return "host.docker.internal"
            
        return host

    @staticmethod
    def _is_external_host(host: str) -> bool:
        """Check if a host appears to be external (IP address or FQDN)."""
        import re

        ip_pattern = re.compile(r"^(\d{1,3}\.){3}\d{1,3}$")
        if ip_pattern.match(host):
            return True
        return bool("." in host and not host.startswith("localhost"))

    def validate_required_vars(
        self, required_vars: Optional[List[str]] = None
    ) -> List[str]:
        """Validate that required environment variables are set."""
        if required_vars is None:
            required_vars = ["POSTGRES_PASSWORD", "NEO4J_PASSWORD"]
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        if not self.postgres.password and "POSTGRES_PASSWORD" not in missing_vars:
            missing_vars.append("POSTGRES_PASSWORD (via config)")
        if not self.neo4j.password and "NEO4J_PASSWORD" not in missing_vars:
            missing_vars.append("NEO4J_PASSWORD (via config)")
        return list(set(missing_vars))

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
    """Load aclarai configuration with validation."""
    config = aclaraiConfig.from_env(env_file, config_file)
    if validate:
        missing_vars = config.validate_required_vars(required_vars)
        if missing_vars:
            error_msg = f"Missing required environment variables: {', '.join(sorted(missing_vars))}. Please check your .env file or environment configuration."
            logger.error(error_msg)
            raise ValueError(error_msg)
    config.setup_logging()
    return config
