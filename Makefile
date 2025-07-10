# Aclarai Docker Service Management Makefile

# Default target
.PHONY: help
help:
	@echo "Aclarai Docker Service Management"
	@echo ""
	@echo "Usage:"
	@echo "  make up                    # Start all services (respects .env configuration)"
	@echo "  make down                  # Stop all services"
	@echo "  make logs                  # Show logs from all services"
	@echo "  make build                 # Build all services"
	@echo "  make clean                 # Stop and remove all containers, networks, and volumes"
	@echo ""
	@echo "External Service Scenarios:"
	@echo "  make up-external-postgres  # Start with external PostgreSQL"
	@echo "  make up-external-neo4j     # Start with external Neo4j"
	@echo "  make up-external-all       # Start with all external services"
	@echo "  make up-docker-all         # Start with all Docker services (default)"
	@echo ""
	@echo "Development:"
	@echo "  make dev-logs              # Follow logs from all services"
	@echo "  make status                # Show service status"

# Start services using the manager script
.PHONY: up
up:
	@./scripts/docker-compose-manager.sh up -d

# Stop services
.PHONY: down
down:
	@./scripts/docker-compose-manager.sh down

# Show logs
.PHONY: logs
logs:
	@./scripts/docker-compose-manager.sh logs

# Follow logs
.PHONY: dev-logs
dev-logs:
	@./scripts/docker-compose-manager.sh logs -f

# Build services
.PHONY: build
build:
	@./scripts/docker-compose-manager.sh build

# Clean up everything
.PHONY: clean
clean:
	@./scripts/docker-compose-manager.sh down -v --remove-orphans

# Show service status
.PHONY: status
status:
	@./scripts/docker-compose-manager.sh ps

# External PostgreSQL scenario
.PHONY: up-external-postgres
up-external-postgres:
	@USE_EXTERNAL_POSTGRES=true ./scripts/docker-compose-manager.sh up -d

# External Neo4j scenario
.PHONY: up-external-neo4j
up-external-neo4j:
	@USE_EXTERNAL_NEO4J=true ./scripts/docker-compose-manager.sh up -d

# All external services scenario
.PHONY: up-external-all
up-external-all:
	@USE_EXTERNAL_POSTGRES=true USE_EXTERNAL_NEO4J=true USE_EXTERNAL_RABBITMQ=true ./scripts/docker-compose-manager.sh up -d

# All Docker services scenario (default)
.PHONY: up-docker-all
up-docker-all:
	@USE_EXTERNAL_POSTGRES=false USE_EXTERNAL_NEO4J=false USE_EXTERNAL_RABBITMQ=false ./scripts/docker-compose-manager.sh up -d
