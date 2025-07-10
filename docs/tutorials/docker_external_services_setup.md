# Docker and External Services Setup Tutorial

This tutorial walks you through configuring Aclarai to work with either Docker services or external services (PostgreSQL, Neo4j, RabbitMQ) depending on your development and deployment needs.

## What You'll Learn

- How to configure Aclarai for different deployment scenarios
- How to use Docker Compose profiles for flexible service management
- How to connect to external databases and message brokers
- How to troubleshoot common configuration issues
- How to use the new service management tools

## Prerequisites

Before starting this tutorial, make sure you have:

- Docker and Docker Compose installed
- Basic understanding of environment variables
- Access to external services (if using external configuration)

## Understanding the Configuration System

Aclarai supports three deployment scenarios:

1. **Full Docker Stack** - All services run in Docker containers (default)
2. **External Services** - Connect to existing external services  
3. **Mixed Environment** - Some services in Docker, others external

### Configuration Flags

The system uses two types of configuration flags:

**Application-Level Configuration (`prefer_docker_services`):**
```yaml
# settings/aclarai.config.yaml
service_discovery:
  prefer_docker_services: true  # Default: prefer Docker service names
```

**Environment-Level Configuration:**
```bash
# .env file
USE_EXTERNAL_POSTGRES=false
USE_EXTERNAL_NEO4J=false  
USE_EXTERNAL_RABBITMQ=false
```

## Scenario 1: Full Docker Stack (Default)

This is the easiest setup for development. All services run in Docker containers.

### Configuration

Create your `.env` file from the example:

```bash
cp .env.example .env
```

Edit the `.env` file with secure passwords:

```bash
# Database passwords (required)
POSTGRES_PASSWORD=your_secure_postgres_password
NEO4J_PASSWORD=your_secure_neo4j_password
RABBITMQ_PASSWORD=your_secure_rabbitmq_password

# Service control (defaults shown)
USE_EXTERNAL_POSTGRES=false
USE_EXTERNAL_NEO4J=false
USE_EXTERNAL_RABBITMQ=false

# Service hosts (defaults work for Docker)
POSTGRES_HOST=postgres
NEO4J_HOST=neo4j
RABBITMQ_HOST=rabbitmq
```

### Starting Services

Using the new service manager:

```bash
# Start all services
make up

# Or using the script directly
./scripts/docker-compose-manager.sh up -d

# View status
make status
```

The system will output:
```
🐳 Starting Docker Compose with profiles: default,postgres,neo4j,rabbitmq
   External services:
   - PostgreSQL: Docker
   - Neo4j: Docker
   - RabbitMQ: Docker
```

### Verifying the Setup

```bash
# Check service logs
make logs

# Follow logs in real-time
make dev-logs

# Check that databases are accessible
docker compose exec postgres psql -U aclarai -d aclarai -c "SELECT 1;"
docker compose exec neo4j cypher-shell -u neo4j -p your_neo4j_password "RETURN 1;"
```

## Scenario 2: External Services Only

Connect to external services running outside Docker (e.g., managed cloud databases).

### Configuration

Update your `.env` file:

```bash
# Service control - use external services
USE_EXTERNAL_POSTGRES=true
USE_EXTERNAL_NEO4J=true
USE_EXTERNAL_RABBITMQ=true

# External PostgreSQL configuration
POSTGRES_HOST=my-postgres-server.example.com
POSTGRES_PORT=5432
POSTGRES_USER=aclarai_user
POSTGRES_PASSWORD=secure_password
POSTGRES_DB=aclarai

# External Neo4j configuration  
NEO4J_HOST=my-neo4j-server.example.com
NEO4J_BOLT_PORT=7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=secure_password

# External RabbitMQ configuration
RABBITMQ_HOST=my-rabbitmq-server.example.com
RABBITMQ_PORT=5672
RABBITMQ_USER=aclarai_user
RABBITMQ_PASSWORD=secure_password
```

Update your application configuration:

```yaml
# settings/aclarai.config.yaml
service_discovery:
  prefer_docker_services: false  # Respect external hosts
```

### Starting Services

```bash
# Start with all external services
make up-external-all

# The output will show:
# 🐳 Starting Docker Compose with profiles: default
#    External services:
#    - PostgreSQL: External
#    - Neo4j: External
#    - RabbitMQ: External
```

Only the application services (aclarai-core, vault-watcher, etc.) will start in Docker.

### External Service Requirements

**PostgreSQL:**
- Must have `pgvector` extension installed
- User must have permissions to create databases and extensions
- Recommended: PostgreSQL 14+ with pgvector 0.4.0+

**Neo4j:**
- Neo4j 5.x required
- BOLT protocol enabled (port 7687)
- Authentication configured

**RabbitMQ:**
- RabbitMQ 3.x
- User with permissions to create queues and exchanges

## Scenario 3: Mixed Environment

Use some Docker services and some external services. This is useful when you have access to some external services but want to run others locally.

### Example: External PostgreSQL, Docker Neo4j and RabbitMQ

Update your `.env` file:

```bash
# Mixed configuration
USE_EXTERNAL_POSTGRES=true
USE_EXTERNAL_NEO4J=false
USE_EXTERNAL_RABBITMQ=false

# External PostgreSQL
POSTGRES_HOST=external-postgres.example.com
POSTGRES_USER=aclarai_user
POSTGRES_PASSWORD=secure_password

# Docker services use defaults
NEO4J_HOST=neo4j
NEO4J_PASSWORD=your_neo4j_password
RABBITMQ_HOST=rabbitmq
RABBITMQ_PASSWORD=your_rabbitmq_password
```

### Starting Mixed Services

```bash
# Start with external PostgreSQL only
make up-external-postgres

# Or configure manually
USE_EXTERNAL_POSTGRES=true make up
```

## Advanced Configuration

### Using External Services on the Same Host

When your external services run on the same machine as Docker:

```bash
# .env configuration
POSTGRES_HOST=host.docker.internal
NEO4J_HOST=host.docker.internal
RABBITMQ_HOST=host.docker.internal
```

The application automatically handles this mapping when running inside containers.

### Testing Different Configurations

Switch between configurations without rebuilding:

```bash
# Test external PostgreSQL
make down
make up-external-postgres

# Test external Neo4j  
make down
make up-external-neo4j

# Test all external
make down
make up-external-all

# Back to full Docker
make down
make up-docker-all
```

## Development Workflows

### Quick Development Setup

```bash
# Clone and setup
git clone <repo-url>
cd aclarai
cp .env.example .env

# Edit .env with secure passwords
nano .env

# Start everything
make up

# Verify services are running
make status
```

### External Database Development

```bash
# Configure for external databases
echo "USE_EXTERNAL_POSTGRES=true" >> .env
echo "USE_EXTERNAL_NEO4J=true" >> .env
echo "POSTGRES_HOST=my-external-db.com" >> .env

# Create config override
mkdir -p settings
cat > settings/aclarai.config.yaml << EOF
service_discovery:
  prefer_docker_services: false
EOF

# Start with external services
make up
```

### Debugging Connection Issues

```bash
# Check what services are actually running
make status

# View logs for connection issues
make logs | grep -i "connection\|error"

# Test database connectivity from container
docker compose exec aclarai-core python -c "
from aclarai_shared.config import load_config
config = load_config()
print(f'Postgres: {config.postgres.host}:{config.postgres.port}')
print(f'Neo4j: {config.neo4j.host}:{config.neo4j.port}')
"
```

## Troubleshooting

### Common Issues and Solutions

**Issue: Cannot connect to external service from Docker container**
```bash
# Solution: Verify hostname resolution
docker compose exec aclarai-core nslookup your-external-host.com

# Or use host.docker.internal for same-machine services
POSTGRES_HOST=host.docker.internal
```

**Issue: Application uses Docker service name when external expected**
```yaml
# Solution: Set prefer_docker_services to false
# settings/aclarai.config.yaml
service_discovery:
  prefer_docker_services: false
```

**Issue: Service dependency errors**
```bash
# Check what profiles are active
./scripts/docker-compose-manager.sh config

# Verify environment variables
grep USE_EXTERNAL .env
```

### Validation Script

Create a validation script to test your configuration:

```python
#!/usr/bin/env python3
# validate_config.py

from aclarai_shared.config import load_config
import os

def validate_configuration():
    print("🔍 Validating Aclarai Configuration...")
    
    config = load_config()
    
    print(f"Service Discovery: prefer_docker_services = {config.service_discovery.prefer_docker_services}")
    print(f"PostgreSQL: {config.postgres.host}:{config.postgres.port}")
    print(f"Neo4j: {config.neo4j.host}:{config.neo4j.port}")
    print(f"RabbitMQ: {config.rabbitmq_host}:{config.rabbitmq_port}")
    
    # Check external service flags
    external_flags = {
        'POSTGRES': os.getenv('USE_EXTERNAL_POSTGRES', 'false'),
        'NEO4J': os.getenv('USE_EXTERNAL_NEO4J', 'false'),
        'RABBITMQ': os.getenv('USE_EXTERNAL_RABBITMQ', 'false')
    }
    
    print("\nExternal service flags:")
    for service, flag in external_flags.items():
        print(f"  {service}: {flag}")

if __name__ == "__main__":
    validate_configuration()
```

Run the validation:
```bash
python validate_config.py
```

## Make Commands Reference

The new Makefile provides convenient commands for service management:

```bash
# Basic operations
make up                      # Start with .env configuration
make down                    # Stop all services
make logs                    # Show service logs
make status                  # Show service status
make clean                   # Clean up everything

# External service scenarios
make up-external-postgres    # External PostgreSQL only
make up-external-neo4j       # External Neo4j only  
make up-external-all         # All external services
make up-docker-all           # All Docker services (default)

# Development
make dev-logs                # Follow logs in real-time
make build                   # Build all services
make help                    # Show all available commands
```

## Environment Variable Reference

### Service Control Variables
- `USE_EXTERNAL_POSTGRES` - Skip PostgreSQL container (true/false)
- `USE_EXTERNAL_NEO4J` - Skip Neo4j container (true/false)  
- `USE_EXTERNAL_RABBITMQ` - Skip RabbitMQ container (true/false)

### Service Connection Variables
All connection variables support both Docker service names and external hostnames:

```bash
# PostgreSQL
POSTGRES_HOST=postgres              # or external hostname
POSTGRES_PORT=5432
POSTGRES_USER=aclarai
POSTGRES_PASSWORD=secure_password
POSTGRES_DB=aclarai

# Neo4j
NEO4J_HOST=neo4j                   # or external hostname
NEO4J_BOLT_PORT=7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=secure_password

# RabbitMQ
RABBITMQ_HOST=rabbitmq             # or external hostname
RABBITMQ_PORT=5672
RABBITMQ_USER=user
RABBITMQ_PASSWORD=secure_password
```

This flexible configuration system allows Aclarai to adapt to various deployment scenarios while maintaining consistent behavior across development, testing, and production environments.
