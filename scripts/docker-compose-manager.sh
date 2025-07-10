#!/bin/bash

# Docker Compose Service Manager
# This script manages which services to start based on USE_EXTERNAL_* environment variables

set -e

# Load environment variables from .env file if it exists
if [ -f .env ]; then
    # Load environment variables safely
    while IFS='=' read -r key value; do
        if [[ "$key" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]]; then
            export "$key"="$value"
        fi
    done < <(grep -v '^#' .env | grep '=')
fi

# Default values if not set
USE_EXTERNAL_POSTGRES=${USE_EXTERNAL_POSTGRES:-false}
USE_EXTERNAL_NEO4J=${USE_EXTERNAL_NEO4J:-false}
USE_EXTERNAL_RABBITMQ=${USE_EXTERNAL_RABBITMQ:-false}

# Build the profiles array
PROFILES=("default")

# Add service profiles based on external service flags
if [ "$USE_EXTERNAL_POSTGRES" != "true" ]; then
    PROFILES+=("postgres")
fi

if [ "$USE_EXTERNAL_NEO4J" != "true" ]; then
    PROFILES+=("neo4j")
fi

if [ "$USE_EXTERNAL_RABBITMQ" != "true" ]; then
    PROFILES+=("rabbitmq")
fi

# Convert profiles array to comma-separated string
PROFILE_STRING=$(IFS=,; echo "${PROFILES[*]}")

echo "🐳 Starting Docker Compose with profiles: $PROFILE_STRING"
echo "   External services:"
echo "   - PostgreSQL: $([ "$USE_EXTERNAL_POSTGRES" = "true" ] && echo "External" || echo "Docker")"
echo "   - Neo4j: $([ "$USE_EXTERNAL_NEO4J" = "true" ] && echo "External" || echo "Docker")"
echo "   - RabbitMQ: $([ "$USE_EXTERNAL_RABBITMQ" = "true" ] && echo "External" || echo "Docker")"
echo ""

# Execute docker compose with the determined profiles
# Use multiple --profile flags instead of comma-separated
PROFILE_ARGS=()
for profile in "${PROFILES[@]}"; do
    PROFILE_ARGS+=("--profile" "$profile")
done

exec docker compose "${PROFILE_ARGS[@]}" "$@"
