#!/usr/bin/env bash

set -e  # Exit on any error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Create or update .env file
if [ ! -f .env ]; then
    echo -e "${YELLOW}Creating .env file from .env.example...${NC}"
    cp .env.example .env
fi

# Generate secure random passwords if not already set
if grep -q "your_postgres_password_here" .env; then
    POSTGRES_PASSWORD=$(openssl rand -base64 32 | tr -dc 'a-zA-Z0-9' | head -c 32)
    sed -i '' "s/your_postgres_password_here/${POSTGRES_PASSWORD}/g" .env
fi

if grep -q "your_neo4j_password_here" .env; then
    NEO4J_PASSWORD=$(openssl rand -base64 32 | tr -dc 'a-zA-Z0-9' | head -c 32)
    sed -i '' "s/your_neo4j_password_here/${NEO4J_PASSWORD}/g" .env
fi

if grep -q "your_rabbitmq_password_here" .env; then
    RABBITMQ_PASSWORD=$(openssl rand -base64 32 | tr -dc 'a-zA-Z0-9' | head -c 32)
    sed -i '' "s/your_rabbitmq_password_here/${RABBITMQ_PASSWORD}/g" .env
fi

# Set local development configuration
sed -i '' "s/POSTGRES_HOST=postgres/POSTGRES_HOST=localhost/g" .env
sed -i '' "s/NEO4J_HOST=neo4j/NEO4J_HOST=localhost/g" .env
sed -i '' "s/RABBITMQ_HOST=rabbitmq/RABBITMQ_HOST=localhost/g" .env

# Set vault path to local directory
sed -i '' "s|VAULT_PATH=/vault|VAULT_PATH=$(pwd)/vault|g" .env

# Load variables from .env file
set -a
source .env
set +a

# Generate and export connection URLs
export POSTGRES_URL="postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@localhost:5432/${POSTGRES_DB}"
export NEO4J_URL="bolt://localhost:7687"

# Ensure URLs are in .env file
if ! grep -q "^POSTGRES_URL=" .env; then
    echo "" >> .env
    echo "# Integration test configuration" >> .env
    echo "POSTGRES_URL=${POSTGRES_URL}" >> .env
    echo "NEO4J_URL=${NEO4J_URL}" >> .env
fi

# Show what was configured
echo -e "${GREEN}Environment file updated with secure configuration${NC}"

# PostgreSQL connection parameters
PG_HOST=${POSTGRES_HOST:-localhost}
PG_PORT=${POSTGRES_PORT:-5432}
PG_USER=${POSTGRES_USER:-aclarai}
PG_PASSWORD=${POSTGRES_PASSWORD}
PG_DB=${POSTGRES_DB:-aclarai}

# Neo4j connection parameters
NEO4J_HOST=${NEO4J_HOST:-localhost}
NEO4J_HTTP_PORT=${NEO4J_HTTP_PORT:-7474}
NEO4J_BOLT_PORT=${NEO4J_BOLT_PORT:-7687}

# Function to check if a service is ready
check_service() {
    local service=$1
    local check_command=$2
    local max_attempts=30
    local wait_seconds=2

    echo -e "${YELLOW}Checking $service availability...${NC}"
    
    for ((i=1; i<=max_attempts; i++)); do
        if eval "$check_command"; then
            echo -e "${GREEN}$service is ready!${NC}"
            return 0
        fi
        echo -n "."
        sleep $wait_seconds
    done
    
    echo -e "\n${RED}$service is not available after $((max_attempts * wait_seconds)) seconds${NC}"
    return 1
}

# Check if PostgreSQL is running
check_postgres() {
    check_service "PostgreSQL" "PGPASSWORD=$PG_PASSWORD pg_isready -h $PG_HOST -p $PG_PORT -U $PG_USER"
}

# Check if Neo4j is running
check_neo4j() {
    # Check both HTTP and Bolt ports
    check_service "Neo4j HTTP" "curl -s http://$NEO4J_HOST:$NEO4J_HTTP_PORT/ > /dev/null" && 
    check_service "Neo4j Bolt" "nc -z $NEO4J_HOST $NEO4J_BOLT_PORT"
}

# Check if RabbitMQ is running
check_rabbitmq() {
    # Check both AMQP and management ports
    check_service "RabbitMQ AMQP" "nc -z $RABBITMQ_HOST $RABBITMQ_PORT" &&
    check_service "RabbitMQ Management" "nc -z $RABBITMQ_HOST $RABBITMQ_MANAGEMENT_PORT"
}

# Setup test databases using docker exec
setup_test_db() {
    echo -e "${YELLOW}Setting up test databases in Docker container...${NC}"
    
    # Get the PostgreSQL container ID
    PG_CONTAINER=$(docker ps --filter "name=postgres" --format "{{.ID}}")
    if [ -z "$PG_CONTAINER" ]; then
        echo -e "${RED}PostgreSQL container not found. Make sure it's running.${NC}"
        exit 1
    fi

    # Copy the setup script into the container
    docker cp scripts/setup_test_db.sql $PG_CONTAINER:/tmp/setup_test_db.sql
    
    # Execute the setup script inside the container
    if docker exec $PG_CONTAINER psql -U $POSTGRES_USER -d $POSTGRES_DB -f /tmp/setup_test_db.sql; then
        echo -e "${GREEN}Test databases setup successfully${NC}"
    else
        echo -e "${RED}Failed to setup test databases${NC}"
        exit 1
    fi

    # Clean up
    docker exec $PG_CONTAINER rm /tmp/setup_test_db.sql
}

# Main execution
main() {
    # Check all required services
    echo -e "${YELLOW}Checking required services...${NC}"
    
    check_postgres || {
        echo -e "${RED}PostgreSQL is not running. Start it with: brew services start postgresql${NC}"
        exit 1
    }
    
    check_neo4j || {
        echo -e "${RED}Neo4j is not running. Start it with: brew services start neo4j${NC}"
        exit 1
    }
    
    check_rabbitmq || {
        echo -e "${RED}RabbitMQ is not running. Start it with: brew services start rabbitmq${NC}"
        exit 1
    }

    # Setup test environment
    setup_test_db

echo -e "\n${GREEN}Environment setup complete!${NC}"

echo -e "\n${YELLOW}Service Status:${NC}"
echo -e "PostgreSQL:  ${GREEN}Running on localhost:5432${NC}"
echo -e "Neo4j:      ${GREEN}Running on localhost:7474/7687${NC}"
echo -e "RabbitMQ:   ${GREEN}Running on localhost:5672/15672${NC}"

echo -e "\n${YELLOW}Environment Configuration:${NC}"
echo -e "Database:   ${GREEN}${POSTGRES_URL}${NC}"
echo -e "Graph:      ${GREEN}${NEO4J_URL}${NC}"
echo -e "Vault:      ${GREEN}${VAULT_PATH}${NC}"

echo -e "\n${YELLOW}All services are running and environment is configured.${NC}"

echo -e "\n${YELLOW}To run integration tests, use:${NC}"
echo -e "${GREEN}pytest -v -m integration${NC}             # Run all integration tests"
echo -e "${GREEN}pytest -v path/to/test_file.py -m integration${NC}  # Run specific test file"
}

# Execute main function
main
