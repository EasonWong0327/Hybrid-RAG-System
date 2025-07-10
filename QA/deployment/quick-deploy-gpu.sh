#!/bin/bash

# GPU Version Quick Deploy Script for Linux
# =========================================

echo "Starting Hybrid RAG System (GPU Version)"
echo "Checking Docker and NVIDIA Docker support..."

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "Docker is not installed! Please install Docker first"
    exit 1
fi

# Check NVIDIA drivers
echo "Checking NVIDIA drivers..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi > /dev/null 2>&1
    if [ $? -eq 0 ]; then
        echo "NVIDIA drivers are working"
    else
        echo "NVIDIA drivers are not working properly!"
        echo "Please ensure NVIDIA drivers are installed correctly"
        exit 1
    fi
else
    echo "NVIDIA drivers are not available!"
    echo "Please ensure NVIDIA drivers are installed"
    exit 1
fi

# Check Docker GPU support
echo "Checking Docker GPU support..."
if docker info 2>&1 | grep -q "nvidia"; then
    echo "Docker GPU support detected"
else
    echo "Docker GPU support not detected"
    echo "Continuing with deployment - GPU support will be tested during container startup"
fi

# Stop existing containers
echo "Stopping existing containers..."
docker-compose -f docker-compose.gpu.yml down

# Build and start
echo "Building and starting services..."
docker-compose -f docker-compose.gpu.yml up --build -d

# Wait for services to start
echo "Waiting for services to start..."
sleep 30

# Check service status
echo "Checking service status..."
docker-compose -f docker-compose.gpu.yml ps

echo "Deployment completed!"
echo "Access URL: http://localhost:8000"
echo "View logs: docker-compose -f docker-compose.gpu.yml logs -f" 