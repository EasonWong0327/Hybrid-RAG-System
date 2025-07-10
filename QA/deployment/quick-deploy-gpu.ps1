# GPU Version Quick Deploy Script
# ===============================

Write-Host "Starting Hybrid RAG System (GPU Version)" -ForegroundColor Green
Write-Host "Checking Docker and NVIDIA Docker support..."

# Check if Docker is installed
if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    Write-Host "Docker is not installed! Please install Docker Desktop first" -ForegroundColor Red
    exit 1
}

# Check NVIDIA drivers
Write-Host "Checking NVIDIA drivers..."
try {
    nvidia-smi | Out-Null
    Write-Host "NVIDIA drivers are working" -ForegroundColor Green
} catch {
    Write-Host "NVIDIA drivers are not available!" -ForegroundColor Red
    Write-Host "Please ensure NVIDIA drivers are installed" -ForegroundColor Yellow
    exit 1
}

# Check Docker GPU support (simplified)
Write-Host "Checking Docker GPU support..."
$dockerInfo = docker info 2>&1
if ($dockerInfo -match "nvidia") {
    Write-Host "Docker GPU support detected" -ForegroundColor Green
} else {
    Write-Host "Docker GPU support not detected" -ForegroundColor Yellow
    Write-Host "Continuing with deployment - GPU support will be tested during container startup" -ForegroundColor Yellow
}

# Stop existing containers
Write-Host "Stopping existing containers..."
docker-compose -f docker-compose.gpu.yml down

# Build and start
Write-Host "Building and starting services..."
docker-compose -f docker-compose.gpu.yml up --build -d

# Wait for services to start
Write-Host "Waiting for services to start..."
Start-Sleep -Seconds 30

# Check service status
Write-Host "Checking service status..."
docker-compose -f docker-compose.gpu.yml ps

Write-Host "Deployment completed!" -ForegroundColor Green
Write-Host "Access URL: http://localhost:8000" -ForegroundColor Cyan
Write-Host "View logs: docker-compose -f docker-compose.gpu.yml logs -f" -ForegroundColor Cyan 