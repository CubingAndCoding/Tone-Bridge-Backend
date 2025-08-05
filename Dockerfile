# ToneBridge Backend Dockerfile - Free Tier Optimized

FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies (minimal for free tier)
RUN apt-get update && apt-get install -y \
    gcc \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Copy lightweight requirements
COPY requirements-free-tier.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements-free-tier.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd -m -u 1000 tonebridge && \
    chown -R tonebridge:tonebridge /app
USER tonebridge

# Expose port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Run the application
CMD ["python", "run.py"] 