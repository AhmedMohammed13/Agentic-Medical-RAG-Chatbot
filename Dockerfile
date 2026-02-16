# ========================
# Stage 1 - Builder
# ========================
FROM python:3.10-slim AS builder

# Install build dependencies for compiling Python packages
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy only requirements first (better caching)
COPY requirements.txt .

# Install dependencies to the local user directory
RUN pip install --no-cache-dir --user -r requirements.txt

# ========================
# Stage 2 - Final Runtime Image
# ========================
FROM python:3.10-slim

# Install minimal runtime dependencies
RUN apt-get update && apt-get install -y \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user for security
RUN useradd --create-home --shell /bin/bash appuser

# Set working directory
WORKDIR /app

# Copy installed Python packages from builder
COPY --from=builder /root/.local /home/appuser/.local

# Copy application code (but not data)
COPY . .

# Ensure proper ownership
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Make sure local user-installed packages are on PATH
ENV PATH=/home/appuser/.local/bin:$PATH

# Expose port for Gradio
EXPOSE 7860

# Default command
CMD ["python", "app.py"]
