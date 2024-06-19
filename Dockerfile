# Use a specific Python version
ARG PYTHON_VERSION=3.11.9
FROM python:${PYTHON_VERSION}-slim as base

# Set environment variables to prevent .pyc files and enable unbuffered mode
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory
WORKDIR /app

# Create a non-root user
ARG UID=10001
RUN adduser \
    --disabled-password \
    --gecos "" \
    --home "/nonexistent" \
    --shell "/sbin/nologin" \
    --no-create-home \
    --uid "${UID}" \
    appuser

# Install gunicorn
RUN pip install gunicorn

# Install necessary dependencies for building Python packages and Tesseract OCR
RUN apt-get update && \
    apt-get install -y tesseract-ocr gcc g++ python3-dev libgl1 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy the requirements.txt file into the image
COPY requirements.txt .

# Install Python dependencies
RUN python -m pip install -r requirements.txt

# Switch to non-root user
USER appuser

# Copy the application code
COPY core/ ./core/
COPY app.py .

# Expose port
ENV PORT 8080
EXPOSE 8080

# Use gunicorn to run the Flask application
CMD ["gunicorn", "-b", "0.0.0.0:8080", "app:app"]