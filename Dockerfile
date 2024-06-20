<<<<<<< HEAD
# use the official python 3.11.9 version iamge
FROM python:3.11.9

# set the working directory of the container
WORKDIR /app

# install tesseract OCR engine
RUN apt-get update && apt-get install -y tesseract-ocr
# install the indonesian language
RUN apt install -y tesseract-ocr-ind
# install the english language
RUN apt install -y tesseract-ocr-eng
# install OpenCV2 dependencies
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

# copy requirements.txt and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -U -r requirements.txt

# copy application's code and core module
COPY core/ ./core/  
COPY app.py . 

# expose port 5000
EXPOSE 5000

# run the app
CMD python -m app
=======
# syntax=docker/dockerfile:1

ARG PYTHON_VERSION=3.11.9
FROM python:${PYTHON_VERSION}-slim as base

# Prevents Python from writing pyc files.
ENV PYTHONDONTWRITEBYTECODE=1

# Keeps Python from buffering stdout and stderr to avoid situations where
# the application crashes without emitting any logs due to buffering.
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Create a non-privileged user that the app will run under.
# See https://docs.docker.com/go/dockerfile-user-best-practices/
ARG UID=10001
RUN adduser \
    --disabled-password \
    --gecos "" \
    --home "/nonexistent" \
    --shell "/sbin/nologin" \
    --no-create-home \
    --uid "${UID}" \
    appuser

# Install tesseract OCR engine,
# gcc, g++, and python3-dev for building certain Python packages,
# and dependencies for opencv-python (cv2).
RUN apt-get update && \
    apt-get install -y tesseract-ocr gcc g++ python3-dev libgl1 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy the requirements.txt file into the image
COPY requirements.txt .

# Install Python dependencies
RUN python -m pip install -r requirements.txt

# Switch to the non-privileged user to run the application.
USER appuser

# Copy the application code
COPY core/ ./core/
COPY app.py .
COPY models/ ./models/

# Expose the port Flask is running on
ENV PORT 8080
EXPOSE 8080

# Define the command to run the application
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "app:app"]
>>>>>>> master
