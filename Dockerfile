# Use an official Python runtime as a parent image
FROM python:3.11-slim-buster

# Set the working directory in the container
WORKDIR /

# Add the current directory contents into the container at /
ADD . /

ENV GOOGLE_APPLICATION_CREDENTIALS=/service-account.json


# Install system libraries
RUN apt-get update && apt-get install -y \
    build-essential \
    libsndfile1 \
    ffmpeg \
    curl \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Check if ffmpeg and ffprobe installed
RUN which ffmpeg
RUN which ffprobe

# Install aubio dependencies
RUN pip install --upgrade --use-pep517 pip setuptools wheel  
# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir --use-pep517 -r requirements.txt

# Set the working directory for the downloads
WORKDIR /root/.cache/torch/hub/checkpoints/

# Unzip each model
RUN unzip /Models/unmix/bass-2ca1ce51.zip
RUN unzip /Models/unmix/drums-69e0ebd4.zip
RUN unzip /Models/unmix/other-c8c5b3e6.zip
RUN unzip /Models/unmix/vocals-bccbd9aa.zip

# Switch back to the app directory
WORKDIR /

# Make port 80 available to the world outside this container
EXPOSE 80

# Run app.py when the container launches
CMD ["python", "app.py"]
