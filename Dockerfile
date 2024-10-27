FROM python:3.10

# Set the working directory to /app
WORKDIR /app

# Copy the necessary directories to the container
COPY Assets /app/Assets
COPY Flask /app/Flask
COPY Src /app/Src
COPY TestImages /app/TestImages
COPY requirements.txt /app/Flask/requirements.txt

# Change the working directory to /app/Flask
WORKDIR /app/Flask

# Install the dependencies
RUN pip install -U pip
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install torch==2.2.1 torchvision==0.17.1 --index-url https://download.pytorch.org/whl/cpu

# Expose the port
EXPOSE $PORT

# Command to run the application
CMD gunicorn --workers=4 --bind 0.0.0.0:$PORT app:app

# Label the image
LABEL authors="kausik"