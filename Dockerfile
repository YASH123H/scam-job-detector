# Use official Python base image
FROM python:3.10

# Set working directory in container
WORKDIR /app

# Copy everything from your repo into the container
COPY . .

# Upgrade pip and install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Run your Python app
CMD ["python", "app.py"]
