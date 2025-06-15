# Use official Python base image
FROM python:3.10-slim


# Set working directory in container
WORKDIR /app

# Copy everything from your repo into the container
COPY . .

# Upgrade pip and install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expose port 8000 (Railway default or typical for Streamlit)
EXPOSE 8000

# Start Streamlit app with appropriate network settings
CMD ["streamlit", "run", "app.py", "--server.port=8000", "--server.address=0.0.0.0"]
