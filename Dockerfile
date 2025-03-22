# Use Python 3.11 base image
FROM python:3.11

# Install system dependencies
RUN apt-get update && \
    apt-get install -y sqlite3 libsqlite3-dev

# Set working directory
WORKDIR /app

# Copy app files
COPY . .

# Install dependencies
RUN pip install -r requirements.txt

# Run the app
CMD ["streamlit", "run", "Streamlit_search.py"]
