FROM python:3.11-slim

# Set workdir
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Upgrade pip and install dependencies
RUN pip install --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# Environment variables
# TELEGRAM_TOKEN, TWELVEDATA_API_KEY, NEWSAPI_KEY
ENV PYTHONUNBUFFERED=1

# Run bot
CMD ["python", "main.py"]
