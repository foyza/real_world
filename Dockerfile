FROM python:3.11-slim

# Системные зависимости для ML
RUN apt-get update && apt-get install -y build-essential libgomp1 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "main.py"]
