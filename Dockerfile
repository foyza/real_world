# 1. Используем официальный Python
FROM python:3.11-slim

# 2. Устанавливаем зависимости для компиляции и работы pandas/numpy
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# 3. Устанавливаем рабочую директорию
WORKDIR /app

# 4. Копируем файлы зависимостей и устанавливаем их
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Копируем весь проект
COPY . .

# 6. Добавляем переменные окружения (чтобы токены были безопасны)
ENV PYTHONUNBUFFERED=1

# 7. Команда запуска
CMD ["python", "main.py"]
