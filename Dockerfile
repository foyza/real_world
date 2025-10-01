# Используем официальный образ Python
FROM python:3.11-slim

# Устанавливаем зависимости для работы pip и системы
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Устанавливаем рабочую директорию
WORKDIR /app

# Копируем зависимости и ставим их
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем весь код
COPY . .

# Экспорт переменных окружения (будут браться из .env при деплое)
ENV PYTHONUNBUFFERED=1

# Запуск бота
CMD ["python", "main.py"]

