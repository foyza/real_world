# Используем официальный образ Python 3.11 slim
FROM python:3.11-slim

# Устанавливаем зависимости ОС
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Устанавливаем рабочую директорию
WORKDIR /app

# Копируем файлы проекта
COPY . .

# Устанавливаем Python зависимости
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Копируем .env отдельно для безопасности
# Пользователь должен сам положить .env в папку при сборке или монтировать volume

# Запуск бота
CMD ["python", "main.py"]
