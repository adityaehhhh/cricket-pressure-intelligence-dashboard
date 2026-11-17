# Official lightweight Python base image
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=7860

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files and pre-computed datasets
COPY . .

# Hugging Face Spaces default port
EXPOSE 7860

# Production WSGI server command
CMD ["gunicorn", "app:server", "--bind", "0.0.0.0:7860", "--workers", "2", "--threads", "2", "--timeout", "120"]
