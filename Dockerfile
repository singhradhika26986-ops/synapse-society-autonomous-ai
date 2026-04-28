FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV SYNAPSE_DISABLE_TRANSFORMER=1
ENV HOST=0.0.0.0
ENV PORT=8000

WORKDIR /app

COPY requirements-prod.txt .
RUN pip install --no-cache-dir -r requirements-prod.txt

COPY . .

EXPOSE 8000

CMD ["sh", "start.sh"]
