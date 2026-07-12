FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY abliteratedllmbot ./abliteratedllmbot
COPY pyproject.toml README.md ./
RUN pip install --no-cache-dir -e .

# config.yaml and .env are bind-mounted at runtime, never baked in.
CMD ["python", "-m", "abliteratedllmbot", "/app/config.yaml"]
