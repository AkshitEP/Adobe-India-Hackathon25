FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY main.py .
COPY models/all-MiniLM-L6-v2 /app/models/all-MiniLM-L6-v2
CMD ["python", "main.py"]