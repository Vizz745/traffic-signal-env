FROM python:3.11

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --timeout 100 -r requirements.txt

COPY . .

EXPOSE 7860

CMD ["python", "-m", "uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]