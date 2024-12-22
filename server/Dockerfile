FROM python:3.9-slim

# Installer Git
RUN apt-get update && apt-get install -y git
COPY requirements.txt /app/requirements.txt

WORKDIR /app 
RUN pip install -r requirements.txt

COPY app.py app.py
COPY images images
COPY train.py train.py

CMD  ["uvicorn", "--host", "0.0.0.0", "app:app"]