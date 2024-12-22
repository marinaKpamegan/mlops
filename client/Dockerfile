FROM python:3.9-slim
# Installer Git
RUN apt-get update && apt-get install -y git

COPY requirements.txt /app/requirements.txt

WORKDIR /app 
RUN pip install -r requirements.txt

COPY app.py app.py
COPY pages pages

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
