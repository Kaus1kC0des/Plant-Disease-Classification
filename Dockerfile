FROM python:3.10-slim

COPY . /app
WORKDIR /app/Flask

RUN pip install --no-cache-dir -r requirements.txt
RUN pip install torch==2.2.1 torchvision==0.17.1 --index-url https://download.pytorch.org/whl/cpu

EXPOSE $PORT
CMD gunicorn --workers=4 --bind 0.0.0.0:$PORT app:app

LABEL authors="kausik"