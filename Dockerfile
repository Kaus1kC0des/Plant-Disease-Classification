FROM python:3.10-slim

COPY . /app
WORKDIR /app/Flask
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE $PORT
CMD gunicorn --workers=4 --bind 0.0.0.0:$PORT app:app

LABEL authors="kausik"