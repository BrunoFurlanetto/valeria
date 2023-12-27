FROM python:3.12.1-alpine3.18
LABEL manteiner='bruno.furlanetto@hotmail.com'
RUN apk add build-base
COPY app /app
RUN pip install -r /app/requirements.txt
CMD ["python", "/app/main.py"]