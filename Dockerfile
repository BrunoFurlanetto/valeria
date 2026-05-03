FROM python:3.10-slim
LABEL manteiner='bruno.furlanetto@hotmail.com'

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        portaudio19-dev \
        libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY app ./app
RUN pip install -r app/requirements.txt
CMD ["python", "-m", "app.assistant", "text"]
