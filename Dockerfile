FROM python:3.8

ENV PYTHONBUFFERED = 1
WORKDIR /app
COPY requirements.txt requirements.txt
COPY app/ /app
ADD Procfile /app

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y


EXPOSE 5000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5000"]
