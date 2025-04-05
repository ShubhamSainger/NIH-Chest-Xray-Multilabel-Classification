FROM python:3.9-slim

WORKDIR /app

COPY ./requirements.txt /app/requirements.txt

RUN pip install -r /app/requirements.txt --no-cache-dir

COPY ./fast_api /app

CMD ["python3","training_pipeline.py"]

