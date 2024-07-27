FROM python:3.11
COPY . .
WORKDIR /

RUN pip install --no-cache-dir --upgrade -r /requirements.txt

ENV KAFKA_URL=$KAFKA_URL
ENV KAFKA_API_KEY=$KAFKA_API_KEY
ENV KAFKA_IP=$KAFKA_IP

RUN echo "KAFKA_URL=$KAFKA_URL"

ENV TRANSFORMERS_CACHE=/transformers_cache
RUN mkdir -p  /transformers_cache && chmod -R 777  /transformers_cache

CMD ["python", "main.py"]