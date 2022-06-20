FROM python:3.8-slim-buster

RUN apt-get update

COPY . /nlp-for-healthcare-api

# COPY ./requirements.txt /nlp-for-healthcare-api/requirements.txt

WORKDIR nlp-for-healthcare-api

EXPOSE 8000:8000

RUN python3 -m pip install -r requirements.txt

RUN python3 -m nltk.downloader averaged_perceptron_tagger

CMD [ "uvicorn", "main:app", "--host", "0.0.0.0", "--reload" ]
