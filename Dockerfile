FROM python:3.8-slim-buster

RUN apt-get update

COPY . /CancerNER

COPY ./requirements.txt /CancerNER/requirements.txt

WORKDIR CancerNER

EXPOSE 8000:8000

RUN python3 -m pip install -r requirements.txt

RUN python3 -m nltk.downloader averaged_perceptron_tagger

CMD [ "uvicorn", "main:app", "--host", "0.0.0.0", "--reload" ]
