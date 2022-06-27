FROM python:3.8-slim-buster

RUN apt-get update

COPY . /nlp-for-healthcare-api

# COPY ./requirements.txt /nlp-for-healthcare-api/requirements.txt

WORKDIR nlp-for-healthcare-api

EXPOSE 8000:8000

RUN python3 -m pip install -r requirements.txt

RUN [ "python3", "-c", "import nltk; nltk.download('averaged_perceptron_tagger', download_dir='/usr/local/nltk_data')" ]

CMD [ "python", "main.py"]
# CMD [ "uvicorn", "main:app", "--host", "0.0.0.0", "--reload" ]
