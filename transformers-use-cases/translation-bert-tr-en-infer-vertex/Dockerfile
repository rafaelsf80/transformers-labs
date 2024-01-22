#FROM pytorch/torchserve:latest-cpu
FROM tiangolo/uvicorn-gunicorn-fastapi:python3.8


# install dependencies
RUN python3 -m pip install --upgrade pip
RUN pip3 install transformers sentencepiece torch

COPY ./app /app
#COPY requirements.txt requirements.txt
COPY ./model-output-tr-en/ /model-output-tr-en/

#RUN pip install -r requirements.txt
EXPOSE 8080

# Start the app
CMD ["gunicorn", "-b", "0.0.0.0:8080", "main:app","--workers","1","--timeout","180","-k","uvicorn.workers.UvicornWorker"]