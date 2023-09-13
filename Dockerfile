FROM python:latest

WORKDIR /app/

COPY ./main.py /app/
COPY ./classifier.py /app/
COPY ./requirements.txt /app/
COPY ./data /app/data
COPY ./uploaded_images /app/uploaded_images
#COPY . /app/

RUN pip install -r requirements.txt

EXPOSE 4282

CMD uvicorn --host=0.0.0.0 --port 4282 main:app