FROM python:3.12-slim
WORKDIR /app
COPY . /app

RUN apt update -y && apt install -y

RUN pip install -r requirements.txt
CMD [ "python3","app.py" ]