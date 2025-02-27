FROM python:3.12-slim

WORKDIR /app
COPY . /app

RUN apt update -y && apt install awscli -y
RUN pip install torch pathlib python-box pyyaml Flask pillow

CMD [ "python", "application.py" ]



