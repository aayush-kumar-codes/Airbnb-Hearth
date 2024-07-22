FROM python:3.11-slim

RUN apt-get update

RUN echo "fs.inotify.max_user_watches=524288" >> /etc/sysctl.conf

RUN mkdir /airbnb

WORKDIR /airbnb

RUN chmod -R 777 /airbnb

ENV AWS_DEFAULT_REGION=us-east-1

COPY requirements.txt .

RUN pip3 install -r requirements.txt

EXPOSE 8080

COPY . .

RUN sh download.sh

ENTRYPOINT ["streamlit", "run", "main.py", "--server.port=8080", "--server.address=0.0.0.0"]