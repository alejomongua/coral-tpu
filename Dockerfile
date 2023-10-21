FROM python:3.9.18-bullseye

RUN echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | tee /etc/apt/sources.list.d/coral-edgetpu.list
RUN curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add -
RUN apt-get update
RUN pip install --upgrade pip
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

RUN apt-get -y install libedgetpu1-std python3-pycoral edgetpu-compiler

RUN mkdir /workspace

WORKDIR /workspace

