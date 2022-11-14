FROM ubuntu:focal

RUN apt-get update && apt-get upgrade -y

RUN apt-get install nano
RUN apt-get install -y git
RUN apt-get install -y python3.8 python3-pip
 
RUN pip install scipy
RUN pip install numpy
RUN pip install matplotlib
RUN pip install opencv-python

RUN apt-get update && apt-get upgrade -y

WORKDIR /work

