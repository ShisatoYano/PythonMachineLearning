FROM python:3.10.4-buster

RUN apt -y update
RUN apt -y upgrade

RUN apt -y install git
RUN git config --global user.name Shisato
RUN git config --global user.email shisatoyano@gmail.com

RUN apt -y install libgl1-mesa-dev xorg-dev
RUN apt -y install x11-apps

RUN pip install --upgrade pip
RUN pip install setuptools --upgrade

ENV DISPLAY=host.docker.internal:0.0
ENV QT_X11_NO_MITSHM=1