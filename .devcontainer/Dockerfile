FROM continuumio/miniconda3

ARG DEBIAN_FRONTEND=noninteractive

RUN apt update \
    && apt install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*
RUN apt update \
    && apt upgrade -y

RUN pip install -U --no-cache-dir \
    cython \
    networkx \
    numpy \
    pandas  \
    tsplib95 \
    jupyter notebook \
    matplotlib \
    scipy

