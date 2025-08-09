FROM ubuntu:22.04

# Python 설치
RUN apt-get update && apt-get install -y python3 python3-pip && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

