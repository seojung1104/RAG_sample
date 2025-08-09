# RAG_sample
RAG sample 코드입니다.


# apple silicon
docker build --platform linux/amd64 -t junghoonseo367/rag-app:latest .

# else
docker build -t junghoonseo367/rag-app:latest .

# docker run
docker run -it -d -p 8080:8080 -v /:/mnt/local  junghoonseo367/rag-app /bin/bash 