from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# 간단한 RAG 예시
loader = PyPDFLoader("sample.pdf")
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(documents)
embedder = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = embedder.encode([chunk.page_content for chunk in chunks])

# FAISS 인덱스 생성
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

print("RAG 시스템 준비 완료!")