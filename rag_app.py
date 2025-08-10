from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# 검색 모듈
class RetrievalModule:
    def __init__(self):
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.index = faiss.IndexFlatL2(384)  # all-MiniLM-L6-v2의 출력 차원
        self.documents = []

    def add_documents(self, docs: List[str]):
        """문서를 임베딩하여 Vector DB(FAISS)에 저장"""
        if not docs:
            return
        # 문서 임베딩 생성
        embeddings = self.embedder.encode(docs, show_progress_bar=True).astype(np.float32)
        # FAISS 인덱스에 추가
        self.index.add(embeddings)
        # 문서 저장
        self.documents.extend(docs)

    def retrieve(self, query: str, k: int = 3) -> List[str]:
        """쿼리로 Vector DB에서 유사한 문서 검색"""
        query_embedding = self.embedder.encode([query]).astype(np.float32)
        distances, indices = self.index.search(query_embedding, k)
        return [self.documents[idx] for idx in indices[0] if idx < len(self.documents)]

# 생성 모듈
class GenerationModule:
    def __init__(self, model_name: str = "gpt2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

    def generate(self, prompt: str, max_length: int = 512) -> str:
        """프롬프트로 답변 생성"""
        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
        outputs = self.model.generate(
            inputs.input_ids,
            max_length=max_length,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            num_return_sequences=1
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True).split("Answer:")[-1].strip()

# Modular RAG 시스템
class ModularRAG:
    def __init__(self):
        self.retriever = RetrievalModule()
        self.generator = GenerationModule()

    def add_knowledge_base(self, documents: List[str]):
        """문서를 Vector DB에 추가"""
        self.retriever.add_documents(documents)

    def answer(self, query: str) -> str:
        """쿼리에 대한 답변 생성"""
        # Vector DB에서 관련 문서 검색
        contexts = self.retriever.retrieve(query, k=3)
        context = "\n".join(contexts) if contexts else "관련 정보를 찾을 수 없습니다."
        # 프롬프트 생성
        prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
        # 답변 생성
        return self.generator.generate(prompt)

if __name__ == "__main__":
    rag = ModularRAG()

    # PDF 로드
    try:
        loader = PyPDFLoader("./SPRi AI Brief_8월호_산업동향_F.pdf")
        documents = loader.load()
    except FileNotFoundError:
        print("PDF 파일을 찾을 수 없습니다. 샘플 문서를 사용합니다.")
        documents = [
            "xAI는 Grok이라는 AI 모델을 공개했습니다. Grok은 인간의 질문에 유용하고 진실된 답변을 제공하도록 설계되었습니다.",
            "RAG는 검색과 생성을 결합한 자연어 처리 모델입니다.",
            "딥러닝은 인공신경망을 사용해 데이터를 학습하는 기계학습의 한 분야입니다."
        ]
        texts = documents
    else:
        # 청킹
        splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
        chunked_docs = splitter.split_documents(documents)
        texts = [doc.page_content for doc in chunked_docs]

    # Vector DB에 저장
    print("문서를 Vector DB에 저장 중...")
    rag.add_knowledge_base(texts)
    print(f"{len(texts)}개의 문서가 Vector DB에 저장되었습니다.")

    # 쿼리 조회 및 답변 생성
    query = "xAI에서 어떤 모델을 공개하였나요?"
    print(f"질문: {query}")
    answer = rag.answer(query)
    print(f"답변: {answer}")