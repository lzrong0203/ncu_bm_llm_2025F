#!/usr/bin/env python3
"""
Week 4 - 簡化版 RAG 系統
基於 Week 3 的程式碼，加入實際的檢索和問答功能
"""

import re
import ollama
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


class Document:
    """文件類別，用於儲存文字內容和相關資訊"""

    def __init__(self, content: str, metadata: Dict[str, Any] = None):
        self.content = content
        self.metadata = metadata or {}


class SimpleRAG:
    """簡單的 RAG 系統"""

    def __init__(self, model_name: str = "gemma3:1b"):
        """初始化 RAG 系統"""
        self.llm_model = model_name
        self.embedding_model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2",
            device='cpu'
        )
        self.documents = []
        self.embeddings = None

    def load_pdf(self, pdf_path: str) -> str:
        """讀取單個 PDF 檔案"""
        text = ""
        with open(pdf_path, 'rb') as file:
            pdf_reader = PdfReader(file)
            for page_num, page in enumerate(pdf_reader.pages):
                page_text = page.extract_text()
                if page_text:
                    text += f"\n[Page {page_num + 1}]\n{page_text}"
        return text

    def chunk_text(self, text: str, source: str, chunk_size: int = 500) -> List[Document]:
        """將文字切成小塊"""
        # 清理文字
        text = re.sub(r'\s+', ' ', text).strip()
        words = text.split()

        chunks = []
        for i in range(0, len(words), chunk_size - 50):  # 50字重疊
            chunk_words = words[i:i + chunk_size]
            chunk_text = ' '.join(chunk_words)

            if len(chunk_text) > 50:
                doc = Document(
                    content=chunk_text,
                    metadata={'source': source, 'chunk_id': len(chunks)}
                )
                chunks.append(doc)

        return chunks

    def load_documents(self, data_folder: str = "week04_rag/data"):
        """載入所有文件"""
        pdf_files = list(Path(data_folder).glob("*.pdf"))

        if not pdf_files:
            # 如果沒有 PDF，使用示範文字
            print("找不到 PDF 檔案，使用內建示範文字")
            demo_docs = [
                Document("Multi-agent debate 是一種讓多個 AI 代理互相討論的技術。",
                        {"source": "demo"}),
                Document("透過辯論，代理可以產生更準確和可靠的答案。",
                        {"source": "demo"}),
                Document("RAG 系統結合檢索和生成，提供基於文件的回答。",
                        {"source": "demo"})
            ]
            self.documents = demo_docs
        else:
            # 載入所有 PDF
            print(f"找到 {len(pdf_files)} 個 PDF 檔案")
            for pdf_path in pdf_files:
                print(f"處理: {pdf_path.name}")
                text = self.load_pdf(str(pdf_path))
                chunks = self.chunk_text(text, pdf_path.name)
                self.documents.extend(chunks)
                print(f"  - 新增 {len(chunks)} 個文字塊")

        print(f"總共 {len(self.documents)} 個文字塊")

        # 建立向量索引
        texts = [doc.content for doc in self.documents]
        print("建立向量索引...")
        self.embeddings = self.embedding_model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            normalize_embeddings=True
        )
        print("向量索引建立完成！")

    def search(self, query: str, top_k: int = 3) -> List[Document]:
        """搜尋相關文件"""
        if self.embeddings is None:
            return []

        # 將問題轉成向量
        query_embedding = self.embedding_model.encode(
            [query],
            normalize_embeddings=True
        )

        # 計算相似度（內積）
        similarities = np.dot(self.embeddings, query_embedding.T).flatten()

        # 找出最相似的文件
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            results.append(self.documents[idx])
            print(f"  相似度 {similarities[idx]:.3f}: {self.documents[idx].metadata['source']}")

        return results

    def answer_question(self, question: str) -> str:
        """使用 RAG 回答問題"""
        print(f"\n問題: {question}")

        # 1. 搜尋相關文件
        print("\n搜尋相關文件...")
        relevant_docs = self.search(question, top_k=3)

        if not relevant_docs:
            return "抱歉，找不到相關資料。"

        # 2. 組合上下文
        context = "\n\n".join([
            f"[來源 {i+1}]: {doc.content}"
            for i, doc in enumerate(relevant_docs)
        ])

        # 3. 建立提示詞
        prompt = f"""根據以下參考資料回答問題。請用繁體中文回答。

參考資料：
{context}

問題：{question}

回答："""

        # 4. 呼叫 LLM
        print("\n生成回答...")
        try:
            response = ollama.chat(
                model=self.llm_model,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            return response['message']['content']
        except Exception as e:
            return f"LLM 呼叫失敗: {e}"

    def compare_with_baseline(self, question: str) -> tuple:
        """比較有/無 RAG 的回答"""
        # 使用 RAG 的回答
        rag_answer = self.answer_question(question)

        # 不使用 RAG 的回答（純 LLM）
        print("\n生成基準回答（無 RAG）...")
        try:
            baseline_response = ollama.chat(
                model=self.llm_model,
                messages=[
                    {"role": "user", "content": question}
                ]
            )
            baseline_answer = baseline_response['message']['content']
        except Exception as e:
            baseline_answer = f"LLM 呼叫失敗: {e}"

        return rag_answer, baseline_answer


def main():
    """主程式"""
    print("=== Week 4: 簡單 RAG 系統 ===\n")

    # 初始化 RAG
    rag = SimpleRAG(model_name="gemma3:1b")

    # 載入文件
    rag.load_documents("week04_rag/data")

    # 測試問題
    test_questions = [
        "什麼是 multi-agent debate？",
        "RAG 系統的優點是什麼？"
    ]

    for question in test_questions:
        print("\n" + "="*60)
        print(f"問題：{question}")

        rag_answer, baseline_answer = rag.compare_with_baseline(question)

        print("\n>>> 使用 RAG 的回答:")
        print(rag_answer)

        print("\n>>> 不使用 RAG 的回答（純 LLM）:")
        print(baseline_answer)

    print("\n" + "="*60)
    print("測試完成！")


if __name__ == "__main__":
    main()