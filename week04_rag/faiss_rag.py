#!/usr/bin/env python3
"""
Week 4 - FAISS 版 RAG 系統
使用 FAISS 向量資料庫來提升檢索效能
"""

import re
import faiss
import ollama
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


class Document:
    """文件類別，用於儲存文字內容和相關資訊"""

    def __init__(self, content: str, metadata: Dict[str, Any] = None):
        self.content = content
        self.metadata = metadata or {}


class VectorStore:
    """FAISS 向量資料庫封裝"""

    def __init__(self, dimension: int):
        """初始化 FAISS 索引

        Args:
            dimension: 向量維度
        """
        self.dimension = dimension
        # 使用 Inner Product (內積) 作為相似度計算
        # 因為我們的向量已經正規化，內積等同於餘弦相似度
        self.index = faiss.IndexFlatIP(dimension)
        self.documents: List[Document] = []

    def add(self, embeddings: np.ndarray, documents: List[Document]):
        """新增向量和文件到索引

        Args:
            embeddings: 向量陣列 (n_docs, dimension)
            documents: 對應的文件列表
        """
        if embeddings.ndim != 2 or embeddings.shape[1] != self.dimension:
            raise ValueError(f"嵌入維度必須是 (n_docs, {self.dimension})")

        # 確保是 float32 類型（FAISS 要求）
        embeddings = embeddings.astype(np.float32)

        # 加入索引
        self.index.add(embeddings)
        self.documents.extend(documents)

    def search(self, query_vector: np.ndarray, top_k: int = 3) -> List[Tuple[float, Document]]:
        """搜尋最相似的文件

        Args:
            query_vector: 查詢向量
            top_k: 返回前 k 個最相似的結果

        Returns:
            [(相似度分數, 文件), ...]
        """
        if not self.documents:
            return []

        # 確保查詢向量的格式正確
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)
        query_vector = query_vector.astype(np.float32)

        # FAISS 搜尋
        # distances: 相似度分數, indices: 文件索引
        distances, indices = self.index.search(query_vector, min(top_k, len(self.documents)))

        results = []
        for score, idx in zip(distances[0], indices[0]):
            if idx >= 0 and idx < len(self.documents):
                results.append((float(score), self.documents[idx]))

        return results

    def get_stats(self) -> Dict:
        """取得索引統計資訊"""
        return {
            "total_vectors": self.index.ntotal,
            "dimension": self.dimension,
            "total_documents": len(self.documents)
        }


class FAISSRag:
    """使用 FAISS 的 RAG 系統"""

    def __init__(self, model_name: str = "gemma3:1b"):
        """初始化 RAG 系統"""
        self.llm_model = model_name
        self.embedding_model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2",
            device='cpu'
        )
        # 取得嵌入維度
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        # 初始化 FAISS 向量庫
        self.vector_store = VectorStore(self.embedding_dim)

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
                    metadata={
                        'source': source,
                        'chunk_id': len(chunks),
                        'start_index': i
                    }
                )
                chunks.append(doc)

        return chunks

    def load_documents(self, data_folder: str = "week04_rag/data"):
        """載入所有文件並建立索引"""
        pdf_files = list(Path(data_folder).glob("*.pdf"))

        all_documents = []

        if not pdf_files:
            # 使用示範文字
            print("找不到 PDF 檔案，使用內建示範文字")
            all_documents = [
                Document("Multi-agent debate 是一種讓多個 AI 代理互相討論的技術。",
                        {"source": "demo", "chunk_id": 0}),
                Document("透過辯論，代理可以產生更準確和可靠的答案。",
                        {"source": "demo", "chunk_id": 1}),
                Document("RAG 系統結合檢索和生成，提供基於文件的回答。",
                        {"source": "demo", "chunk_id": 2}),
                Document("FAISS 是 Facebook 開發的高效向量搜尋庫。",
                        {"source": "demo", "chunk_id": 3}),
            ]
        else:
            # 載入所有 PDF
            print(f"找到 {len(pdf_files)} 個 PDF 檔案")
            for pdf_path in pdf_files:
                print(f"處理: {pdf_path.name}")
                text = self.load_pdf(str(pdf_path))
                chunks = self.chunk_text(text, pdf_path.name)
                all_documents.extend(chunks)
                print(f"  - 新增 {len(chunks)} 個文字塊")

        print(f"\n總共 {len(all_documents)} 個文字塊")

        # 建立向量
        print("建立向量索引...")
        texts = [doc.content for doc in all_documents]
        embeddings = self.embedding_model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            normalize_embeddings=True  # 正規化以使用內積作為餘弦相似度
        )

        # 加入 FAISS 索引
        self.vector_store.add(embeddings, all_documents)

        # 顯示統計
        stats = self.vector_store.get_stats()
        print(f"\n=== FAISS 索引統計 ===")
        print(f"向量總數: {stats['total_vectors']}")
        print(f"向量維度: {stats['dimension']}")
        print(f"文件總數: {stats['total_documents']}")
        print("索引建立完成！\n")

    def search(self, query: str, top_k: int = 3) -> List[Tuple[float, Document]]:
        """搜尋相關文件"""
        # 將問題轉成向量
        query_embedding = self.embedding_model.encode(
            [query],
            normalize_embeddings=True
        )

        # 使用 FAISS 搜尋
        results = self.vector_store.search(query_embedding, top_k)

        # 顯示搜尋結果
        for score, doc in results:
            print(f"  相似度 {score:.3f}: {doc.metadata['source']} (chunk {doc.metadata['chunk_id']})")

        return results

    def answer_question(self, question: str) -> str:
        """使用 RAG 回答問題"""
        print(f"\n問題: {question}")

        # 1. 搜尋相關文件
        print("\n搜尋相關文件 (使用 FAISS)...")
        search_results = self.search(question, top_k=3)

        if not search_results:
            return "抱歉，找不到相關資料。"

        # 2. 組合上下文
        context = "\n\n".join([
            f"[來源 {i+1} - {doc.metadata['source']}]: {doc.content}"
            for i, (score, doc) in enumerate(search_results)
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
    print("=== Week 4: FAISS RAG 系統 ===\n")

    # 初始化 FAISS RAG
    rag = FAISSRag(model_name="gemma3:1b")

    # 載入文件
    rag.load_documents("week04_rag/data")

    # 測試問題
    test_questions = [
        "什麼是 multi-agent debate？",
        "FAISS 是什麼？有什麼優點？",
        "RAG 系統如何運作？"
    ]

    for question in test_questions:
        print("\n" + "="*60)
        print(f"問題：{question}")

        rag_answer, baseline_answer = rag.compare_with_baseline(question)

        print("\n>>> 使用 FAISS RAG 的回答:")
        print(rag_answer)

        print("\n>>> 不使用 RAG 的回答（純 LLM）:")
        print(baseline_answer)

    print("\n" + "="*60)
    print("測試完成！")


if __name__ == "__main__":
    main()