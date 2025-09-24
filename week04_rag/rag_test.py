#!/usr/bin/env python3
"""
Week 4 - Lesson 1: Retrieval-Augmented Generation 實作

更新重點：
- 以 `data/` 資料夾中的學術論文（PDF）作為主要知識來源
- 示範針對「Multi-Agent Debate」主題的提問流程
- 同時比較「有使用 RAG」與「單純 LLM 自有知識」兩種回答，方便課堂討論

執行前請確認：
1. 已啟動 Ollama (`ollama serve`)
2. 已下載 Gemma 模型 (`ollama pull gemma3:1b`)
3. 已安裝 `requirements.txt` 中的依賴（包含 sentence-transformers、faiss-cpu、pypdf 等）

使用方式：
```bash
# 預設會載入 data/ 內的 PDF 並詢問預設問題
python week04_rag/rag_test.py

# 自訂問題或資料夾
python week04_rag/rag_test.py -q "Multi-Agent Debate 的流程是什麼？" --data-folder my_papers
```
"""

from __future__ import annotations

import argparse
import textwrap
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import faiss
import numpy as np
import ollama
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


@dataclass
class Document:
    """儲存文字內容與來源資訊的資料結構"""

    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class PDFProcessor:
    """負責載入 PDF 並切割成文字區塊"""

    def __init__(self, chunk_size: int = 600, chunk_overlap: int = 100):
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap 必須小於 chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def load_pdf(self, pdf_path: Path) -> str:
        """讀取 PDF 檔案並回傳完整文字"""
        text = ""
        with pdf_path.open("rb") as file:
            reader = PdfReader(file)
            for page_num, page in enumerate(reader.pages, start=1):
                page_text = page.extract_text() or ""
                if page_text.strip():
                    text += f"\n[Page {page_num}]\n{page_text}"
        return text.strip()

    def chunk_text(self, text: str, source: str) -> List[Document]:
        """將長文本切成帶有重疊的區塊"""
        text = re.sub(r"\s+", " ", text).strip()
        words = text.split()
        if not words:
            return []

        step = self.chunk_size - self.chunk_overlap
        chunks: List[Document] = []

        for start in range(0, len(words), step):
            window = words[start : start + self.chunk_size]
            chunk_text = " ".join(window)
            if len(chunk_text) < 80:  # 過短段落容易產生噪音，直接跳過
                continue
            chunks.append(
                Document(
                    content=chunk_text,
                    metadata={
                        "source": source,
                        "chunk_id": len(chunks),
                        "start_word": start,
                    },
                )
            )
        return chunks


class EmbeddingModel:
    """使用 sentence-transformers 產生文本向量"""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        print(f"載入嵌入模型: {model_name} (CPU mode)")
        self.model = SentenceTransformer(model_name, device="cpu")
        self.dimension = self.model.get_sentence_embedding_dimension()

    def encode(self, texts: Sequence[str], batch_size: int = 32) -> np.ndarray:
        embeddings = self.model.encode(
            list(texts),
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embeddings.astype(np.float32)


class VectorStore:
    """FAISS 向量資料庫的封裝，使用 Inner Product 做相似度"""

    def __init__(self, dimension: int):
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)
        self.documents: List[Document] = []

    def add(self, embeddings: np.ndarray, documents: Iterable[Document]) -> None:
        if embeddings.ndim != 2 or embeddings.shape[1] != self.dimension:
            raise ValueError("嵌入維度與索引不一致")
        if embeddings.size == 0:
            return
        self.index.add(embeddings)
        self.documents.extend(list(documents))

    def search(self, query: np.ndarray, top_k: int = 3) -> List[Tuple[float, Document]]:
        if not self.documents:
            return []
        if query.ndim == 1:
            query = query.reshape(1, -1)
        distances, indices = self.index.search(query, min(top_k, len(self.documents)))
        results: List[Tuple[float, Document]] = []
        for score, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self.documents):
                continue
            results.append((float(score), self.documents[idx]))
        return results


class RAGPipeline:
    """串起文件處理、向量檢索與 LLM 生成的完整流程"""

    def __init__(
        self,
        data_folder: str = "data",
        retriever_top_k: int = 3,
        chunk_size: int = 600,
        chunk_overlap: int = 100,
        llm_model: str = "gemma3:1b",
        baseline_system_prompt: str | None = None,
    ):
        self.data_folder = Path(data_folder)
        self.retriever_top_k = retriever_top_k
        self.llm_model = llm_model
        self.baseline_system_prompt = baseline_system_prompt or (
            "You are a well-read AI researcher. Answer using your prior knowledge."
        )

        self.processor = PDFProcessor(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.embedder = EmbeddingModel()
        self.vector_store = VectorStore(self.embedder.dimension)
        self.corpus: List[Document] = []
        self.ready = False

    # ------------------------------------------------------------------
    # 資料處理
    # ------------------------------------------------------------------
    def _build_fallback_documents(self) -> List[Document]:
        """當 `data/` 內沒有 PDF 時，建立內建示範資料"""
        demo_text = textwrap.dedent(
            """
            Multi-Agent Debate 是一種讓多個代理人同時討論與互相質疑的推理技巧，
            常用於改善大型語言模型的推理能力。透過多輪辯論，模型可以產生更多元的觀點，
            並挑選最合理的答案，進而提升決策品質與可靠性。
            """
        ).strip()

        course_text = textwrap.dedent(
            """
            在本課程的 RAG 章節，我們示範如何結合企業內部文件與本地模型 Gemma3:1b，
            建立能回答 Multi-Agent Debate 等進階主題的問答系統。
            情境包括客服知識庫、研究報告摘要與專案討論紀錄等。
            """
        ).strip()

        return [
            Document(content=demo_text, metadata={"source": "demo", "chunk_id": 0}),
            Document(content=course_text, metadata={"source": "demo", "chunk_id": 1}),
        ]

    def _load_pdf_documents(self) -> List[Document]:
        pdf_files = sorted(self.data_folder.glob("*.pdf"))
        if not pdf_files:
            return []

        documents: List[Document] = []
        for pdf_path in pdf_files:
            print(f"讀取 PDF: {pdf_path.name}")
            try:
                text = self.processor.load_pdf(pdf_path)
            except Exception as exc:  # pragma: no cover - 以防外部錯誤
                print(f"  無法讀取 {pdf_path.name}: {exc}")
                continue

            chunks = self.processor.chunk_text(text, source=pdf_path.name)
            print(f"  切割成 {len(chunks)} 個區塊")
            documents.extend(chunks)
        return documents

    def prepare_corpus(self) -> None:
        if not self.data_folder.exists():
            print(f"找不到資料夾 {self.data_folder}，改用內建示範文本。")
            documents = self._build_fallback_documents()
        else:
            documents = self._load_pdf_documents()
            if not documents:
                print("資料夾內沒有有效的 PDF，改用內建示範文本。")
                documents = self._build_fallback_documents()

        self.corpus = documents
        print(f"共收集 {len(self.corpus)} 個文字區塊，開始建立向量索引...")
        embeddings = self.embedder.encode([doc.content for doc in self.corpus])
        self.vector_store.add(embeddings, self.corpus)
        self.ready = len(self.corpus) > 0
        print(f"完成：索引文件 {len(self.corpus)} 筆，向量維度 {self.embedder.dimension}。")

    # ------------------------------------------------------------------
    # 問答與比較
    # ------------------------------------------------------------------
    def ask(self, question: str, *, verbose: bool = True) -> Tuple[str, List[Document]]:
        if not self.ready:
            raise RuntimeError("尚未載入知識庫，請先呼叫 prepare_corpus()")

        if verbose:
            print(f"\n使用者問題：{question}")

        query_embedding = self.embedder.encode([question])
        results = self.vector_store.search(query_embedding, top_k=self.retriever_top_k)
        if not results:
            return "抱歉，目前沒有相關資料可以回答。", []

        contexts = [doc for _score, doc in results]
        prompt = self._build_prompt(question, contexts)
        answer = self._call_llm(prompt)
        return answer, contexts

    def compare_with_baseline(self, question: str) -> Tuple[str, str, List[Document]]:
        rag_answer, contexts = self.ask(question, verbose=False)
        baseline_answer = self._call_llm_baseline(question)
        return rag_answer, baseline_answer, contexts

    # ------------------------------------------------------------------
    # Prompt & LLM 呼叫
    # ------------------------------------------------------------------
    def _build_prompt(self, question: str, contexts: Sequence[Document]) -> str:
        context_block = "\n\n".join(
            f"[來源 {idx + 1}] ({doc.metadata.get('source', 'unknown')} 第{doc.metadata.get('chunk_id')}段)\n"
            f"{doc.content}"
            for idx, doc in enumerate(contexts)
        )

        instructions = textwrap.dedent(
            f"""
            你是一位善於閱讀論文的 AI 研究助理，請根據提供的參考資料回答使用者的問題。
            - 請優先引用參考資料的內容，並在段落結尾標註來源編號（例如：來源1）。
            - 若資料不足，請清楚說明尚無完整資訊。
            - 以繁體中文條列式回答，條理清楚。

            參考資料：
            {context_block}

            使用者問題：{question}
            """
        ).strip()

        return instructions

    def _call_llm(self, prompt: str) -> str:
        try:
            response = ollama.chat(
                model=self.llm_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that strictly follows the provided context.",
                    },
                    {"role": "user", "content": prompt},
                ],
                options={"temperature": 0.2, "top_p": 0.9},
            )
        except Exception as exc:  # pragma: no cover - 依賴外部服務
            return (
                "無法連線到 Ollama，請確認服務已啟動並安裝 gemma3:1b 模型。\n"
                f"原始錯誤：{exc}"
            )

        return response.get("message", {}).get("content", "(無回應)").strip()

    def _call_llm_baseline(self, question: str) -> str:
        try:
            response = ollama.chat(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": self.baseline_system_prompt},
                    {"role": "user", "content": question},
                ],
                options={"temperature": 0.4, "top_p": 0.9},
            )
        except Exception as exc:  # pragma: no cover
            return (
                "無法連線到 Ollama，請確認服務已啟動並安裝 gemma3:1b 模型。\n"
                f"原始錯誤：{exc}"
            )

        return response.get("message", {}).get("content", "(無回應)").strip()


# ----------------------------------------------------------------------
# 指令列介面
# ----------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Week 4 RAG 教學範例")
    parser.add_argument(
        "-q",
        "--question",
        action="append",
        help="想詢問的問題，若多次提供會逐一比較 RAG 與 baseline 回答",
    )
    parser.add_argument(
        "--data-folder",
        default="data",
        help="PDF 資料來源的資料夾（預設: data）",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="檢索的參考段落數量 (default: 3)",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    pipeline = RAGPipeline(
        data_folder=args.data_folder,
        retriever_top_k=args.top_k,
    )
    pipeline.prepare_corpus()
    if not pipeline.ready:
        print("尚未成功建立知識庫，請確認 data/ 內是否有 PDF 或使用 fallback 文本。")
        return

    questions = (
        args.question
        if args.question
        else [
            "multi-agent debate is useful？",
        ]
    )

    for question in questions:
        print("\n" + "=" * 72)
        print(f"問題：{question}")

        rag_answer, baseline_answer, contexts = pipeline.compare_with_baseline(question)

        print("\n>>> 使用 RAG (含檢索上下文)：")
        print(rag_answer)
        if contexts:
            refs = ", ".join(
                f"來源{idx + 1}:{doc.metadata.get('source', 'demo')}"
                for idx, doc in enumerate(contexts)
            )
            print(f"參考資料：{refs}")

        print("\n>>> 未使用 RAG (僅模型既有知識)：")
        print(baseline_answer)

    print("\n提示：可加入更多問題 (-q) 或更換資料夾 (--data-folder) 來測試其他論文。")


if __name__ == "__main__":
    main()
