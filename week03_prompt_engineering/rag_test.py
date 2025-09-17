from typing import List, Dict, Any 

class Document:
    """文件類別，用於儲存文字內容和相關資訊"""

    def __init__(self, content: str, metadata: Dict[str, Any] = None):
        self.content = content
        self.metadata = metadata or {}

class PDFProcessor:
    """處理 PDF 檔案的類別"""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """
        初始化設定

        參數解釋：
        - chunk_size: 每個文字塊的大小（字數）
          想像成：每張筆記卡片可以寫 500 個字

        - chunk_overlap: 相鄰塊的重疊字數
          想像成：為了保持連貫，下一張卡片會重複前一張的最後 50 個字
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def load_pdf(self, pdf_path: str) -> str:
        """
        讀取 PDF 檔案

        流程：
        1. 開啟 PDF 檔案
        2. 逐頁讀取文字
        3. 合併所有頁面的文字
        """
        text = ""
        with open(pdf_path, 'rb') as file:
            pdf_reader = PdfReader(file)

            # 逐頁處理
            for page_num, page in enumerate(pdf_reader.pages):
                page_text = page.extract_text()
                if page_text:
                    # 加入頁碼標記，方便追蹤來源
                    text += f"\n[Page {page_num + 1}]\n{page_text}"

        return text

    def chunk_text(self, text: str, source: str) -> List[Document]:
        """
        將長文字切成小塊

        步驟詳解：
        1. 清理文字（移除多餘空白）
        2. 按照字數切割
        3. 保留重疊部分
        4. 記錄每塊的來源資訊
        """
        # 步驟1：清理文字
        text = re.sub(r'\s+', ' ', text)  # 多個空白變一個
        text = text.strip()                # 移除頭尾空白

        # 步驟2：分割成字詞
        words = text.split()

        chunks = []
        # 步驟3：建立文字塊
        for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
            # 取出 chunk_size 個字
            chunk_words = words[i:i + self.chunk_size]
            chunk_text = ' '.join(chunk_words)

            # 只保留有意義的塊（至少 50 個字元）
            if len(chunk_text) > 50:
                doc = Document(
                    content=chunk_text,
                    metadata={
                        'source': source,      # 來源檔案
                        'chunk_id': len(chunks),  # 第幾塊
                        'start_index': i       # 在原文的位置
                    }
                )
                chunks.append(doc)

        return chunks


def process_all_pdfs(data_folder: str = "data"):
    """
    處理資料夾中的所有 PDF 檔案並生成 embeddings

    參數：
    - data_folder: PDF 檔案所在的資料夾

    返回：
    - all_results: 包含所有檔案的 chunks 和 embeddings 的字典
    """
    # 初始化
    pdf_processor = PDFProcessor()
    embedding_model = EmbeddingModel()
    all_results = {}

    # 獲取所有 PDF 檔案
    pdf_files = list(Path(data_folder).glob("*.pdf"))
    print(f"找到 {len(pdf_files)} 個 PDF 檔案")

    # 處理每個 PDF
    for pdf_path in pdf_files:
        print(f"\n處理檔案: {pdf_path.name}")

        try:
            # 1. 讀取 PDF
            text = pdf_processor.load_pdf(str(pdf_path))
            print(f"  - 已讀取文字，長度: {len(text)} 字元")

            # 2. 切割文字
            chunks = pdf_processor.chunk_text(text, str(pdf_path))
            print(f"  - 切割成 {len(chunks)} 個文字塊")

            # 3. 生成向量
            if chunks:
                texts = [chunk.content for chunk in chunks]
                embeddings = embedding_model.encode(texts)
                print(f"  - 已生成 {len(embeddings)} 個向量，維度: {embeddings.shape}")

                # 儲存結果
                all_results[pdf_path.name] = {
                    "chunks": chunks,
                    "embeddings": embeddings
                }

        except Exception as e:
            print(f"  - 錯誤: {e}")
            continue

    # 顯示統計資訊
    print(f"\n=== 處理完成 ===")
    print(f"總共處理: {len(pdf_files)} 個 PDF 檔案")

    total_chunks = sum(len(result["chunks"]) for result in all_results.values())
    print(f"總文件塊: {total_chunks} 個")

    return all_results


class EmbeddingModel:
    """將文字轉換成向量的類別"""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        載入嵌入模型

        all-MiniLM-L6-v2 模型介紹：
        - sentence-transformers: 專門處理句子嵌入的框架
        - MiniLM: 輕量化的語言模型（Microsoft 開發）
        - L6: 6層 Transformer（較少層數，更快速）
        - v2: 第二版，改進的版本

        優點：
        - 檔案小（只有 22MB）
        - 速度快（比 BGE-large 快 5 倍）
        - 不需要 GPU，CPU 就能流暢運行
        - 準確度仍然很好

        輸出維度：384 維（384個數字表示一段文字）
        """
        print(f"載入嵌入模型: {model_name}")
        # 強制使用 CPU，避免 CUDA 相容性問題
        self.model = SentenceTransformer(model_name, device='cpu')
        self.dimension = 384  # all-MiniLM-L6-v2 的向量維度

    def encode(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        將文字列表轉換成向量

        參數說明：
        - texts: 要轉換的文字列表
        - batch_size: 批次處理大小（一次處理幾個）

        處理流程：
        1. 將文字分批（避免記憶體不足）
        2. 每批轉換成向量
        3. 正規化向量（讓長度為1，方便計算相似度）
        """
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,  # 顯示進度條
            convert_to_numpy=True,  # 轉成 NumPy 陣列
            normalize_embeddings=True,  # 正規化（重要！）
        )
        return embeddings



if __name__ == "__main__":
    process_all_pdfs()