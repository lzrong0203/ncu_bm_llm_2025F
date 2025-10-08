# Week 5: LangChain + HuggingFace Transformers RAG 實作

## 課程目標
學習使用 LangChain 框架整合 HuggingFace Transformers，在 Google Colab 環境建立完整的 RAG（Retrieval-Augmented Generation）系統。

## 課程內容

### 📚 Lesson 1: HuggingFace Transformers 基礎
- **實作檔案**：`transformer_test.ipynb`
- **學習重點**：
  - HuggingFace Hub 認證設定
  - Transformers Pipeline 基礎使用
  - Text Generation Pipeline 實作
  - 模型載入與配置
  - LangChain + HuggingFace 整合
  - ChatHuggingFace 對話模型應用

**執行環境：** Google Colab (GPU)

**涵蓋內容：**
```python
# 1. HuggingFace 認證
from google.colab import userdata
from huggingface_hub import login

# 2. Transformers Pipeline
from transformers import pipeline
pipe = pipeline("text-generation", model="google/gemma-3-1b-it")

# 3. LangChain 整合
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
llm = HuggingFacePipeline.from_model_id(
    model_id="google/gemma-3-1b-it",
    task="text-generation"
)
chat_model = ChatHuggingFace(llm=llm)
```

### 🔍 Lesson 2: 完整 RAG 系統實作
- **實作檔案**：`langchain_rag_HF_transformers.ipynb`
- **學習重點**：
  - PDF 文件載入與處理
  - 文件分割策略 (RecursiveCharacterTextSplitter)
  - HuggingFace Embeddings 向量化
  - FAISS 向量資料庫建立與檢索
  - RAG Prompt Template 設計
  - RetrievalQA Chain 實作
  - 多語言支援 (繁體中文)

**執行環境：** Google Colab (GPU)

**完整 RAG 流程：**
```python
# 1. 載入 PDF 文件
from langchain_community.document_loaders import PyPDFLoader
loader = PyPDFLoader("your_file.pdf")
doc = loader.load()

# 2. 文件分割
from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
texts = text_splitter.split_documents(doc)

# 3. 建立 Embeddings
from langchain_huggingface import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
)

# 4. 建立向量資料庫
from langchain_community.vectorstores import FAISS
vectorstore = FAISS.from_documents(texts, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# 5. 建立 RAG Chain
from langchain.chains import RetrievalQA
qa_chain = RetrievalQA.from_chain_type(
    llm=chat_model,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)

# 6. 查詢
result = qa_chain.invoke({"query": "你的問題"})
```

## 實作範例說明

### 範例 1: 中文詩詞生成
展示 ChatHuggingFace 處理繁體中文的能力，使用 SystemMessage 和 HumanMessage 建立對話。

**示範內容：**
- 使用五言絕句格式
- 主題：Python 程式語言
- 完全使用繁體中文輸出

### 範例 2: 學術論文問答
使用真實的學術論文 PDF (Multi-Agent Debate 相關論文) 建立知識問答系統。

**示範查詢：**
- "請問 multi agent debate 有沒有用?"
- 系統會根據 PDF 內容檢索相關段落並回答
- 同時提供來源文件引用

## 環境需求

### Google Colab 設定
```python
# 掛載 Google Drive (儲存資料與模型)
from google.colab import drive
drive.mount('/content/drive')

# 設定工作目錄
%cd drive/MyDrive/data_rag
```

### 必要套件安裝
```bash
# HuggingFace 相關
pip install langchain-huggingface
pip install transformers

# LangChain 相關
pip install langchain-community
pip install langchain

# 文件處理
pip install pypdf

# 向量資料庫
pip install faiss-cpu==1.10.0
```

### HuggingFace Token 設定
1. 前往 [HuggingFace Settings](https://huggingface.co/settings/tokens)
2. 建立 Access Token
3. 在 Colab Secrets 中設定 `HF_TOKEN`

## 使用模型

### LLM 模型
- **google/gemma-3-1b-it**: 1B 參數的 Gemma 3 指令調整模型
- 適合 Colab 免費 GPU (T4)
- 支援繁體中文

### Embedding 模型
- **sentence-transformers/paraphrase-multilingual-mpnet-base-v2**
- 多語言支援 (包含繁體中文)
- 優秀的語義理解能力

## 學習建議

1. **先理解基礎**：從 `transformer_test.ipynb` 開始，了解 Transformers 基本用法
2. **完整流程**：跟著 `langchain_rag_HF_transformers.ipynb` 建立完整 RAG 系統
3. **實驗參數**：嘗試調整 chunk_size、k值、temperature 等參數
4. **使用自己的文件**：上傳自己的 PDF 文件進行測試

## 作業建議

### 初級作業
1. 修改 chunk_size 和 chunk_overlap，觀察對檢索品質的影響
2. 嘗試不同的查詢問題，測試系統回答準確度
3. 調整檢索數量 k 值 (2, 5, 10)，比較效能差異

### 中級作業
1. 使用自己的 PDF 文件建立專屬知識庫
2. 設計更複雜的 RAG Prompt Template
3. 比較不同 Embedding 模型的效果

### 進階作業
1. 實作混合檢索 (Hybrid Retrieval)
2. 加入對話記憶功能，支援多輪問答
3. 建立評估指標，量化 RAG 系統品質
4. 嘗試使用更大的模型 (gemma-3-4b-it)

## 常見問題

### Q: Colab GPU 記憶體不足？
**A:**
- 使用較小的模型 (gemma-3-1b-it 而非 4b)
- 減少 max_new_tokens 參數
- 定期清理記憶體：`del model; torch.cuda.empty_cache()`

### Q: 中文顯示亂碼？
**A:**
- 確保所有檔案使用 UTF-8 編碼
- 在 Prompt Template 中明確要求使用繁體中文

### Q: FAISS 版本衝突？
**A:**
```bash
pip install -U faiss-cpu==1.10.0
```

### Q: PDF 無法載入？
**A:**
- 確認 PDF 路徑正確
- 確認已安裝 pypdf：`pip install pypdf`
- 檢查 PDF 是否加密或損壞

### Q: HuggingFace 下載速度慢？
**A:**
- Colab 已有良好網路連線，通常不需要代理
- 首次載入會較慢，後續會使用快取

## 技術重點

### 1. 文件分割策略
- **chunk_size**: 每個文字區塊的大小
- **chunk_overlap**: 區塊間的重疊，避免語意斷裂
- 建議：中文文件使用 500-1000 字元，overlap 50-100

### 2. 檢索策略
- **k值**: 返回的相關文件數量
- **similarity_threshold**: 相似度閾值
- 平衡：k太小可能漏掉資訊，k太大會引入雜訊

### 3. Prompt 設計
- 明確指示如何使用檢索到的資訊
- 設定「不知道」的回答機制
- 要求引用來源增加可信度

## 延伸學習

### 官方資源
- [LangChain 文檔](https://python.langchain.com/)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers)
- [FAISS 文檔](https://github.com/facebookresearch/faiss)

### 推薦閱讀
- RAG 系統設計模式
- 向量資料庫比較 (FAISS vs ChromaDB vs Pinecone)
- Embedding 模型選擇指南

### 實務應用
- 企業內部知識庫系統
- 法律/醫療文件問答
- 技術文檔助手
- 客服知識管理

## 商業應用場景

1. **企業知識管理**
   - 內部文件查詢系統
   - 員工培訓材料問答
   - 政策規章快速檢索

2. **客戶服務**
   - 產品說明書智慧查詢
   - 技術支援文件助手
   - FAQ 自動回答

3. **研究與學習**
   - 學術論文問答系統
   - 教材內容檢索
   - 研究文獻整理

4. **法律與合規**
   - 合約條款查詢
   - 法規文件檢索
   - 案例分析助手

## 課程反饋
如有問題或建議，請聯繫課程助教或在課程討論區提出。

---
**NCU BM 2025 Fall - 本地大型語言模型的實踐與應用**
**Last Updated**: 2025-10-08
