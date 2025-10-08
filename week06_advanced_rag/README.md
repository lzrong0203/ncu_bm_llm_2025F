# Week 6: Output Parser 進階 RAG 應用

## 課程目標
學習使用 LangChain Output Parser 建立結構化的 RAG 輸出系統，並掌握模型量化技術，將進階技術應用於實際商業場景。

## 課程內容

### 📚 Lesson 1: 量化技術實戰
- **學習重點**：
  - 理解模型量化原理與優勢
  - 實作 4-bit BitsAndBytes 量化
  - 比較量化前後的記憶體使用與效能
  - 優化 Colab 免費版 GPU 使用

**核心技術**：
```python
from transformers import BitsAndBytesConfig
import torch

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,                      # 啟用 4-bit 量化
    bnb_4bit_compute_dtype=torch.bfloat16,  # 計算精度
    bnb_4bit_quant_type="nf4",             # NormalFloat 4-bit
    bnb_4bit_use_double_quant=True,        # 雙重量化
)
```

**量化優勢**：
- 記憶體使用減少約 75%
- 1B 模型從 ~2GB 降至 ~0.5GB
- 推理速度略有提升
- 精度損失極小（<1%）
- Colab 免費版可跑更大模型

### 🔍 Lesson 2: Output Parser 完整教學
- **學習重點**：
  - 掌握 LangChain 各類 Output Parser
  - 理解結構化輸出的重要性
  - 實作型別安全的資料提取
  - 錯誤處理與驗證機制

**Parser 類型**：

1. **StrOutputParser** - 基本字串解析
   - 最簡單的 parser
   - 直接返回 LLM 輸出字串
   - 適用場景：簡單問答

2. **CommaSeparatedListOutputParser** - 列表解析
   - 解析逗號分隔的列表
   - 自動轉換為 Python List
   - 適用場景：多項目列舉

3. **JsonOutputParser** - JSON 格式解析
   - 解析 JSON 格式輸出
   - 支援 Pydantic 模型定義
   - 適用場景：結構化資料提取

4. **PydanticOutputParser** - 型別安全解析（推薦！）
   - 使用 Pydantic 模型定義結構
   - 自動型別檢查與驗證
   - 提供詳細的格式指示
   - 適用場景：複雜結構化輸出

5. **自定義 Parser**
   - 繼承 BaseOutputParser
   - 處理特殊格式需求
   - 適用場景：特殊業務邏輯

### 🚀 Lesson 3: 進階檢索技術 - BM25、Hybrid Search、Rerank
- **學習重點**：
  - 掌握 BM25 關鍵字檢索演算法
  - 實作 Hybrid Search (BM25 + Vector)
  - 使用 Cross-Encoder Reranker 提升精確度
  - 比較不同檢索方法的效能與適用場景

**核心技術**：

#### 1. BM25 Retriever - 關鍵字檢索
```python
from langchain.retrievers import BM25Retriever

bm25_retriever = BM25Retriever.from_documents(texts)
bm25_retriever.k = 3
```

**特點**：
- 基於 TF-IDF 的排序演算法
- 精確關鍵字匹配
- 無需向量化，速度快
- 不理解語義，對同義詞不敏感

#### 2. Hybrid Search - 混合檢索
```python
from langchain.retrievers import EnsembleRetriever

hybrid_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.5, 0.5]  # BM25 與 Vector 各佔 50%
)
```

**優勢**：
- 結合關鍵字與語義檢索
- 更穩健的檢索效果
- 適應不同查詢類型

#### 3. Reranker - 重新排序
```python
from sentence_transformers import CrossEncoder

reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

def rerank_documents(query, documents, top_k=3):
    pairs = [[query, doc.page_content] for doc in documents]
    scores = reranker.predict(pairs)
    # 按分數排序並返回 top_k
```

**特點**：
- Cross-Encoder 同時考慮查詢與文檔
- 更精確的相關性評分
- 速度較慢，適合二次精煉
- 顯著提升 RAG 系統品質

#### 4. 完整 Pipeline: Hybrid + Rerank
```python
def hybrid_rerank_retriever(query, k=3, candidate_k=10):
    # Step 1: Hybrid 檢索獲取候選
    candidates = hybrid_retriever.invoke(query)[:candidate_k]

    # Step 2: Rerank 精煉結果
    reranked = rerank_documents(query, candidates, top_k=k)

    return reranked
```

**檢索方法比較**：

| 方法 | 優點 | 缺點 | 適用場景 |
|------|------|------|----------|
| Vector | 語義理解強 | 關鍵字匹配弱 | 語義查詢 |
| BM25 | 關鍵字精確 | 無語義理解 | 精確匹配 |
| Hybrid | 兼具兩者優勢 | 參數調整複雜 | 通用查詢 |
| Hybrid+Rerank | 最高準確度 | 速度較慢 | 品質要求高 ⭐ |

### 🎯 Lesson 4: RAG + Output Parser 整合
- **學習重點**：
  - 理解為何 RAG 需要結構化輸出
  - 設計 RAG Prompt + Parser 組合
  - 實作輸出驗證與錯誤處理
  - 提升 RAG 系統可用性

**實作案例**：

#### 案例 1: 學術論文關鍵資訊提取
```python
class PaperAnalysis(BaseModel):
    title: str = Field(description="論文標題")
    main_finding: str = Field(description="主要研究發現")
    methodology: str = Field(description="研究方法")
    key_contributions: List[str] = Field(description="主要貢獻")
    confidence: float = Field(description="回答信心度", ge=0, le=1)
```

**應用場景**：
- 快速分析多篇論文
- 提取關鍵研究資訊
- 建立論文摘要資料庫

#### 案例 2: 技術問答系統
```python
class TechnicalQA(BaseModel):
    question_type: str = Field(description="問題類型")
    answer: str = Field(description="詳細回答")
    key_terms: List[str] = Field(description="關鍵術語")
    difficulty: str = Field(description="難度等級")
    related_topics: List[str] = Field(description="相關主題")
    sources_used: int = Field(description="使用來源數量")
```

**應用場景**：
- 技術文件智慧問答
- 程式教學輔助
- API 文件查詢

#### 案例 3: 研究方法比較分析
```python
class MethodComparison(BaseModel):
    method_name: str = Field(description="方法名稱")
    advantages: List[str] = Field(description="優點")
    disadvantages: List[str] = Field(description="缺點")
    use_cases: List[str] = Field(description="適用場景")
    performance_note: str = Field(description="效能說明")
    recommendation: str = Field(description="使用建議")
```

**應用場景**：
- 學術研究比較
- 技術選型分析
- 方案評估報告

### 💼 Lesson 4: 商業應用實戰
- **學習重點**：
  - 設計完整的商業級 RAG 系統
  - 實作智慧客服知識庫
  - 多欄位結構化輸出
  - 系統可靠性與錯誤處理

**完整商業案例**：

#### 智慧客服系統
```python
class CustomerServiceResponse(BaseModel):
    intent: str = Field(description="客戶意圖")
    answer: str = Field(description="客服回答")
    sentiment: str = Field(description="問題情緒")
    confidence: float = Field(description="信心度", ge=0, le=1)
    suggested_action: str = Field(description="建議後續動作")
    escalate: bool = Field(description="是否需要轉人工")
    related_docs: List[str] = Field(description="相關文件")
```

**系統功能**：
- 自動意圖識別
- 情感分析
- 智慧回答生成
- 人工轉接判斷
- 相關資源推薦

**商業價值**：
- 24/7 自動化客服
- 減少人工成本
- 提升回應速度
- 提高客戶滿意度

## 📓 主要教學檔案

### week06_output_parser_rag.ipynb
完整的 Colab Notebook，包含：

**Part 1: 環境設定** (Cell 1-4)
- 套件安裝與認證
- Google Drive 掛載
- 基礎套件導入

**Part 2: 量化技術實戰** (Cell 5-7)
- 4-bit 量化配置
- 量化模型載入
- 效能測試與比較

**Part 3: Output Parser 基礎教學** (Cell 8-12)
- StrOutputParser 範例
- CommaSeparatedListOutputParser 範例
- JsonOutputParser 範例
- PydanticOutputParser 深入講解
- Parser 類型比較總結

**Part 4: RAG 系統建立** (Cell 13-15)
- 載入論文資料（使用 week04 論文）
- 文件分割與向量化
- 建立基礎 RAG Chain

**Part 5: RAG + Parser 整合應用** (Cell 16-18)
- 案例 1: 論文關鍵資訊提取
- 案例 2: 結構化技術問答
- 案例 3: 方法比較分析

**Part 6: 商業應用實戰** (Cell 19-20)
- 智慧客服系統範例
- 錯誤處理與重試機制

**Part 7: 進階技巧與優化** (Cell 21-22)
- 自定義 Output Parser
- 效能優化建議

**Part 8: 總結與作業** (Cell 23-24)
- 課程總結
- 作業說明

### langchain_rag_HF_transformers.ipynb
已整合量化技術的基礎 RAG 範例

### 02_prompt_templates.py
Output Parser 參考範例（Python 版本）

## 環境需求

### Google Colab 設定
```python
# 掛載 Google Drive
from google.colab import drive
drive.mount('/content/drive')

# 切換到工作目錄
%cd drive/MyDrive/data_rag
```

### 必要套件安裝
```bash
# HuggingFace 相關
pip install langchain-huggingface transformers accelerate bitsandbytes

# LangChain 相關
pip install langchain-community langchain

# 文件處理與向量資料庫
pip install pypdf faiss-cpu==1.10.0

# 資料驗證
pip install pydantic
```

### HuggingFace Token 設定
1. 前往 [HuggingFace Settings](https://huggingface.co/settings/tokens)
2. 建立 Access Token
3. 在 Colab Secrets 中設定 `HF_TOKEN`

## 使用模型

### LLM 模型
- **google/gemma-3-1b-it**: 1B 參數模型
  - 量化後: ~0.5GB
  - 適合 Colab 免費版
  - 支援繁體中文
  - 推薦用於結構化輸出

### Embedding 模型
- **sentence-transformers/paraphrase-multilingual-mpnet-base-v2**
  - 多語言支援（包含繁體中文）
  - 優秀的語義理解能力
  - 適合學術文件

## 資料來源

### 論文資料
使用 `week04_rag/data/` 中的學術論文：
- `2509.05396v1.pdf` - Multi-Agent Debate 論文（主要使用）
- `2305.14325v1.pdf`
- `2506.08292v1.pdf`
- `2502.14767v2.pdf`

**優點**：
- 真實學術資料
- 複雜的技術內容
- 適合測試 RAG 系統

## 技術重點

### 1. 模型量化策略
**4-bit 量化參數說明**：
- `load_in_4bit`: 將權重從 16-bit 降至 4-bit
- `bnb_4bit_compute_dtype`: 計算時的資料型別（bfloat16）
- `bnb_4bit_quant_type`: 量化類型（NF4 適合神經網路）
- `bnb_4bit_use_double_quant`: 量化常數也量化

**效果評估**：
- 記憶體：↓ 75%
- 速度：↑ 10-20%
- 精度：↓ <1%
- 結論：極高性價比

### 2. Output Parser 選擇指南
| 場景 | 推薦 Parser | 理由 |
|------|------------|------|
| 簡單問答 | StrOutputParser | 直接、快速 |
| 列表輸出 | CommaSeparatedListOutputParser | 自動解析 |
| 複雜結構 | PydanticOutputParser | 型別安全 |
| 彈性資料 | JsonOutputParser | 靈活性高 |
| 特殊格式 | 自定義 Parser | 完全控制 |

**最佳實踐**：
- 結構化輸出優先選 PydanticOutputParser
- 使用 temperature=0 或極低值
- 實作重試機制
- 提供詳細的格式指示

### 3. RAG 系統優化
**文件分割**：
- chunk_size: 500-1000（依文件類型）
- chunk_overlap: 50-100
- 平衡：語意完整性 vs 檢索精度

**檢索策略**：
- k值: 3-5（平衡品質與速度）
- similarity_threshold: 依需求調整
- 考慮 re-ranking 機制

**Prompt 設計**：
- 明確的輸出格式要求
- 提供範例（Few-shot）
- 強調「不知道」的處理

### 4. 錯誤處理機制
**重試策略**：
```python
def safe_rag_with_parser(query, parser, max_retries=2):
    for attempt in range(max_retries):
        try:
            # RAG + Parser 流程
            return result
        except Exception as e:
            if attempt == max_retries - 1:
                # 返回降級方案
                return fallback_result
```

**防禦措施**：
- Pydantic 欄位驗證
- 設定預設值
- 記錄失敗案例
- 監控成功率

## 學習建議

1. **循序漸進**
   - 先理解量化原理
   - 逐個學習 Parser 類型
   - 由簡到繁實作案例

2. **動手實作**
   - 執行所有 Notebook cells
   - 修改參數觀察變化
   - 嘗試不同的 Pydantic 模型

3. **實驗參數**
   - 調整 temperature (0.0 - 0.2)
   - 變更 chunk_size 和 k 值
   - 比較不同量化配置

4. **應用場景**
   - 思考自己的業務需求
   - 設計對應的資料模型
   - 建立專屬的 RAG 系統

## 作業建議

### 初級作業
1. **Pydantic 模型設計**
   - 設計 `ProductReview` 模型（5+ 欄位）
   - 包含：名稱、評分、摘要、優缺點

2. **量化效能比較**
   - 測量量化前後記憶體使用
   - 比較推理速度
   - 評估輸出品質

3. **基礎 RAG + Parser**
   - 使用 JsonOutputParser 提取論文資訊
   - 包含：標題、作者、主要貢獻

### 中級作業
4. **產品推薦系統**
   - 建立結構化推薦輸出
   - 包含：產品資訊、理由、場景、信心度

5. **技術文件問答**
   - 整合 RAG + 多欄位輸出
   - 包含：分類、回答、程式碼、資源

6. **錯誤處理機制**
   - 實作 max_retries (3次)
   - 設計降級方案
   - 記錄錯誤日誌

### 中級作業（續）
7. **Hybrid Search 實驗**
   - 比較 BM25、Vector、Hybrid 三種檢索方法
   - 測試不同 weights 配置 (如 [0.3, 0.7])
   - 記錄不同查詢類型的檢索效果

8. **Reranker 整合**
   - 實作完整 Hybrid + Rerank pipeline
   - 比較 Rerank 前後的檢索品質
   - 測量延遲增加幅度

### 進階作業
9. **多文件檢索系統**
   - 使用所有 week04 論文
   - 跨文件檢索與比較
   - 文件來源標註

10. **RAG 品質評估**
    - 答案相關性評分
    - 來源可信度評估
    - 格式完整性檢查

11. **完整客服系統**
    - 意圖分類 + 情感分析
    - 自動回覆生成
    - 人工轉接判斷
    - 滿意度追蹤

12. **大模型量化實驗**
    - 嘗試 gemma-3-4b-it
    - 優化量化配置
    - 比較 1B vs 4B 效果

13. **檢索優化挑戰**
    - 實作自適應權重調整（根據查詢類型動態調整 BM25 vs Vector 權重）
    - 嘗試不同的 Reranker 模型
    - 建立檢索效能監控系統

## 常見問題

### Q: Colab 提示 CUDA out of memory？
**A:**
```python
# 1. 使用更小的模型
model_id = "google/gemma-3-1b-it"  # 而非 4b

# 2. 減少 max_new_tokens
pipeline_kwargs = {"max_new_tokens": 256}  # 從 512 降低

# 3. 清理記憶體
import gc
import torch
gc.collect()
torch.cuda.empty_cache()

# 4. 重啟 Runtime
# Runtime > Restart runtime
```

### Q: Output Parser 解析失敗？
**A:**
```python
# 1. 降低 temperature
temperature = 0.0  # 或 0.1

# 2. 簡化 Pydantic 模型
# 減少欄位數量，避免過於複雜

# 3. 提供更詳細的 format_instructions
# 在 prompt 中加入範例

# 4. 實作重試機制
max_retries = 3
```

### Q: bitsandbytes 安裝失敗？
**A:**
```bash
# 1. 安裝最新版本
!pip install -U bitsandbytes accelerate

# 2. 重啟 Runtime（必須！）
# Runtime > Restart runtime

# 3. 確認 CUDA 可用
import torch
print(torch.cuda.is_available())  # 應為 True
```

### Q: RAG 回答品質不佳？
**A:**
```python
# 1. 調整檢索參數
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 5}  # 增加 k 值
)

# 2. 優化文件分割
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,        # 增加區塊大小
    chunk_overlap=100      # 增加重疊
)

# 3. 改善 Prompt
# 提供更詳細的指示和範例

# 4. 使用更好的 Embedding 模型
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-large-en-v1.5"  # 更強的模型
)
```

### Q: 如何持久化向量資料庫？
**A:**
```python
# 1. 儲存到 Google Drive
vectorstore.save_local("drive/MyDrive/vector_db")

# 2. 下次直接載入
vectorstore = FAISS.load_local(
    "drive/MyDrive/vector_db",
    embeddings
)

# 好處：不用每次都重新建立
```

## 延伸學習

### 官方資源
- [LangChain Output Parsers](https://python.langchain.com/docs/modules/model_io/output_parsers/)
- [Pydantic Documentation](https://docs.pydantic.dev/)
- [BitsAndBytes GitHub](https://github.com/TimDettmers/bitsandbytes)
- [HuggingFace Quantization](https://huggingface.co/docs/transformers/main_classes/quantization)

### 推薦閱讀
- **量化技術**：
  - GPTQ vs AWQ vs BitsAndBytes 比較
  - 量化對不同任務的影響
  - 最佳量化配置指南

- **Output Parser**：
  - 結構化輸出最佳實踐
  - Pydantic 進階技巧
  - 自定義 Parser 設計模式

- **RAG 優化**：
  - RAG 系統評估指標
  - 進階檢索策略（Hybrid Search, Re-ranking）
  - RAG vs Fine-tuning 選擇

### 實務應用參考
1. **企業知識管理**
   - 內部文件問答系統
   - 政策規章快速查詢
   - 員工培訓材料檢索

2. **客戶服務**
   - 智慧客服機器人
   - 技術支援文件助手
   - FAQ 自動回答系統

3. **研究與分析**
   - 學術論文摘要提取
   - 市場研究報告分析
   - 專利文件檢索

4. **法律與合規**
   - 合約條款分析
   - 法規文件查詢
   - 風險評估報告

## 效能優化清單

### ✅ 已優化
- [x] 4-bit 模型量化
- [x] Pydantic 型別驗證
- [x] 錯誤處理與重試
- [x] 結構化輸出設計

### 🚀 可進一步優化
- [ ] Embedding 快取機制
- [ ] 向量資料庫持久化
- [ ] 批次處理多個查詢
- [ ] GPU 記憶體池管理
- [ ] 非同步 RAG Pipeline
- [ ] 查詢結果快取
- [ ] Re-ranking 機制
- [ ] Hybrid Search 整合

## 課程反饋
如有問題或建議，請聯繫課程助教或在課程討論區提出。

---
**NCU BM 2025 Fall - 本地大型語言模型的實踐與應用**
**Last Updated**: 2025-10-08
**Week**: 6 - Output Parser 進階 RAG 應用
