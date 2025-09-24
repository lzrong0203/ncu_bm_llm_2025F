# Week 4: RAG 暖身與專案提案
## Multi-Agent Debate × 企業知識庫

---

## 📚 本週學習目標
1. **理解 RAG (Retrieval-Augmented Generation) 流程**
   - 資料蒐集 → 切割 → 向量化 → 檢索 → 生成
2. **熟悉 Gemma3:1b 在本地端的 RAG 實作**
   - 以 `week04_rag/rag_test.py` 建立 PDF 知識庫
   - 觀察有／無 RAG 回答的差異
3. **準備期末專案提案**
   - 確認資料來源與授權
   - 定義專案目標、指標與人力分工

---

## 🧭 本週課程結構
1. **RAG 概念速成**
   - 為什麼需要 RAG？
   - RAG 與 Multi-Agent Debate 的關聯
2. **程式實作：`week04_rag/rag_test.py`**
   - 以 data/ 論文為例
   - 比較 RAG vs LLM baseline
3. **資料工作坊**
   - 清點每組可用的 PDF / 文件
   - 規劃資料切割與更新流程
4. **專案提案規劃**
   - 商業問題 → AI 解方 → 評估方法

---

## 🛠️ Demo 環境需求
```bash
ollama serve
ollama pull gemma3:1b
python -m pip install -r requirements.txt

# 建議將研究論文或 SOP 檔案放入
mkdir -p data
cp <your_papers>.pdf data/
```

### demo 指令
```bash
python week04_rag/rag_test.py
# 或自訂問題 / 資料夾
python week04_rag/rag_test.py -q "Multi-Agent Debate 的優勢是什麼？" --data-folder data
```

---

## 🔍 `rag_test.py` 重點解析
1. **PDFProcessor**：切割論文，保留來源與段落位置
2. **EmbeddingModel**：使用 `sentence-transformers/all-MiniLM-L6-v2`
3. **VectorStore**：FAISS 內積檢索，支援 top-k 參考
4. **RAGPipeline.ask()**：
   - 編碼問題 → 檢索最相關段落
   - 建立含來源的 Prompt
   - 呼叫 `ollama.chat` 產生回答
5. **compare_with_baseline()**：
   - 直接詢問 LLM（未提供上下文）
   - 與 RAG 回答並列，利於討論差異

---

## 🤖 Multi-Agent Debate 問題建議
- 「Multi-Agent Debate 與單一模型 Chain-of-Thought 有何不同？」
- 「辯論設定如何提升決策品質？對 RAG 有什麼幫助？」
- 「若企業導入 Multi-Agent Debate，需要準備哪些資料與流程？」

> 建議每組準備 2-3 類問題，觀察 RAG 與 Baseline 的回答差異、引用來源與可信度。

---

## 📂 資料準備清單
- ✅ 已授權或公開的 PDF / DOCX / TXT
- ✅ 內容與專案情境高度相關
- ✅ 頁面掃描品質良好（可讀文字）
- ✅ 若包含表格，預先轉為可解析的文字
- ✅ 標註資料來源、更新時間、敏感度

---

## 🧑‍🤝‍🧑 分組提案任務
| 項目 | 說明 |
|------|------|
| 商業問題 | 目前產業痛點 / 內部效率瓶頸 |
| AI 解決方案 | 目標使用者、核心功能、關鍵指標 |
| 資料資產 | 來源、格式、數量、更新頻率 |
| 技術架構 | 模型選擇、RAG pipeline、必要工具 |
| 驗證計畫 | 成效指標、測試流程、時程規畫 |

> 交付成果：簡報 5-7 頁，於 Week 5 前完成初稿。

---

## ✅ 本週課後任務
- [ ] 在自己的資料上成功執行 `week04_rag/rag_test.py`
- [ ] 整理「RAG vs Baseline」的回答差異與洞察
- [ ] 與組員確認專案題目與資料來源
- [ ] 產出提案簡報第一版（至少包含問題、資料、技術構想）

---

## 📎 參考資源
- [Retrieval-Augmented Generation Survey (Lewis et al., 2020)](https://arxiv.org/abs/2005.11401)
- [Multi-Agent Debate Improves Reasoning in Language Models](https://arxiv.org/abs/2305.14325)
- [LangChain Retrieval 模組官方文件](https://python.langchain.com/docs/modules/data_connection/)
- [FAISS 官方教學](https://faiss.ai/)

---

## 🙋 Q&A
歡迎帶著你們的資料集與想法到課堂上，助教會協助檢查資料品質與技術可行性。
