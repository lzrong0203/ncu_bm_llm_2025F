# Week 5: LangChain 框架入門

## 課程目標
學習使用 LangChain 框架建立結構化的 LLM 應用程式，從基礎概念到實作完整的商業聊天機器人。

## 課程內容

### 📚 Lesson 1: LangChain 基礎與鏈 (01_langchain_basics.py)
- LangChain 核心概念介紹
- 基本 LLM 和 Chain 的使用
- 提示模板 (PromptTemplate) 入門
- 簡單的順序鏈實作
- LCEL (LangChain Expression Language) 基礎
- 錯誤處理與重試機制

**執行方式：**
```bash
python week05_langchain/01_langchain_basics.py
```

### 📝 Lesson 2: 提示模板與輸出解析器 (02_prompt_templates.py)
- 各種提示模板的使用 (基本、複雜、Few-Shot)
- 輸出解析器 (Output Parsers) 詳解
- JSON 和 Pydantic 結構化輸出
- 自定義解析器開發
- 輸出驗證與錯誤處理

**執行方式：**
```bash
python week05_langchain/02_prompt_templates.py
```

### 💭 Lesson 3: 記憶體與對話管理 (03_memory_management.py)
- ConversationBufferMemory 完整對話記錄
- ConversationBufferWindowMemory 視窗記憶體
- ConversationSummaryMemory 摘要記憶體
- 混合記憶體策略
- 多用戶記憶體管理
- 對話持久化存儲

**執行方式：**
```bash
python week05_langchain/03_memory_management.py
```

### 🔧 Lesson 4: LangChain + Ollama 深度整合 (04_ollama_integration.py)
- Ollama vs ChatOllama 差異比較
- 模型參數調整與優化
- 串流輸出實作
- 多模型效能比較
- 非同步操作處理
- 動態模型切換策略

**執行方式：**
```bash
python week05_langchain/04_ollama_integration.py
```

### 🤖 Lesson 5: 商業聊天機器人專案 (05_business_chatbot.py)
- 完整客服系統實作
- 查詢意圖分類
- 產品查詢與推薦
- 訂單狀態追蹤
- FAQ 智慧回答
- Streamlit Web 介面整合

**執行方式：**
```bash
# 測試模式（命令列）
python week05_langchain/05_business_chatbot.py

# Web 介面模式
streamlit run week05_langchain/05_business_chatbot.py
```

## 資料檔案

### 📦 products.json
包含 10 種電子產品的完整資訊：
- 筆記型電腦 (2款)
- 智慧手機 (2款)
- 智慧手錶 (2款)
- 無線耳機 (2款)
- 平板電腦 (1款)
- 運動相機 (1款)

### ❓ faqs.json
包含 24 個常見問題，涵蓋：
- 訂購與付款
- 運送相關
- 退換貨服務
- 保固維修
- 會員相關
- 商品諮詢
- 優惠活動
- 技術支援

## 環境需求

### 必要套件
```bash
pip install langchain langchain-community langchain-ollama
pip install streamlit
pip install pydantic
```

### Ollama 模型
```bash
# 啟動 Ollama 服務
ollama serve

# 下載必要模型
ollama pull gemma3:1b      # 主要使用
ollama pull gemma3:270m    # 輕量版（選用）
```

## 學習建議

1. **循序漸進**：從 Lesson 1 開始，逐步學習每個概念
2. **動手實作**：修改範例程式碼，觀察不同參數的影響
3. **整合應用**：嘗試結合不同課程的概念
4. **專案實踐**：基於 Lesson 5 開發自己的聊天機器人

## 作業建議

### 初級作業
1. 修改 Lesson 1，建立一個產品描述生成器
2. 使用 Lesson 2 的技術，解析結構化的客戶評論

### 中級作業
1. 擴展 Lesson 3，實作對話歷史匯出功能
2. 結合 Lesson 4，比較不同模型在特定任務的表現

### 進階作業
1. 基於 Lesson 5，加入多語言支援
2. 整合真實的產品資料庫和訂單系統
3. 加入情感分析和客戶滿意度追蹤

## 常見問題

### Q: Ollama 連接失敗？
A: 確認 Ollama 服務已啟動：`ollama serve`

### Q: 記憶體不足？
A: 使用較小的模型如 `gemma3:270m`

### Q: Streamlit 無法啟動？
A: 確認已安裝：`pip install streamlit`

### Q: 中文顯示問題？
A: 確保檔案編碼為 UTF-8

## 延伸學習

- [LangChain 官方文檔](https://python.langchain.com/)
- [Ollama 官方網站](https://ollama.ai/)
- [Streamlit 文檔](https://docs.streamlit.io/)

## 課程反饋
如有問題或建議，請聯繫課程助教或在課程討論區提出。

---
**NCU BM 2025 Fall - 本地大型語言模型的實踐與應用**