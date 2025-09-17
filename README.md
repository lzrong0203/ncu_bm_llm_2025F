# 本地大型語言模型的實踐與應用
NCU BM 2025 Fall - Applications of Local Large Language Models
企業管理學系 高年級/研究所 選修課程

## ⚠️ 重要說明

> **注意**：本課程內容由 Claude Opus 4.1 協作產生，將會持續修正。經過老師親自驗證過後會加上標註。

## 📚 課程簡介

本課程專為企管系學生設計，無需程式背景，透過實作學習如何在商業環境中應用大型語言模型（LLM）。課程強調實用性，每週都會學習可立即應用於職場的 AI 工具。

### 課程特色
- 💼 **商業導向**：聚焦行銷、客服、人資、財務等商業應用
- 🎯 **零程式門檻**：提供簡化程式碼，專注於理解概念與應用
- 💻 **筆電友好**：使用 Gemma (2B/9B) 輕量級模型
- 📊 **實務專案**：每週完成可用於履歷的商業 AI 專案
- 📝 **簡化範例**：提供簡化版程式碼（`*_simple.py`），方便理解核心概念

## 🗂️ 課程結構

```
ncu_bm_llm_2025F/
├── week01_setup/                # Week 1: 環境設置與快速入門
│   ├── 01_hello_llm.py         # 基礎對話範例
│   ├── 02_personal_assistant.py # 個人助理範例
│   └── 03_ollama_basics.py     # API 功能展示
├── week03_prompt_engineering/   # Week 3: Prompt Engineering 實作
│   ├── 01_prompting_basics.py  # 提示技巧展示
│   ├── 02_structured_output.py  # 結構化輸出
│   ├── 03_smart_form_processor.py  # 智慧表單處理
│   └── 04_openai_agent_basic.py  # API 比較
├── docs/                        # 課程文檔
│   ├── week01_slides.md
│   └── week03_slides.md
├── LLM_No_framework.pdf        # Week 2 概念投影片
├── utils/                       # 工具函數
├── requirements.txt             # Python 套件需求
└── README.md                    # 本文件
```

## 🚀 快速開始

### 1. 環境需求

#### 硬體需求
- **最低配置**：8GB RAM, 4 核心 CPU
- **建議配置**：16GB RAM, 6GB VRAM GPU
- **儲存空間**：至少 10GB 可用空間

#### 軟體需求
- Python 3.9+
- Git
- Ollama

### 2. 安裝步驟（簡化版）

#### Windows 使用者
```bash
# 1. 下載課程檔案（點擊 GitHub 綠色 Code 按鈕 > Download ZIP）
# 2. 解壓縮到桌面
# 3. 下載並安裝 Ollama：https://ollama.com/download/windows
# 4. 開啟命令提示字元，執行：
ollama pull gemma2:2b                 # 超輕量，適合展示
ollama pull gemma2:9b-instruct-q4_0  # 平衡效能版
ollama pull llama3.2:3b              # Meta 最新版（備選）
```

#### Mac 使用者
```bash
# 1. 安裝 Homebrew（如未安裝）
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
# 2. 安裝 Ollama
brew install ollama
# 3. 下載模型
ollama pull gemma2:2b
```

### 3. 驗證安裝

```bash
# 測試 Ollama
ollama list

# 測試範例程式（自動執行所有範例）
python week01_setup/01_hello_llm_simple.py
```

## 📅 13週課程大綱

### 第一階段：基礎建立（Week 1-4）

#### Week 1: AI 助理初體驗 ✅
- **商業情境**：打造個人工作助理
- **實作內容**：
  - Ollama 圖形化安裝（10分鐘完成）
  - 運行 Gemma 對話
  - 客服機器人原型
- **範例程式**：`01_hello_llm_simple.py`（簡單對話、串流、溫度控制）

#### Week 2: LLM 概念與原理 ✅
- **理論內容**：參考 `LLM_No_framework.pdf`
  - 大型語言模型基本概念
  - 開源與閉源的選擇考量
  - Building an LLM application
  - Prompt Engineering 原則
  - Chain of Thought、ReAct Prompting
  - RAG 概念介紹
- **課堂重點**：概念理解，不涉及程式碼

#### Week 3: Prompt Engineering 實作 🔄
- **商業情境**：專業文件撰寫與自動化
- **實作內容**：
  - Zero-shot、Few-shot、Chain-of-Thought 技巧
  - 結構化輸出（JSON、CSV）
  - 智慧表單處理
- **範例程式**：
  - `01_prompting_basics_simple.py` - 基礎技巧展示
  - `02_structured_output_simple.py` - 結構化資料處理
  - `03_smart_form_processor_simple.py` - 商業表單自動化

#### Week 4: 專案提案 📋
- **活動內容**：
  - 學生分組（3-4人一組）
  - 提出期末專案構想
  - 專案範圍討論
  - 技術可行性評估
- **提案要求**：
  - 明確的商業問題
  - LLM 應用方案
  - 預期成果展示

### 第二階段：進階應用（Week 5-9）

#### Week 5: RAG 系統入門
- 文檔處理基礎
- 向量化概念
- 簡單 RAG 實作

#### Week 6: RAG 系統實作
- Vector Store 整合
- Retrieval Chain
- 問答系統開發

#### Week 7: Web UI 開發
- Streamlit/Gradio 整合
- 對話介面設計
- 部署基礎

#### Week 8: Agent 基礎
- Agent 概念
- Tool Use 實作
- 簡單 Agent 應用

#### Week 9: 期中專案進度報告
- 專案進度檢視
- 技術問題討論
- 同儕互評

### 第三階段：專案開發（Week 10-13）

#### Week 10-11: 專案開發時間
- 專案實作
- 個別指導
- 技術支援

#### Week 12: 專案優化與測試
- 效能優化
- 使用者測試
- 最終調整

#### Week 13: 期末專案展示
- 專案發表（每組15分鐘）
- 同儕評審
- 最佳專案頒獎

## 💻 程式碼說明

### 簡化版 vs 完整版

| 版本 | 檔名格式 | 特點 | 適用場景 |
|------|---------|------|---------|
| 簡化版 | `*_simple.py` | 直接執行、無互動、展示核心功能 | 課堂展示、快速理解 |
| 完整版 | `*.py` | 互動式選單、多功能、完整體驗 | 深入學習、自主探索 |

### Week 1 程式碼 - 商業 AI 助理

| 檔案 | 功能說明 | 商業應用 |
|------|---------|----------|
| `01_hello_llm_simple.py` | 基本對話、串流、溫度測試 | 客服對話基礎 |
| `02_personal_assistant_simple.py` | 有記憶的助理、對話儲存 | 個人化服務 |
| `03_ollama_basics_simple.py` | API 功能展示、效能測試 | 系統整合 |

### Week 3 程式碼 - Prompt Engineering

| 檔案 | 功能說明 | 商業應用 |
|------|---------|----------|
| `01_prompting_basics_simple.py` | Zero-shot、Few-shot、CoT 等技巧 | 文案生成、決策分析 |
| `02_structured_output_simple.py` | JSON/CSV 輸出、批次處理 | 訂單處理、報表生成 |
| `03_smart_form_processor_simple.py` | 情感分析、意圖分類、優先級 | 客服自動化、分流系統 |
| `04_openai_agent_basic_simple.py` | 本地vs雲端模型比較 | 成本效益分析 |

## 📖 學習資源

### 官方文檔
- [Ollama Documentation](https://github.com/ollama/ollama)
- [LangChain Documentation](https://python.langchain.com/)
- [Gemma Model Card](https://ai.google.dev/gemma)

### 課程投影片
- Week 1: `docs/week01_slides.md`
- Week 2: `LLM_No_framework.pdf`（概念講解）
- Week 3: `docs/week03_slides.md`（實作指南）

### 參考資源
- 📂 [Dr. Steve Lai 的 GitHub](https://github.com/lzrong0203/iSpan_LLM09)

## ⚠️ 注意事項

### 執行簡化版程式
```bash
# 直接執行，自動完成所有範例
python week01_setup/01_hello_llm_simple.py

# 不需要任何互動，適合課堂展示
python week02_prompt_engineering/01_prompting_basics_simple.py
```

### 常見問題

#### Q1: Ollama 連接失敗？
```bash
# 啟動 Ollama 服務
ollama serve

# 檢查模型列表
ollama list
```

#### Q2: 簡化版程式的優點？
- 無需互動輸入
- 直接看到結果
- 容易理解核心概念
- 適合課堂快速展示

#### Q3: 如何選擇適合的模型？
- **gemma2:2b**：最輕量，適合展示和學習
- **gemma2:9b-instruct-q4_0**：效果較好，需要較多資源
- **llama3.2:3b**：Meta 最新模型，平衡選擇

## 📊 課程進度追蹤

| 週次 | 主題 | 狀態 | 備註 |
|------|------|------|------|
| Week 1 | 環境設置與入門 | ✅ 完成 | 已提供簡化版程式 |
| Week 2 | LLM 概念講解 | ✅ 完成 | 參考 PDF 投影片 |
| Week 3 | Prompt Engineering 實作 | 🔄 本週 | 使用簡化版範例 |
| Week 4 | 專案提案 | ⏳ 下週 | 分組討論 |
| Week 5-13 | 進階內容 | ⏳ 待開始 | 持續更新中 |

## 🤝 貢獻指南

歡迎貢獻程式碼、文檔或建議！

1. Fork 專案
2. 建立 Feature Branch
3. Commit 更改
4. Push 到 Branch
5. 開啟 Pull Request

## 📝 授權

本課程教材採用 MIT License

## 👨‍🏫 講師資訊

- **講師**：Steve Lai
- **聯絡方式**：lzrong0203@gmail.com

---

**Last Updated**: 2025-09-17
**Course Version**: 1.1.0