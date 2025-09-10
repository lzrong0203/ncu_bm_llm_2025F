# 本地大型語言模型的實踐與應用
NCU BM 2025 Fall - Applications of Local Large Language Models
企業管理學系 高年級/研究所 選修課程

## 📚 課程簡介

本課程專為企管系學生設計，無需程式背景，透過實作學習如何在商業環境中應用大型語言模型（LLM）。課程強調實用性，每週都會學習可立即應用於職場的 AI 工具。

### 課程特色
- 💼 **商業導向**：聚焦行銷、客服、人資、財務等商業應用
- 🎯 **零程式門檻**：提供完整程式碼，專注於應用而非開發
- 💻 **筆電友好**：使用 Gemma 3 (2B/9B) 輕量級模型
- 🌐 **最新技術**：Gemma 3 支援網路搜尋，可即時獲取資訊
- 📊 **實務專案**：每週完成可用於履歷的商業 AI 專案

## 🗂️ 課程結構

```
ncu_bm_llm_2025F/
├── week01_setup/           # Week 1: 環境設置與快速入門
│   ├── 01_hello_llm.py
│   ├── 02_personal_assistant.py
│   └── 03_ollama_basics.py
├── week02_prompt_engineering/  # Week 2: Prompt Engineering
│   ├── 01_prompting_basics.py
│   ├── 02_structured_output.py
│   └── 03_smart_form_processor.py
├── docs/                   # 課程文檔
│   ├── week01_slides.md
│   └── week02_slides.md
├── utils/                  # 工具函數
├── requirements.txt        # Python 套件需求
└── README.md              # 本文件
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
ollama pull gemma2:9b-instruct-q4_0  # 最推薦：平衡效能
ollama pull gemma2:2b                 # 備選：超輕量
ollama pull llama3.2:3b               # 備選：Meta 最新
```

#### Mac 使用者
```bash
# 1. 安裝 Homebrew（如未安裝）
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
# 2. 安裝 Ollama
brew install ollama
# 3. 下載模型
ollama pull gemma2:9b-instruct-q4_0
```

### 3. 驗證安裝

```bash
# 測試 Ollama
ollama list

# 測試第一個程式
python week01_setup/01_hello_llm.py
```

## 📅 13週課程大綱

### 第一階段：基礎建立（Week 1-3）

#### Week 1: AI 助理初體驗 ✅
- **商業情境**：打造個人工作助理
- **實作內容**：
  - Ollama 圖形化安裝（10分鐘完成）
  - 運行 Gemma 3 對話
  - 客服機器人原型
- **商業案例**：ChatGPT 企業版成功案例分析

#### Week 2: 與 AI 高效溝通 ✅
- **商業情境**：專業文件撰寫
- **實作內容**：
  - 5 種商業 Prompt 技巧
  - 自動生成行銷文案
  - Email 自動回覆系統
  - 商業報告產生器
- **商業案例**：Jasper AI 如何改變內容行銷

#### Week 3: AI 工作流程自動化
- **商業情境**：日常任務自動化
- **實作內容**：
  - 無程式碼 AI 串接
  - 多步驟任務自動化
  - 商業流程優化
- **商業案例**：RPA + AI 在企業的應用

### 第二階段：RAG 應用開發（Week 4-7）

#### Week 4: 文檔處理與向量化
- Document Loaders
- Text Splitters
- Embeddings 基礎
- 個人知識庫建立

#### Week 5: RAG 系統實作
- Vector Store 整合
- Retrieval Chain
- 課程筆記問答系統

#### Week 6: 進階 RAG 技術
- Hybrid Search
- Query Rewriting
- Re-ranking
- 智慧文件助理

#### Week 7: Web UI 開發
- Streamlit/Gradio 整合
- 對話記憶管理
- 完整 RAG Web 應用

### 第三階段：Agent 與進階應用（Week 8-11）

#### Week 8: LangChain Agents 基礎
- ReAct Agent
- Tools 開發
- 資料分析 Agent

#### Week 9: LangGraph 入門
- 狀態機架構
- Graph-based workflows
- Human-in-the-loop

#### Week 10: 多模態應用
- 圖像理解
- 圖文問答
- 文件視覺分析

#### Week 11: 模型微調入門
- LoRA/QLoRA
- 快速微調
- 領域專用模型

### 第四階段：整合與部署（Week 12-13）

#### Week 12: 系統整合與優化
- 效能優化
- API 服務建構
- 生產級部署

#### Week 13: 期末專案展示
- 專案發表
- 同儕評審
- 最佳實踐分享

## 💻 程式碼說明（提供完整程式，無需撰寫）

### Week 1 程式碼 - 商業 AI 助理

| 檔案 | 商業應用 | 使用難度 |
|------|------|----------|
| `01_hello_llm.py` | 客戶服務對話 | ⭐ 初級 |
| `02_personal_assistant.py` | 工作助理（會議記錄、備忘） | ⭐⭐ 中級 |
| `03_ollama_basics.py` | AI 能力探索 | ⭐ 初級 |

### Week 2 程式碼 - 商業文件處理

| 檔案 | 商業應用 | 使用難度 |
|------|------|----------|
| `01_prompting_basics.py` | 行銷文案生成 | ⭐ 初級 |
| `02_structured_output.py` | 訂單/發票處理 | ⭐⭐ 中級 |
| `03_smart_form_processor.py` | 客戶資料自動化 | ⭐⭐ 中級 |

## 📖 學習資源

### 官方文檔
- [Ollama Documentation](https://github.com/ollama/ollama)
- [LangChain Documentation](https://python.langchain.com/)
- [Gemma Model Card](https://ai.google.dev/gemma)

### 推薦閱讀
- 📚 《Prompt Engineering Guide》
- 📚 《Building LLM Applications》
- 📚 Chain-of-Thought Prompting Papers

### 社群支援
- 💬 課程 Discord：[加入連結]
- 📧 課程信箱：course@example.com
- 🕐 Office Hour：週四 14:00-16:00

## 🎯 作業繳交

### 繳交方式（適合初學者）
1. 完成每週實作練習
2. 截圖記錄執行結果
3. 撰寫簡短心得（200-300字）
4. 上傳到 Google Classroom 或 Email
5. 檔案命名：`學號_姓名_week{N}.pdf`

### 評分標準
- 實作完成度 (40%)
- 理解程度（心得與問答） (30%)
- 期中/期末專案 (20%)
- 課堂參與 (10%)

## ⚠️ 注意事項

### 學術誠信
- 作業須獨立完成
- 可討論但不可抄襲
- 引用須註明來源

### 硬體限制處理
- RAM 不足：使用更小的模型或量化版本
- 無 GPU：使用 CPU 版本，調整 batch size
- 儲存不足：定期清理模型快取

### 常見問題

#### Q1: Ollama 連接失敗？
```bash
# 檢查服務狀態
ollama serve
# 重啟服務
killall ollama
ollama serve
```

#### Q2: 模型下載很慢？
- 使用較小模型（gemma:2b）
- 檢查網路代理設定
- 嘗試不同時間下載

#### Q3: Python 套件安裝失敗？
```bash
# 更新 pip
pip install --upgrade pip
# 使用國內鏡像
pip install -r requirements.txt -i https://pypi.org/simple
```

## 📊 課程進度追蹤

| 週次 | 主題 | 狀態 | 作業 |
|------|------|------|------|
| Week 1 | 環境設置 | ✅ 完成 | 個人助理 |
| Week 2 | Prompt Engineering | ✅ 完成 | 5個模板 |
| Week 3 | LangChain | 🔄 進行中 | Chain 應用 |
| Week 4-13 | ... | ⏳ 待開始 | ... |

## 🤝 貢獻指南

歡迎貢獻程式碼、文檔或建議！

1. Fork 專案
2. 建立 Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit 更改 (`git commit -m 'Add some AmazingFeature'`)
4. Push 到 Branch (`git push origin feature/AmazingFeature`)
5. 開啟 Pull Request

## 📝 授權

本課程教材採用 MIT License - 詳見 [LICENSE](LICENSE) 檔案

## 👨‍🏫 講師資訊

- **講師**：[講師姓名]
- **聯絡方式**：instructor@ncu.edu.tw
- **Office Hour**：週四 14:00-16:00
- **辦公室**：[辦公室位置]

---

**Last Updated**: 2025-01-10

**Course Version**: 1.0.0