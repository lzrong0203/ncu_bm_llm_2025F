#!/usr/bin/env python3
"""
Week 2 - Lab: 智慧表單處理器
整合 Prompt Engineering 技巧的實用範例
"""

import ollama
import json
import re
from enum import Enum

class TaskType(Enum):
    """任務類型"""
    EXTRACT = "extract"      # 資訊提取
    CLASSIFY = "classify"     # 文字分類
    VALIDATE = "validate"     # 資料驗證
    SUMMARIZE = "summarize"   # 摘要生成

def example1_contact_extraction():
    """範例1: 聯絡資訊提取"""
    print("\n範例1: 從文字提取聯絡資訊")
    print("=" * 50)

    model = "gemma:2b"
    text = """
    您好，我是張明華，在台積電擔任資深工程師。
    我的電子郵件是 ming.zhang@tsmc.com，
    手機號碼是 0912-345-678。
    公司地址在新竹科學園區力行六路8號。
    """

    prompt = f"""從以下文字中提取聯絡資訊：

文字：{text}

請提取以下資訊（JSON格式）：
{{
    "name": "姓名",
    "email": "電子郵件",
    "phone": "電話號碼",
    "company": "公司名稱",
    "title": "職稱"
}}

JSON輸出：
```json"""

    response = ollama.generate(model=model, prompt=prompt, options={'temperature': 0.1})

    print("原始文字：", text.strip())
    print("\n提取結果：")
    try:
        json_text = response['response'].strip()
        if "```" in json_text:
            json_text = json_text.split("```")[0]
        result = json.loads(json_text)
        for key, value in result.items():
            print(f"  {key}: {value}")
    except:
        print("處理失敗")

def example2_sentiment_analysis():
    """範例2: 情感分析"""
    print("\n範例2: 情感分類")
    print("=" * 50)

    model = "gemma:2b"
    texts = [
        "產品很棒，超出預期！",
        "普普通通，沒什麼特別",
        "完全是垃圾，浪費錢"
    ]

    for text in texts:
        prompt = f"""分析以下文字的情感：

文字：{text}

請分類為：正面(positive)、負面(negative)、中性(neutral)

輸出格式：
{{
    "sentiment": "分類結果",
    "confidence": 信心分數(0-1)
}}

JSON輸出：
```json"""

        response = ollama.generate(model=model, prompt=prompt, options={'temperature': 0.2})

        print(f"\n文字: {text}")
        try:
            json_text = response['response'].strip()
            if "```" in json_text:
                json_text = json_text.split("```")[0]
            result = json.loads(json_text)
            print(f"  情感: {result.get('sentiment', 'N/A')}")
            print(f"  信心度: {result.get('confidence', 'N/A')}")
        except:
            print("  處理失敗")

def example3_intent_classification():
    """範例3: 意圖分類"""
    print("\n範例3: 客戶意圖分類")
    print("=" * 50)

    model = "gemma:2b"
    customer_messages = [
        "我上週買的產品有問題，要求退貨！",
        "請問這個產品有哪些顏色可選？",
        "我要取消我的訂單"
    ]

    for message in customer_messages:
        prompt = f"""判斷用戶意圖：

用戶輸入：{message}

可能的意圖：
- 詢問 (inquiry)
- 投訴 (complaint)
- 購買 (purchase)
- 取消 (cancel)

輸出格式：
{{
    "intent": "意圖類別",
    "confidence": 信心分數
}}

JSON輸出：
```json"""

        response = ollama.generate(model=model, prompt=prompt, options={'temperature': 0.2})

        print(f"\n客戶訊息: {message}")
        try:
            json_text = response['response'].strip()
            if "```" in json_text:
                json_text = json_text.split("```")[0]
            result = json.loads(json_text)
            print(f"  意圖: {result.get('intent', 'N/A')}")
            print(f"  信心度: {result.get('confidence', 'N/A')}")
        except:
            print("  處理失敗")

def example4_order_extraction():
    """範例4: 訂單資訊提取"""
    print("\n範例4: 訂單資訊提取")
    print("=" * 50)

    model = "gemma:2b"
    order_text = """
    訂單編號 ORD-2024-001234
    客戶：王小明
    購買項目：iPhone 15 Pro x1 (NT$38,900)、AirPods Pro x2 (NT$7,990)
    總金額：NT$54,880
    送貨地址：台北市信義區信義路五段7號
    """

    prompt = f"""從以下文字中提取訂單資訊：

文字：{order_text}

請提取以下資訊（JSON格式）：
{{
    "order_id": "訂單編號",
    "customer_name": "客戶姓名",
    "items": ["購買項目"],
    "total_amount": "總金額",
    "delivery_address": "送貨地址"
}}

JSON輸出：
```json"""

    response = ollama.generate(model=model, prompt=prompt, options={'temperature': 0.1})

    print("訂單文字：", order_text.strip())
    print("\n提取結果：")
    try:
        json_text = response['response'].strip()
        if "```" in json_text:
            json_text = json_text.split("```")[0]
        result = json.loads(json_text)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except:
        print("處理失敗")

def example5_priority_classification():
    """範例5: 優先級判斷"""
    print("\n範例5: 支援請求優先級判斷")
    print("=" * 50)

    model = "gemma:2b"
    support_requests = [
        "系統完全當機，所有用戶無法登入！",
        "想了解產品的詳細規格",
        "發票金額好像有誤差，請協助確認"
    ]

    for request in support_requests:
        prompt = f"""判斷以下請求的優先級：

請求內容：{request}

優先級別：緊急(urgent)、高(high)、中(medium)、低(low)

輸出格式：
{{
    "priority": "優先級",
    "reason": "判斷理由"
}}

JSON輸出：
```json"""

        response = ollama.generate(model=model, prompt=prompt, options={'temperature': 0.2})

        print(f"\n請求: {request}")
        try:
            json_text = response['response'].strip()
            if "```" in json_text:
                json_text = json_text.split("```")[0]
            result = json.loads(json_text)
            print(f"  優先級: {result.get('priority', 'N/A')}")
            print(f"  理由: {result.get('reason', 'N/A')}")
        except:
            print("  處理失敗")

def example6_data_summary():
    """範例6: 文字摘要"""
    print("\n範例6: 自動摘要生成")
    print("=" * 50)

    model = "gemma:2b"
    long_text = """
    人工智慧正在改變我們的生活方式。從智慧型手機的語音助理，
    到自動駕駛汽車，再到醫療診斷系統，AI 技術已經滲透到各個領域。
    這些應用不僅提高了效率，也為人類帶來了前所未有的便利。
    然而，AI 的發展也帶來了新的挑戰，包括隱私保護、就業影響和倫理問題。
    """

    # 簡短摘要
    prompt1 = f"""生成簡短摘要（一句話）：

原文：{long_text}

摘要："""

    response = ollama.generate(model=model, prompt=prompt1, options={'temperature': 0.3})
    print("原文：", long_text.strip())
    print("\n一句話摘要：", response['response'].strip())

    # 要點列表
    prompt2 = f"""提取要點（bullet points）：

原文：{long_text}

要點列表：
•"""

    response = ollama.generate(model=model, prompt=prompt2, options={'temperature': 0.3})
    print("\n要點列表：")
    print("•" + response['response'].strip())

def main():
    """執行所有範例"""
    print("智慧表單處理器範例程式")
    print("=" * 50)

    # 檢查 Ollama
    try:
        ollama.list()
        print("已連接到 Ollama")
    except Exception as e:
        print(f"請先啟動 Ollama: ollama serve")
        return

    # 執行範例
    example1_contact_extraction()
    example2_sentiment_analysis()
    example3_intent_classification()
    example4_order_extraction()
    example5_priority_classification()
    example6_data_summary()

    print("\n所有範例執行完成！")

if __name__ == "__main__":
    main()