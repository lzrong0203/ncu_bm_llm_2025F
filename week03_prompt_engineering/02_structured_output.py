#!/usr/bin/env python3
"""
Week 2 - Lesson 2: 結構化輸出
讓 LLM 輸出 JSON、CSV 等結構化格式範例
"""

import ollama
import json
import csv
import io

def example1_json_output():
    """範例1: 生成 JSON 格式輸出"""
    print("\n範例1: JSON 輸出")
    print("=" * 50)

    model = "gemma:2b"
    text = "iPhone 15 Pro 是蘋果公司的旗艦手機，售價 999 美元，配備 A17 Pro 晶片"

    prompt = f"""分析以下文字並以 JSON 格式輸出：

文字：{text}

請輸出以下格式的 JSON：
{{
    "product": "產品名稱",
    "brand": "品牌",
    "price": 價格數字,
    "features": ["特點1", "特點2"]
}}

JSON輸出：
```json"""

    response = ollama.generate(model=model, prompt=prompt, options={'temperature': 0.1})

    print("輸入文字：", text)
    print("JSON 輸出：")
    try:
        # 清理回應文字
        json_text = response['response'].strip()
        if "```" in json_text:
            json_text = json_text.split("```")[0]

        result = json.loads(json_text)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except:
        print("原始輸出：", response['response'][:200])

def example2_csv_output():
    """範例2: 生成 CSV 格式輸出"""
    print("\n範例2: CSV 輸出")
    print("=" * 50)

    model = "gemma:2b"
    columns = ["產品名稱", "價格", "庫存量"]

    prompt = f"""生成 3 個虛擬產品的 CSV 資料。

CSV 格式要求：
- 欄位：{",".join(columns)}
- 使用逗號分隔
- 生成 3 行資料

CSV輸出：
```csv
{",".join(columns)}"""

    response = ollama.generate(model=model, prompt=prompt, options={'temperature': 0.3})

    print("CSV 輸出：")
    csv_text = response['response'].strip()
    if "```" in csv_text:
        csv_text = csv_text.split("```")[0]
    print(f"{','.join(columns)}")
    print(csv_text)

def example3_extract_info():
    """範例3: 提取結構化資訊"""
    print("\n範例3: 資訊提取")
    print("=" * 50)

    model = "gemma:2b"
    text = """張三昨天在台北 101 大樓與來自 Google 的李四會面，
    討論關於人工智慧的合作項目。會議氣氛很好。"""

    prompt = f"""從以下文字提取資訊：

文字：{text}

請提取並輸出（JSON格式）：
{{
    "people": ["人名"],
    "locations": ["地點"],
    "organizations": ["組織"],
    "sentiment": "正面/負面/中性"
}}

JSON輸出：
```json"""

    response = ollama.generate(model=model, prompt=prompt, options={'temperature': 0.2})

    print("原始文字：", text)
    print("\n提取結果：")
    try:
        json_text = response['response'].strip()
        if "```" in json_text:
            json_text = json_text.split("```")[0]

        result = json.loads(json_text)
        print(f"人名: {result.get('people', [])}")
        print(f"地點: {result.get('locations', [])}")
        print(f"組織: {result.get('organizations', [])}")
        print(f"情感: {result.get('sentiment', '')}")
    except:
        print("原始輸出：", response['response'][:200])

def example4_markdown_table():
    """範例4: 生成 Markdown 表格"""
    print("\n範例4: Markdown 表格")
    print("=" * 50)

    # 直接建立表格（不依賴 LLM）
    data = [
        {"產品": "筆電", "價格": "30000", "評分": "4.5"},
        {"產品": "手機", "價格": "20000", "評分": "4.8"},
        {"產品": "平板", "價格": "15000", "評分": "4.2"}
    ]

    headers = list(data[0].keys())
    table = "| " + " | ".join(headers) + " |\n"
    table += "| " + " | ".join(["---"] * len(headers)) + " |\n"

    for row in data:
        values = [str(row.get(h, "")) for h in headers]
        table += "| " + " | ".join(values) + " |\n"

    print("Markdown 表格：")
    print(table)

def example5_batch_processing():
    """範例5: 批次處理多個項目"""
    print("\n範例5: 批次處理")
    print("=" * 50)

    model = "gemma:2b"
    items = [
        "這部電影很精彩",
        "服務態度差",
        "產品品質不錯"
    ]

    print("批次情感分析：")
    for i, item in enumerate(items, 1):
        prompt = f"""分析情感（1-5分）：

評論：{item}

輸出JSON：
{{"score": 分數, "reason": "原因"}}

```json"""

        response = ollama.generate(model=model, prompt=prompt, options={'temperature': 0.2})

        print(f"\n項目 {i}: {item}")
        try:
            json_text = response['response'].strip()
            if "```" in json_text:
                json_text = json_text.split("```")[0]

            result = json.loads(json_text)
            print(f"  分數: {result.get('score', 'N/A')}")
            print(f"  原因: {result.get('reason', 'N/A')}")
        except:
            print(f"  處理失敗")

def example6_template_usage():
    """範例6: 使用提示模板"""
    print("\n範例6: 提示模板使用")
    print("=" * 50)

    model = "gemma:2b"

    # 分類模板
    categories = ["科技", "體育", "娛樂", "政治"]
    text = "新款遊戲主機發表，搭載最新處理器"

    prompt = f"""將以下內容分類到其中一個類別：[{', '.join(categories)}]

內容：{text}

類別："""

    response = ollama.generate(model=model, prompt=prompt, options={'temperature': 0.1})

    print(f"文字：{text}")
    print(f"可選類別：{categories}")
    print(f"分類結果：{response['response'].strip()}")

def main():
    """執行所有範例"""
    print("結構化輸出範例程式")
    print("=" * 50)

    # 檢查 Ollama
    try:
        ollama.list()
        print("已連接到 Ollama")
    except Exception as e:
        print(f"請先啟動 Ollama: ollama serve")
        return

    # 執行範例
    example1_json_output()
    example2_csv_output()
    example3_extract_info()
    example4_markdown_table()
    example5_batch_processing()
    example6_template_usage()

    print("\n所有範例執行完成！")

if __name__ == "__main__":
    main()