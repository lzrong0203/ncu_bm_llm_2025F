#!/usr/bin/env python3
"""
Week 2 - Lesson 2: 結構化輸出
讓 LLM 輸出 JSON、CSV 等結構化格式
"""

import ollama
import json
import csv
import io
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum
import re

class OutputFormat(Enum):
    """輸出格式列舉"""
    JSON = "json"
    CSV = "csv"
    XML = "xml"
    MARKDOWN_TABLE = "markdown"
    PYTHON_DICT = "python"
    YAML = "yaml"

@dataclass
class Product:
    """產品資料結構"""
    name: str
    price: float
    category: str
    rating: float
    in_stock: bool

@dataclass
class ExtractedInfo:
    """提取的資訊結構"""
    entities: List[str]
    sentiment: str
    keywords: List[str]
    summary: str

class StructuredOutputGenerator:
    """結構化輸出生成器"""
    
    def __init__(self, model: str = "gemma:2b"):
        self.model = model
    
    def generate_json_output(self, text: str, schema: Dict) -> Optional[Dict]:
        """生成 JSON 格式輸出"""
        print("\n🎯 JSON 輸出生成")
        print("=" * 60)
        
        schema_str = json.dumps(schema, ensure_ascii=False, indent=2)
        
        prompt = f"""分析以下文字並以 JSON 格式輸出結果。

文字：
{text}

請嚴格遵循以下 JSON 格式：
{schema_str}

注意：
1. 輸出必須是有效的 JSON
2. 所有欄位都必須填寫
3. 不要包含額外的說明文字

JSON 輸出：
```json"""
        
        try:
            response = ollama.generate(
                model=self.model,
                prompt=prompt,
                options={'temperature': 0.1}  # 低溫度確保一致性
            )
            
            # 提取 JSON 部分
            json_text = response['response'].strip()
            
            # 清理可能的 markdown 標記
            if "```" in json_text:
                json_text = json_text.split("```")[0]
            
            # 嘗試解析 JSON
            result = json.loads(json_text)
            
            print("✅ JSON 生成成功")
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return result
            
        except json.JSONDecodeError as e:
            print(f"❌ JSON 解析失敗: {e}")
            print(f"原始輸出: {response['response'][:200]}...")
            return None
        except Exception as e:
            print(f"❌ 生成失敗: {e}")
            return None
    
    def generate_csv_output(self, data_description: str, columns: List[str]) -> Optional[str]:
        """生成 CSV 格式輸出"""
        print("\n🎯 CSV 輸出生成")
        print("=" * 60)
        
        columns_str = ",".join(columns)
        
        prompt = f"""根據以下描述生成 CSV 格式的數據。

描述：
{data_description}

CSV 格式要求：
- 欄位：{columns_str}
- 第一行必須是標題行
- 使用逗號分隔
- 如果值包含逗號，用引號包圍
- 至少生成 3 行數據

CSV 輸出：
```csv
{columns_str}"""
        
        try:
            response = ollama.generate(
                model=self.model,
                prompt=prompt,
                options={'temperature': 0.3}
            )
            
            # 提取 CSV 部分
            csv_text = response['response'].strip()
            
            # 清理 markdown 標記
            if "```" in csv_text:
                csv_text = csv_text.split("```")[0]
            
            # 驗證 CSV 格式
            csv_reader = csv.DictReader(io.StringIO(columns_str + "\n" + csv_text))
            rows = list(csv_reader)
            
            if rows:
                print("✅ CSV 生成成功")
                print(f"標題: {columns_str}")
                print(csv_text)
                return columns_str + "\n" + csv_text
            else:
                print("❌ CSV 格式無效")
                return None
                
        except Exception as e:
            print(f"❌ 生成失敗: {e}")
            return None
    
    def extract_information(self, text: str) -> Optional[ExtractedInfo]:
        """從文字中提取結構化資訊"""
        print("\n🎯 資訊提取")
        print("=" * 60)
        
        prompt = f"""從以下文字中提取資訊，並以指定格式輸出。

文字：
{text}

請提取並輸出以下資訊（JSON格式）：
{{
    "entities": ["人名", "地名", "組織名稱等"],
    "sentiment": "正面/負面/中性",
    "keywords": ["關鍵詞1", "關鍵詞2", ...],
    "summary": "一句話摘要"
}}

JSON輸出：
```json"""
        
        try:
            response = ollama.generate(
                model=self.model,
                prompt=prompt,
                options={'temperature': 0.2}
            )
            
            # 提取 JSON
            json_text = response['response'].strip()
            if "```" in json_text:
                json_text = json_text.split("```")[0]
            
            result = json.loads(json_text)
            
            # 轉換為 dataclass
            extracted = ExtractedInfo(
                entities=result.get('entities', []),
                sentiment=result.get('sentiment', ''),
                keywords=result.get('keywords', []),
                summary=result.get('summary', '')
            )
            
            print("✅ 資訊提取成功")
            print(f"實體: {extracted.entities}")
            print(f"情感: {extracted.sentiment}")
            print(f"關鍵詞: {extracted.keywords}")
            print(f"摘要: {extracted.summary}")
            
            return extracted
            
        except Exception as e:
            print(f"❌ 提取失敗: {e}")
            return None
    
    def generate_markdown_table(self, data: List[Dict]) -> str:
        """生成 Markdown 表格"""
        print("\n🎯 Markdown 表格生成")
        print("=" * 60)
        
        if not data:
            return ""
        
        # 獲取所有欄位
        headers = list(data[0].keys())
        
        # 建立表格
        table = "| " + " | ".join(headers) + " |\n"
        table += "| " + " | ".join(["---"] * len(headers)) + " |\n"
        
        for row in data:
            values = [str(row.get(h, "")) for h in headers]
            table += "| " + " | ".join(values) + " |\n"
        
        print("✅ Markdown 表格生成成功")
        print(table)
        return table
    
    def parse_product_info(self, description: str) -> Optional[Product]:
        """解析產品資訊為結構化資料"""
        print("\n🎯 產品資訊解析")
        print("=" * 60)
        
        prompt = f"""從以下產品描述中提取資訊，輸出 JSON 格式。

產品描述：
{description}

輸出格式：
{{
    "name": "產品名稱",
    "price": 價格數字,
    "category": "產品類別",
    "rating": 評分(1-5),
    "in_stock": true/false
}}

JSON：
```json"""
        
        try:
            response = ollama.generate(
                model=self.model,
                prompt=prompt,
                options={'temperature': 0.1}
            )
            
            json_text = response['response'].strip()
            if "```" in json_text:
                json_text = json_text.split("```")[0]
            
            result = json.loads(json_text)
            
            product = Product(
                name=result['name'],
                price=float(result['price']),
                category=result['category'],
                rating=float(result['rating']),
                in_stock=bool(result['in_stock'])
            )
            
            print("✅ 產品資訊解析成功")
            print(f"產品: {product.name}")
            print(f"價格: ${product.price}")
            print(f"類別: {product.category}")
            print(f"評分: {product.rating}/5")
            print(f"庫存: {'有' if product.in_stock else '無'}")
            
            return product
            
        except Exception as e:
            print(f"❌ 解析失敗: {e}")
            return None
    
    def batch_process_items(self, items: List[str], task: str) -> List[Dict]:
        """批次處理多個項目"""
        print("\n🎯 批次處理")
        print("=" * 60)
        
        results = []
        
        for i, item in enumerate(items, 1):
            print(f"\n處理項目 {i}/{len(items)}: {item[:50]}...")
            
            prompt = f"""{task}

輸入：{item}

輸出（JSON格式）：
```json"""
            
            try:
                response = ollama.generate(
                    model=self.model,
                    prompt=prompt,
                    options={'temperature': 0.2}
                )
                
                json_text = response['response'].strip()
                if "```" in json_text:
                    json_text = json_text.split("```")[0]
                
                result = json.loads(json_text)
                result['_original'] = item
                results.append(result)
                
                print(f"✅ 成功")
                
            except Exception as e:
                print(f"❌ 失敗: {e}")
                results.append({'_original': item, '_error': str(e)})
        
        print(f"\n📊 批次處理完成: {len(results)}/{len(items)} 成功")
        return results

class PromptTemplateLibrary:
    """Prompt 模板庫"""
    
    @staticmethod
    def get_json_template(schema: Dict) -> str:
        """獲取 JSON 輸出模板"""
        schema_str = json.dumps(schema, ensure_ascii=False, indent=2)
        return f"""請分析輸入並生成符合以下格式的 JSON：

Schema:
{schema_str}

規則：
1. 嚴格遵循 schema 格式
2. 所有必填欄位都要有值
3. 數字類型不要加引號
4. 布林值使用 true/false

輸入：{{input}}

JSON輸出：
```json"""
    
    @staticmethod
    def get_extraction_template(fields: List[str]) -> str:
        """獲取資訊提取模板"""
        fields_str = "\n".join([f'- {field}' for field in fields])
        return f"""從文字中提取以下資訊：

需要提取的欄位：
{fields_str}

輸出格式為 JSON，每個欄位作為一個 key。

文字：{{input}}

提取結果：
```json"""
    
    @staticmethod
    def get_classification_template(categories: List[str]) -> str:
        """獲取分類模板"""
        categories_str = ", ".join(categories)
        return f"""將以下內容分類到其中一個類別：[{categories_str}]

內容：{{input}}

分類結果（只輸出類別名稱）："""

def demonstrate_structured_outputs():
    """示範各種結構化輸出"""
    generator = StructuredOutputGenerator()
    
    # 1. JSON 輸出示範
    print("\n" + "=" * 70)
    print("📚 示範 1: JSON 輸出")
    print("=" * 70)
    
    text = "iPhone 15 Pro 是蘋果公司的旗艦手機，售價 999 美元，配備 A17 Pro 晶片，用戶評分 4.5 星，目前有貨。"
    schema = {
        "product": "產品名稱",
        "brand": "品牌",
        "price": 0,
        "features": ["特點1", "特點2"],
        "available": True
    }
    generator.generate_json_output(text, schema)
    
    # 2. CSV 輸出示範
    print("\n" + "=" * 70)
    print("📚 示範 2: CSV 輸出")
    print("=" * 70)
    
    data_desc = "生成 3 個虛擬的電商產品資料，包含手機、筆電、平板"
    columns = ["產品名稱", "價格", "庫存量", "評分"]
    generator.generate_csv_output(data_desc, columns)
    
    # 3. 資訊提取示範
    print("\n" + "=" * 70)
    print("📚 示範 3: 資訊提取")
    print("=" * 70)
    
    text = """
    張三昨天在台北 101 大樓與來自 Google 的李四會面，
    討論關於人工智慧和機器學習的合作項目。
    會議氣氛很好，雙方都對未來的合作充滿期待。
    """
    generator.extract_information(text)
    
    # 4. 產品資訊解析
    print("\n" + "=" * 70)
    print("📚 示範 4: 產品資訊解析")
    print("=" * 70)
    
    description = """
    MacBook Pro 14吋筆記型電腦，搭載 M3 Pro 晶片，
    售價 1999 美元，屬於專業筆電類別，
    用戶平均評分 4.8 顆星，目前倉庫有現貨供應。
    """
    product = generator.parse_product_info(description)
    
    # 5. Markdown 表格
    if product:
        print("\n" + "=" * 70)
        print("📚 示範 5: Markdown 表格")
        print("=" * 70)
        
        data = [asdict(product)]
        generator.generate_markdown_table(data)
    
    # 6. 批次處理
    print("\n" + "=" * 70)
    print("📚 示範 6: 批次處理")
    print("=" * 70)
    
    items = [
        "這部電影很精彩，特效一流",
        "服務態度差，不推薦",
        "產品品質不錯，價格合理"
    ]
    task = "分析情感並給出分數(1-5)和理由"
    results = generator.batch_process_items(items, task)
    
    print("\n批次處理結果：")
    for r in results:
        print(json.dumps(r, ensure_ascii=False, indent=2))

def main():
    """主程式"""
    print("=" * 70)
    print("🎓 結構化輸出生成")
    print("=" * 70)
    
    # 檢查 Ollama
    try:
        models = ollama.list()
        print("✅ 已連接到 Ollama")
    except Exception as e:
        print(f"❌ 無法連接到 Ollama: {e}")
        return
    
    # 選擇執行模式
    print("\n選擇執行模式：")
    print("1. 執行所有示範")
    print("2. 互動模式")
    
    choice = input("\n選擇 (1-2): ").strip()
    
    if choice == '1':
        demonstrate_structured_outputs()
    
    elif choice == '2':
        model = input("\n選擇模型 [預設: gemma:2b]: ").strip() or "gemma:2b"
        generator = StructuredOutputGenerator(model)
        
        while True:
            print("\n" + "=" * 60)
            print("選擇功能：")
            print("1. 生成 JSON")
            print("2. 生成 CSV")
            print("3. 提取資訊")
            print("4. 解析產品資訊")
            print("5. 生成 Markdown 表格")
            print("6. 批次處理")
            print("0. 結束")
            print("-" * 60)
            
            func_choice = input("選擇 (0-6): ").strip()
            
            if func_choice == '0':
                break
            
            elif func_choice == '1':
                text = input("\n輸入文字: ")
                schema_str = input("輸入 JSON schema (JSON 格式): ")
                try:
                    schema = json.loads(schema_str)
                    generator.generate_json_output(text, schema)
                except json.JSONDecodeError:
                    print("❌ 無效的 JSON schema")
            
            elif func_choice == '2':
                desc = input("\n輸入資料描述: ")
                columns_str = input("輸入欄位名稱 (逗號分隔): ")
                columns = [c.strip() for c in columns_str.split(',')]
                generator.generate_csv_output(desc, columns)
            
            elif func_choice == '3':
                text = input("\n輸入要分析的文字: ")
                generator.extract_information(text)
            
            elif func_choice == '4':
                desc = input("\n輸入產品描述: ")
                generator.parse_product_info(desc)
            
            elif func_choice == '5':
                print("請輸入資料 (JSON 格式的陣列):")
                data_str = input()
                try:
                    data = json.loads(data_str)
                    generator.generate_markdown_table(data)
                except json.JSONDecodeError:
                    print("❌ 無效的 JSON 資料")
            
            elif func_choice == '6':
                print("輸入項目 (每行一個，輸入空行結束):")
                items = []
                while True:
                    item = input()
                    if not item:
                        break
                    items.append(item)
                
                if items:
                    task = input("輸入處理任務描述: ")
                    generator.batch_process_items(items, task)
            
            input("\n按 Enter 繼續...")
    
    print("\n👋 課程結束！")

if __name__ == "__main__":
    main()