#!/usr/bin/env python3
"""
Week 2 - Lab: 智慧表單處理器
整合 Prompt Engineering 技巧的實用應用
"""

import ollama
import json
import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
import os

class TaskType(Enum):
    """任務類型"""
    EXTRACT = "extract"          # 資訊提取
    CLASSIFY = "classify"         # 文字分類
    VALIDATE = "validate"         # 資料驗證
    TRANSFORM = "transform"       # 格式轉換
    SUMMARIZE = "summarize"       # 摘要生成

@dataclass
class FormField:
    """表單欄位定義"""
    name: str
    field_type: str  # text, number, email, phone, date, select
    required: bool = True
    validation_rules: List[str] = field.default_factory(list)
    options: List[str] = field.default_factory(list)  # for select type

@dataclass
class ProcessingResult:
    """處理結果"""
    success: bool
    task_type: TaskType
    input_text: str
    output_data: Dict[str, Any]
    errors: List[str] = field.default_factory(list)
    confidence: float = 0.0
    processing_time: float = 0.0

class PromptTemplate:
    """Prompt 模板管理"""
    
    def __init__(self):
        self.templates = {
            TaskType.EXTRACT: {
                "contact_info": self.contact_extraction_template(),
                "order_info": self.order_extraction_template(),
                "event_info": self.event_extraction_template(),
                "invoice_info": self.invoice_extraction_template(),
                "resume_info": self.resume_extraction_template()
            },
            TaskType.CLASSIFY: {
                "sentiment": self.sentiment_classification_template(),
                "intent": self.intent_classification_template(),
                "priority": self.priority_classification_template(),
                "category": self.category_classification_template(),
                "language": self.language_detection_template()
            },
            TaskType.VALIDATE: {
                "email": self.email_validation_template(),
                "phone": self.phone_validation_template(),
                "data_quality": self.data_quality_template()
            },
            TaskType.TRANSFORM: {
                "date_format": self.date_transform_template(),
                "currency": self.currency_transform_template(),
                "units": self.unit_conversion_template()
            },
            TaskType.SUMMARIZE: {
                "brief": self.brief_summary_template(),
                "detailed": self.detailed_summary_template(),
                "bullet_points": self.bullet_points_template()
            }
        }
    
    def contact_extraction_template(self) -> str:
        """聯絡資訊提取模板"""
        return """從以下文字中提取聯絡資訊：

文字：
{input}

請提取以下資訊（JSON格式）：
{{
    "name": "姓名",
    "email": "電子郵件",
    "phone": "電話號碼",
    "address": "地址",
    "company": "公司名稱",
    "title": "職稱"
}}

注意：如果某項資訊不存在，請填 null

JSON輸出：
```json"""
    
    def order_extraction_template(self) -> str:
        """訂單資訊提取模板"""
        return """從以下文字中提取訂單資訊：

文字：
{input}

請提取以下資訊（JSON格式）：
{{
    "order_id": "訂單編號",
    "customer_name": "客戶姓名",
    "products": [
        {{
            "name": "產品名稱",
            "quantity": 數量,
            "price": 價格
        }}
    ],
    "total_amount": 總金額,
    "order_date": "訂單日期",
    "delivery_address": "送貨地址"
}}

JSON輸出：
```json"""
    
    def event_extraction_template(self) -> str:
        """事件資訊提取模板"""
        return """從以下文字中提取事件資訊：

文字：
{input}

請提取以下資訊（JSON格式）：
{{
    "event_name": "事件名稱",
    "date": "日期",
    "time": "時間",
    "location": "地點",
    "participants": ["參與者1", "參與者2"],
    "description": "描述"
}}

JSON輸出：
```json"""
    
    def invoice_extraction_template(self) -> str:
        """發票資訊提取模板"""
        return """從以下文字中提取發票資訊：

文字：
{input}

請提取以下資訊（JSON格式）：
{{
    "invoice_number": "發票號碼",
    "date": "日期",
    "vendor": "賣方",
    "buyer": "買方",
    "items": [
        {{
            "description": "項目描述",
            "quantity": 數量,
            "unit_price": 單價,
            "amount": 金額
        }}
    ],
    "subtotal": 小計,
    "tax": 稅額,
    "total": 總計
}}

JSON輸出：
```json"""
    
    def resume_extraction_template(self) -> str:
        """履歷資訊提取模板"""
        return """從以下文字中提取履歷資訊：

文字：
{input}

請提取以下資訊（JSON格式）：
{{
    "name": "姓名",
    "email": "電子郵件",
    "phone": "電話",
    "education": [
        {{
            "degree": "學位",
            "school": "學校",
            "year": "年份"
        }}
    ],
    "experience": [
        {{
            "position": "職位",
            "company": "公司",
            "duration": "期間"
        }}
    ],
    "skills": ["技能1", "技能2"]
}}

JSON輸出：
```json"""
    
    def sentiment_classification_template(self) -> str:
        """情感分類模板"""
        return """分析以下文字的情感：

文字：{input}

請分類為以下其中一種：
- 正面 (positive)
- 負面 (negative)
- 中性 (neutral)

同時給出信心分數 (0-1)

輸出格式：
{{
    "sentiment": "分類結果",
    "confidence": 信心分數,
    "reason": "簡短理由"
}}

JSON輸出：
```json"""
    
    def intent_classification_template(self) -> str:
        """意圖分類模板"""
        return """判斷用戶意圖：

用戶輸入：{input}

可能的意圖：
- 詢問 (inquiry)
- 投訴 (complaint)
- 請求協助 (support)
- 購買 (purchase)
- 取消 (cancel)
- 其他 (other)

輸出格式：
{{
    "intent": "意圖類別",
    "confidence": 信心分數,
    "entities": ["相關實體"]
}}

JSON輸出：
```json"""
    
    def priority_classification_template(self) -> str:
        """優先級分類模板"""
        return """判斷以下請求的優先級：

請求內容：{input}

優先級別：
- 緊急 (urgent)
- 高 (high)
- 中 (medium)
- 低 (low)

考慮因素：時間敏感性、影響範圍、關鍵詞

輸出格式：
{{
    "priority": "優先級",
    "factors": ["判斷因素"],
    "suggested_response_time": "建議回應時間"
}}

JSON輸出：
```json"""
    
    def category_classification_template(self) -> str:
        """類別分類模板"""
        return """將以下內容分類：

內容：{input}

類別選項：
{categories}

輸出格式：
{{
    "primary_category": "主要類別",
    "secondary_categories": ["次要類別"],
    "confidence": 信心分數
}}

JSON輸出：
```json"""
    
    def language_detection_template(self) -> str:
        """語言檢測模板"""
        return """檢測文字語言：

文字：{input}

輸出格式：
{{
    "primary_language": "主要語言",
    "language_code": "語言代碼",
    "confidence": 信心分數,
    "mixed_languages": ["其他檢測到的語言"]
}}

JSON輸出：
```json"""
    
    def email_validation_template(self) -> str:
        """電子郵件驗證模板"""
        return """驗證電子郵件地址：

輸入：{input}

檢查項目：
1. 格式是否正確
2. 域名是否合理
3. 是否包含特殊字符

輸出格式：
{{
    "valid": true/false,
    "email": "標準化的電子郵件",
    "issues": ["問題列表"]
}}

JSON輸出：
```json"""
    
    def phone_validation_template(self) -> str:
        """電話號碼驗證模板"""
        return """驗證電話號碼：

輸入：{input}

輸出格式：
{{
    "valid": true/false,
    "formatted": "格式化的電話號碼",
    "country": "國家/地區",
    "type": "類型(手機/固話)",
    "issues": ["問題列表"]
}}

JSON輸出：
```json"""
    
    def data_quality_template(self) -> str:
        """資料品質檢查模板"""
        return """檢查資料品質：

資料：{input}

檢查項目：
- 完整性
- 一致性
- 準確性
- 格式正確性

輸出格式：
{{
    "quality_score": 品質分數(0-100),
    "completeness": 完整性分數,
    "consistency": 一致性分數,
    "issues": ["發現的問題"],
    "suggestions": ["改進建議"]
}}

JSON輸出：
```json"""
    
    def date_transform_template(self) -> str:
        """日期格式轉換模板"""
        return """轉換日期格式：

輸入：{input}
目標格式：{target_format}

輸出格式：
{{
    "original": "原始日期",
    "formatted": "格式化日期",
    "iso_format": "ISO格式",
    "timestamp": Unix時間戳
}}

JSON輸出：
```json"""
    
    def currency_transform_template(self) -> str:
        """貨幣轉換模板"""
        return """轉換貨幣格式：

輸入：{input}

輸出格式：
{{
    "amount": 數值,
    "currency": "貨幣代碼",
    "formatted": "格式化顯示",
    "in_words": "文字表示"
}}

JSON輸出：
```json"""
    
    def unit_conversion_template(self) -> str:
        """單位轉換模板"""
        return """轉換單位：

輸入：{input}
目標單位：{target_unit}

輸出格式：
{{
    "original_value": 原始值,
    "original_unit": "原始單位",
    "converted_value": 轉換值,
    "target_unit": "目標單位",
    "formula": "轉換公式"
}}

JSON輸出：
```json"""
    
    def brief_summary_template(self) -> str:
        """簡短摘要模板"""
        return """生成簡短摘要（50字以內）：

原文：{input}

摘要："""
    
    def detailed_summary_template(self) -> str:
        """詳細摘要模板"""
        return """生成詳細摘要：

原文：{input}

請包含：
1. 主要觀點
2. 重要細節
3. 結論

摘要："""
    
    def bullet_points_template(self) -> str:
        """要點列表模板"""
        return """提取要點（bullet points）：

原文：{input}

要點列表：
•"""

class SmartFormProcessor:
    """智慧表單處理器主類"""
    
    def __init__(self, model: str = "gemma:2b"):
        self.model = model
        self.template_library = PromptTemplate()
        self.processing_history = []
    
    def process(self, 
                input_text: str, 
                task_type: TaskType, 
                template_name: str,
                **kwargs) -> ProcessingResult:
        """處理輸入文字"""
        
        import time
        start_time = time.time()
        
        # 獲取模板
        template = self.get_template(task_type, template_name)
        if not template:
            return ProcessingResult(
                success=False,
                task_type=task_type,
                input_text=input_text,
                output_data={},
                errors=[f"Template {template_name} not found for {task_type.value}"]
            )
        
        # 格式化 prompt
        prompt = template.format(input=input_text, **kwargs)
        
        try:
            # 調用 LLM
            response = ollama.generate(
                model=self.model,
                prompt=prompt,
                options={'temperature': 0.2}
            )
            
            # 解析回應
            output_data = self.parse_response(response['response'], task_type)
            
            processing_time = time.time() - start_time
            
            result = ProcessingResult(
                success=True,
                task_type=task_type,
                input_text=input_text,
                output_data=output_data,
                confidence=output_data.get('confidence', 0.8),
                processing_time=processing_time
            )
            
            self.processing_history.append(result)
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            
            result = ProcessingResult(
                success=False,
                task_type=task_type,
                input_text=input_text,
                output_data={},
                errors=[str(e)],
                processing_time=processing_time
            )
            
            self.processing_history.append(result)
            return result
    
    def get_template(self, task_type: TaskType, template_name: str) -> Optional[str]:
        """獲取指定模板"""
        if task_type in self.template_library.templates:
            templates = self.template_library.templates[task_type]
            if template_name in templates:
                return templates[template_name]
        return None
    
    def parse_response(self, response: str, task_type: TaskType) -> Dict[str, Any]:
        """解析 LLM 回應"""
        # 嘗試提取 JSON
        if "```json" in response:
            json_str = response.split("```json")[1].split("```")[0].strip()
        elif "{" in response and "}" in response:
            # 嘗試找到 JSON 部分
            start = response.index("{")
            end = response.rindex("}") + 1
            json_str = response[start:end]
        else:
            # 非 JSON 格式
            if task_type == TaskType.SUMMARIZE:
                return {"summary": response.strip()}
            else:
                return {"result": response.strip()}
        
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            # 如果 JSON 解析失敗，返回原始文字
            return {"raw_response": response}
    
    def batch_process(self, 
                     items: List[Tuple[str, TaskType, str]], 
                     **kwargs) -> List[ProcessingResult]:
        """批次處理多個項目"""
        results = []
        
        for i, (text, task_type, template_name) in enumerate(items, 1):
            print(f"\n處理 {i}/{len(items)}: {text[:50]}...")
            result = self.process(text, task_type, template_name, **kwargs)
            results.append(result)
            
            if result.success:
                print(f"✅ 成功")
            else:
                print(f"❌ 失敗: {result.errors}")
        
        return results
    
    def validate_form_data(self, data: Dict[str, Any], form_fields: List[FormField]) -> Dict[str, Any]:
        """驗證表單資料"""
        validation_results = {
            "valid": True,
            "errors": {},
            "warnings": {}
        }
        
        for field in form_fields:
            value = data.get(field.name)
            
            # 檢查必填欄位
            if field.required and not value:
                validation_results["valid"] = False
                validation_results["errors"][field.name] = "此欄位為必填"
                continue
            
            # 類型驗證
            if value:
                if field.field_type == "email":
                    if not re.match(r'^[\w\.-]+@[\w\.-]+\.\w+$', value):
                        validation_results["valid"] = False
                        validation_results["errors"][field.name] = "無效的電子郵件格式"
                
                elif field.field_type == "phone":
                    if not re.match(r'^[\d\-\+\(\)\s]+$', value):
                        validation_results["valid"] = False
                        validation_results["errors"][field.name] = "無效的電話號碼格式"
                
                elif field.field_type == "number":
                    try:
                        float(value)
                    except ValueError:
                        validation_results["valid"] = False
                        validation_results["errors"][field.name] = "必須是數字"
                
                elif field.field_type == "select":
                    if value not in field.options:
                        validation_results["valid"] = False
                        validation_results["errors"][field.name] = f"必須是以下選項之一: {field.options}"
        
        return validation_results
    
    def export_history(self, filename: str = "processing_history.json") -> None:
        """匯出處理歷史"""
        history_data = []
        
        for result in self.processing_history:
            history_data.append({
                "success": result.success,
                "task_type": result.task_type.value,
                "input": result.input_text[:100],
                "output": result.output_data,
                "errors": result.errors,
                "confidence": result.confidence,
                "processing_time": result.processing_time,
                "timestamp": datetime.now().isoformat()
            })
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(history_data, f, ensure_ascii=False, indent=2)
        
        print(f"✅ 處理歷史已匯出到 {filename}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """獲取處理統計"""
        if not self.processing_history:
            return {}
        
        total = len(self.processing_history)
        successful = sum(1 for r in self.processing_history if r.success)
        
        task_counts = {}
        for result in self.processing_history:
            task_type = result.task_type.value
            task_counts[task_type] = task_counts.get(task_type, 0) + 1
        
        avg_time = sum(r.processing_time for r in self.processing_history) / total
        avg_confidence = sum(r.confidence for r in self.processing_history if r.success) / max(successful, 1)
        
        return {
            "total_processed": total,
            "successful": successful,
            "success_rate": successful / total,
            "task_distribution": task_counts,
            "average_processing_time": avg_time,
            "average_confidence": avg_confidence
        }

def demo_scenarios():
    """示範不同場景"""
    processor = SmartFormProcessor()
    
    print("=" * 70)
    print("🎯 智慧表單處理器 - 場景示範")
    print("=" * 70)
    
    # 場景1：聯絡資訊提取
    print("\n📚 場景1：從非結構化文字提取聯絡資訊")
    print("-" * 60)
    
    text1 = """
    您好，我是張明華，在台積電擔任資深工程師。
    我的電子郵件是 ming.zhang@tsmc.com，
    手機號碼是 0912-345-678。
    公司地址在新竹科學園區力行六路8號。
    """
    
    result1 = processor.process(text1, TaskType.EXTRACT, "contact_info")
    if result1.success:
        print("✅ 提取成功：")
        print(json.dumps(result1.output_data, ensure_ascii=False, indent=2))
    
    # 場景2：客戶意圖分類
    print("\n📚 場景2：客戶服務意圖分類")
    print("-" * 60)
    
    text2 = "我上週買的產品有問題，一直無法開機，要求退貨退款！"
    
    result2 = processor.process(text2, TaskType.CLASSIFY, "intent")
    if result2.success:
        print("✅ 分類結果：")
        print(json.dumps(result2.output_data, ensure_ascii=False, indent=2))
    
    # 場景3：訂單資訊提取
    print("\n📚 場景3：訂單資訊提取")
    print("-" * 60)
    
    text3 = """
    訂單編號 ORD-2024-001234
    客戶：王小明
    購買項目：iPhone 15 Pro x1 (NT$38,900)、AirPods Pro x2 (NT$7,990 each)
    總金額：NT$54,880
    送貨地址：台北市信義區信義路五段7號
    訂購日期：2024年1月15日
    """
    
    result3 = processor.process(text3, TaskType.EXTRACT, "order_info")
    if result3.success:
        print("✅ 訂單資訊：")
        print(json.dumps(result3.output_data, ensure_ascii=False, indent=2))
    
    # 場景4：優先級判斷
    print("\n📚 場景4：支援請求優先級判斷")
    print("-" * 60)
    
    requests = [
        "系統完全當機，所有用戶無法登入！",
        "想了解產品的詳細規格",
        "發票金額好像有誤差，請協助確認"
    ]
    
    for req in requests:
        result = processor.process(req, TaskType.CLASSIFY, "priority")
        if result.success:
            print(f"\n請求: {req}")
            print(f"優先級: {result.output_data.get('priority', 'unknown')}")
    
    # 場景5：批次處理情感分析
    print("\n📚 場景5：批次情感分析")
    print("-" * 60)
    
    reviews = [
        ("產品很棒，超出預期！", TaskType.CLASSIFY, "sentiment"),
        ("普普通通，沒什麼特別", TaskType.CLASSIFY, "sentiment"),
        ("完全是垃圾，浪費錢", TaskType.CLASSIFY, "sentiment")
    ]
    
    results = processor.batch_process(reviews)
    
    print("\n情感分析結果：")
    for i, result in enumerate(results, 1):
        if result.success:
            sentiment = result.output_data.get('sentiment', 'unknown')
            confidence = result.output_data.get('confidence', 0)
            print(f"{i}. {sentiment} (信心度: {confidence:.2f})")
    
    # 顯示統計
    print("\n" + "=" * 70)
    print("📊 處理統計")
    print("=" * 70)
    
    stats = processor.get_statistics()
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    
    # 匯出歷史
    processor.export_history("demo_history.json")

def interactive_mode():
    """互動模式"""
    processor = SmartFormProcessor()
    
    print("=" * 70)
    print("🤖 智慧表單處理器 - 互動模式")
    print("=" * 70)
    
    while True:
        print("\n選擇任務類型：")
        print("1. 資訊提取 (Extract)")
        print("2. 文字分類 (Classify)")
        print("3. 資料驗證 (Validate)")
        print("4. 格式轉換 (Transform)")
        print("5. 摘要生成 (Summarize)")
        print("6. 查看統計")
        print("7. 匯出歷史")
        print("0. 結束")
        
        choice = input("\n選擇 (0-7): ").strip()
        
        if choice == '0':
            break
        
        elif choice in ['1', '2', '3', '4', '5']:
            task_type_map = {
                '1': TaskType.EXTRACT,
                '2': TaskType.CLASSIFY,
                '3': TaskType.VALIDATE,
                '4': TaskType.TRANSFORM,
                '5': TaskType.SUMMARIZE
            }
            
            task_type = task_type_map[choice]
            
            # 顯示可用模板
            templates = processor.template_library.templates.get(task_type, {})
            print(f"\n可用模板：")
            for i, template_name in enumerate(templates.keys(), 1):
                print(f"{i}. {template_name}")
            
            template_idx = input("選擇模板編號: ").strip()
            try:
                template_name = list(templates.keys())[int(template_idx) - 1]
            except (ValueError, IndexError):
                print("❌ 無效的選擇")
                continue
            
            # 輸入文字
            print("\n輸入要處理的文字（輸入空行結束）：")
            lines = []
            while True:
                line = input()
                if not line:
                    break
                lines.append(line)
            
            text = "\n".join(lines)
            
            if text:
                result = processor.process(text, task_type, template_name)
                
                if result.success:
                    print("\n✅ 處理成功：")
                    print(json.dumps(result.output_data, ensure_ascii=False, indent=2))
                    print(f"\n處理時間: {result.processing_time:.2f} 秒")
                else:
                    print(f"\n❌ 處理失敗: {result.errors}")
        
        elif choice == '6':
            stats = processor.get_statistics()
            if stats:
                print("\n📊 處理統計：")
                print(json.dumps(stats, ensure_ascii=False, indent=2))
            else:
                print("\n尚無處理記錄")
        
        elif choice == '7':
            filename = input("輸入檔案名稱 [預設: history.json]: ").strip() or "history.json"
            processor.export_history(filename)

def main():
    """主程式"""
    print("=" * 70)
    print("🎓 Week 2 Lab - 智慧表單處理器")
    print("=" * 70)
    
    # 檢查 Ollama
    try:
        models = ollama.list()
        print("✅ 已連接到 Ollama")
    except Exception as e:
        print(f"❌ 無法連接到 Ollama: {e}")
        return
    
    print("\n選擇執行模式：")
    print("1. 示範場景")
    print("2. 互動模式")
    
    choice = input("\n選擇 (1-2): ").strip()
    
    if choice == '1':
        demo_scenarios()
    elif choice == '2':
        interactive_mode()
    else:
        print("❌ 無效的選擇")
    
    print("\n👋 課程結束！")

if __name__ == "__main__":
    main()