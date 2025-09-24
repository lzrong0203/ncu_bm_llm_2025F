#!/usr/bin/env python3
"""
Week 5 - Lesson 2: 提示模板與輸出解析器
深入學習 LangChain 的提示工程與結構化輸出
"""

from langchain_community.llms import Ollama
from langchain_core.prompts import (
    PromptTemplate,
    ChatPromptTemplate,
    FewShotPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)
from langchain_core.output_parsers import (
    StrOutputParser,
    JsonOutputParser,
    PydanticOutputParser,
    CommaSeparatedListOutputParser
)
from langchain.chains import LLMChain
from pydantic import BaseModel, Field
from typing import List, Dict, Any
import json

def example1_basic_prompt_template():
    """範例1: 基本提示模板"""
    print("\n範例1: 基本提示模板")
    print("=" * 50)

    # 簡單模板
    simple_template = PromptTemplate(
        input_variables=["subject"],
        template="解釋什麼是{subject}，用簡單易懂的方式。"
    )

    # 複雜模板with多個變數
    complex_template = PromptTemplate(
        input_variables=["role", "task", "context", "format"],
        template="""你是一個{role}。

任務：{task}

背景資訊：
{context}

請按照以下格式回答：
{format}

回答："""
    )

    llm = Ollama(model="gemma3:1b")

    # 測試簡單模板
    simple_chain = LLMChain(llm=llm, prompt=simple_template)
    result1 = simple_chain.invoke({"subject": "區塊鏈"})
    print("簡單模板結果：")
    print(result1['text'][:200] + "...")

    # 測試複雜模板
    complex_chain = LLMChain(llm=llm, prompt=complex_template)
    result2 = complex_chain.invoke({
        "role": "資深財務顧問",
        "task": "分析投資風險",
        "context": "客戶想投資加密貨幣，資金10萬元",
        "format": "1. 風險評估 2. 建議配置 3. 注意事項"
    })
    print("\n複雜模板結果：")
    print(result2['text'][:300] + "...")

def example2_few_shot_template():
    """範例2: Few-Shot 提示模板"""
    print("\n範例2: Few-Shot 提示模板")
    print("=" * 50)

    # 定義範例
    examples = [
        {
            "product": "筆記型電腦",
            "features": "輕薄、高效能、長續航",
            "tagline": "輕盈隨行，效能不妥協"
        },
        {
            "product": "智慧手錶",
            "features": "健康監測、運動追蹤、訊息提醒",
            "tagline": "你的健康，腕上掌握"
        },
        {
            "product": "無線耳機",
            "features": "主動降噪、高音質、舒適配戴",
            "tagline": "沉浸音樂，隔絕喧囂"
        }
    ]

    # 範例模板
    example_template = """
產品：{product}
特點：{features}
標語：{tagline}"""

    # 建立範例提示
    example_prompt = PromptTemplate(
        input_variables=["product", "features", "tagline"],
        template=example_template
    )

    # Few-shot 提示模板
    few_shot_prompt = FewShotPromptTemplate(
        examples=examples,
        example_prompt=example_prompt,
        prefix="你是一個廣告文案專家。根據產品特點創造吸引人的標語。\n以下是一些範例：",
        suffix="\n產品：{product}\n特點：{features}\n標語：",
        input_variables=["product", "features"]
    )

    # 測試
    llm = Ollama(model="gemma3:1b")
    chain = LLMChain(llm=llm, prompt=few_shot_prompt)

    result = chain.invoke({
        "product": "智慧門鎖",
        "features": "指紋辨識、遠端控制、訪客記錄"
    })

    print(f"新產品：智慧門鎖")
    print(f"特點：指紋辨識、遠端控制、訪客記錄")
    print(f"生成的標語：{result['text'].strip()}")

def example3_output_parser_basics():
    """範例3: 基本輸出解析器"""
    print("\n範例3: 基本輸出解析器")
    print("=" * 50)

    llm = Ollama(model="gemma3:1b")

    # 1. 字串解析器
    str_parser = StrOutputParser()

    # 2. 逗號分隔列表解析器
    list_parser = CommaSeparatedListOutputParser()

    # 逗號分隔列表範例
    list_prompt = PromptTemplate(
        template="列出5個{category}的例子，用逗號分隔：",
        input_variables=["category"]
    )

    list_chain = list_prompt | llm | list_parser

    try:
        items = list_chain.invoke({"category": "程式語言"})
        print("程式語言列表：")
        for i, item in enumerate(items, 1):
            print(f"  {i}. {item}")
    except Exception as e:
        print(f"列表解析錯誤: {e}")

def example4_json_output_parser():
    """範例4: JSON 輸出解析器"""
    print("\n範例4: JSON 輸出解析器")
    print("=" * 50)

    # 定義 Pydantic 模型
    class Product(BaseModel):
        name: str = Field(description="產品名稱")
        price: int = Field(description="價格（新台幣）")
        category: str = Field(description="產品類別")
        in_stock: bool = Field(description="是否有庫存")
        rating: float = Field(description="評分（1-5分）")

    # 建立 JSON 解析器
    parser = JsonOutputParser(pydantic_object=Product)

    # 建立提示模板
    prompt = PromptTemplate(
        template="""根據以下描述，提取產品資訊並以JSON格式輸出。

{format_instructions}

產品描述：{description}

JSON輸出：""",
        input_variables=["description"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )

    llm = Ollama(model="gemma3:1b", temperature=0)
    chain = prompt | llm | parser

    # 測試
    description = "AirPods Pro 是Apple的高階無線耳機，售價7990元，屬於音訊配件類別，目前有現貨，用戶評分4.5分"

    try:
        result = chain.invoke({"description": description})
        print("解析的產品資訊：")
        print(json.dumps(result, indent=2, ensure_ascii=False))
    except Exception as e:
        print(f"JSON解析錯誤: {e}")
        # 降級處理
        simple_chain = prompt | llm
        raw_result = simple_chain.invoke({"description": description})
        print("原始輸出：")
        print(raw_result)

def example5_pydantic_parser():
    """範例5: Pydantic 解析器（結構化輸出）"""
    print("\n範例5: Pydantic 解析器")
    print("=" * 50)

    # 定義客戶資料模型
    class CustomerFeedback(BaseModel):
        customer_name: str = Field(description="客戶姓名")
        product: str = Field(description="產品名稱")
        rating: int = Field(description="評分1-5", ge=1, le=5)
        sentiment: str = Field(description="情感：正面/中性/負面")
        key_points: List[str] = Field(description="主要意見點")
        needs_followup: bool = Field(description="是否需要跟進")

    # 建立 Pydantic 解析器
    parser = PydanticOutputParser(pydantic_object=CustomerFeedback)

    # 建立提示
    prompt = PromptTemplate(
        template="""分析以下客戶評論，提取結構化資訊。

{format_instructions}

客戶評論：
{review}

結構化分析：""",
        input_variables=["review"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )

    llm = Ollama(model="gemma3:1b", temperature=0)
    chain = prompt | llm | parser

    # 測試評論
    review = """
    張先生購買了我們的智慧手錶，給了4分評價。
    他說："手錶功能很齊全，特別是健康監測非常準確。
    但是電池續航有點短，希望能改進。另外，錶帶材質可以更好一些。
    整體來說還是很滿意的，會推薦給朋友。"
    """

    try:
        result = chain.invoke({"review": review})
        print("客戶反饋分析：")
        print(f"客戶：{result.customer_name}")
        print(f"產品：{result.product}")
        print(f"評分：{result.rating}/5")
        print(f"情感：{result.sentiment}")
        print(f"主要意見：")
        for point in result.key_points:
            print(f"  - {point}")
        print(f"需要跟進：{'是' if result.needs_followup else '否'}")
    except Exception as e:
        print(f"Pydantic解析錯誤: {e}")

def example6_custom_parser():
    """範例6: 自定義解析器"""
    print("\n範例6: 自定義解析器")
    print("=" * 50)

    class TableParser:
        """表格解析器"""

        def parse(self, text: str) -> List[Dict[str, Any]]:
            """解析表格格式的文字"""
            lines = text.strip().split('\n')
            if len(lines) < 2:
                return []

            # 假設第一行是標題
            headers = [h.strip() for h in lines[0].split('|') if h.strip()]

            # 解析資料行
            data = []
            for line in lines[1:]:
                if '|' in line:
                    values = [v.strip() for v in line.split('|') if v.strip()]
                    if len(values) == len(headers):
                        row = dict(zip(headers, values))
                        data.append(row)

            return data

    # 建立提示
    prompt = PromptTemplate(
        template="""列出前3名{category}，用表格格式呈現。
包含以下欄位：排名、名稱、特點

請使用以下格式：
排名 | 名稱 | 特點
1 | xxx | xxx
2 | xxx | xxx
3 | xxx | xxx

{category}排行榜：""",
        input_variables=["category"]
    )

    llm = Ollama(model="gemma3:1b", temperature=0)
    chain = prompt | llm

    # 執行並解析
    result = chain.invoke({"category": "程式語言"})
    print("原始輸出：")
    print(result)

    # 使用自定義解析器
    parser = TableParser()
    parsed_data = parser.parse(result)

    print("\n解析後的資料：")
    for row in parsed_data:
        print(f"  {row}")

def example7_chain_with_validation():
    """範例7: 帶驗證的輸出鏈"""
    print("\n範例7: 帶驗證的輸出鏈")
    print("=" * 50)

    class EmailGenerator(BaseModel):
        subject: str = Field(description="郵件主旨", min_length=5, max_length=100)
        greeting: str = Field(description="問候語")
        body: str = Field(description="郵件內容", min_length=20)
        closing: str = Field(description="結尾敬語")
        tone: str = Field(description="語氣：正式/友善/專業")

    parser = PydanticOutputParser(pydantic_object=EmailGenerator)

    prompt = PromptTemplate(
        template="""撰寫一封商業郵件。

{format_instructions}

情境：{scenario}

請生成郵件：""",
        input_variables=["scenario"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )

    llm = Ollama(model="gemma3:1b", temperature=0.3)

    def safe_parse_email(chain, scenario, max_retries=2):
        """安全解析郵件，包含重試和驗證"""
        for attempt in range(max_retries):
            try:
                raw_output = chain.invoke({"scenario": scenario})
                parsed = parser.parse(raw_output)

                # 額外驗證
                if len(parsed.subject) < 5:
                    raise ValueError("主旨太短")
                if len(parsed.body) < 20:
                    raise ValueError("內容太短")

                return parsed
            except Exception as e:
                print(f"  解析嘗試 {attempt + 1} 失敗: {e}")
                if attempt == max_retries - 1:
                    # 返回預設值
                    return EmailGenerator(
                        subject="商業郵件",
                        greeting="您好",
                        body="感謝您的來信，我們會盡快回覆。",
                        closing="謝謝",
                        tone="專業"
                    )

    chain = prompt | llm

    scenario = "通知客戶訂單已出貨，訂單編號#12345，預計3天內送達"

    email = safe_parse_email(chain, scenario)

    print("生成的郵件：")
    print(f"主旨：{email.subject}")
    print(f"語氣：{email.tone}")
    print(f"\n{email.greeting}")
    print(f"\n{email.body}")
    print(f"\n{email.closing}")

def main():
    """主程式"""
    print("="*60)
    print("Week 5 - Lesson 2: 提示模板與輸出解析器")
    print("="*60)

    # 檢查 Ollama
    try:
        test_llm = Ollama(model="gemma3:1b")
        test_llm.invoke("測試")
        print("✓ Ollama 連接成功\n")
    except Exception as e:
        print(f"✗ 請先啟動 Ollama: ollama serve")
        return

    # 執行範例
    example1_basic_prompt_template()
    example2_few_shot_template()
    example3_output_parser_basics()
    example4_json_output_parser()
    example5_pydantic_parser()
    example6_custom_parser()
    example7_chain_with_validation()

    print("\n" + "="*60)
    print("課程重點總結：")
    print("1. 提示模板提供了管理和重用提示詞的結構化方法")
    print("2. Few-Shot 模板通過範例改善模型輸出品質")
    print("3. 輸出解析器確保獲得結構化的資料")
    print("4. Pydantic 模型提供類型安全和驗證")
    print("5. 自定義解析器可處理特殊格式需求")
    print("6. 錯誤處理和驗證確保系統穩定性")
    print("="*60)

if __name__ == "__main__":
    main()