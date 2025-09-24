#!/usr/bin/env python3
"""
Week 5 - Lesson 1: LangChain 基礎與鏈 (Chains)
學習 LangChain 的核心概念與基本使用
"""

from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_core.output_parsers import StrOutputParser
import json
from typing import Dict, Any

def example1_first_langchain():
    """範例1: 第一個 LangChain 應用"""
    print("\n範例1: 第一個 LangChain 應用")
    print("=" * 50)

    # 初始化 Ollama LLM
    llm = Ollama(model="gemma3:1b")

    # 直接呼叫 LLM
    response = llm.invoke("什麼是 LangChain？用一句話說明。")
    print(f"直接呼叫 LLM:\n{response}")

    # 使用 invoke 方法（新版推薦）
    response2 = llm.invoke("LangChain 的主要功能是什麼？")
    print(f"\n使用 invoke 方法:\n{response2}")

def example2_prompt_and_chain():
    """範例2: 提示模板與鏈"""
    print("\n範例2: 提示模板與鏈")
    print("=" * 50)

    # 建立提示模板
    prompt = PromptTemplate(
        input_variables=["product", "feature"],
        template="""你是一個產品經理。
請為以下產品撰寫一個強調{feature}的行銷標語：

產品：{product}
標語："""
    )

    # 建立 LLM
    llm = Ollama(model="gemma3:1b")

    # 建立鏈
    chain = LLMChain(llm=llm, prompt=prompt)

    # 執行鏈
    result = chain.invoke({
        "product": "智慧手錶",
        "feature": "健康監測"
    })

    print(f"輸入：產品=智慧手錶, 特點=健康監測")
    print(f"輸出：{result['text']}")

    # 測試不同的輸入
    result2 = chain.invoke({
        "product": "無線耳機",
        "feature": "降噪功能"
    })

    print(f"\n輸入：產品=無線耳機, 特點=降噪功能")
    print(f"輸出：{result2['text']}")

def example3_simple_sequential_chain():
    """範例3: 簡單的順序鏈"""
    print("\n範例3: 簡單的順序鏈")
    print("=" * 50)

    llm = Ollama(model="gemma3:1b")

    # 第一個鏈：生成公司名稱
    name_prompt = PromptTemplate(
        input_variables=["industry"],
        template="為一家{industry}產業的新創公司取一個有創意的中文名稱："
    )
    name_chain = LLMChain(llm=llm, prompt=name_prompt)

    # 第二個鏈：生成公司標語
    slogan_prompt = PromptTemplate(
        input_variables=["company_name"],
        template="為「{company_name}」這家公司創造一個吸引人的標語："
    )
    slogan_chain = LLMChain(llm=llm, prompt=slogan_prompt)

    # 手動串接兩個鏈
    industry = "環保科技"
    print(f"產業：{industry}")

    # 執行第一個鏈
    company_name = name_chain.invoke({"industry": industry})['text'].strip()
    print(f"生成的公司名稱：{company_name}")

    # 執行第二個鏈
    slogan = slogan_chain.invoke({"company_name": company_name})['text'].strip()
    print(f"生成的標語：{slogan}")

def example4_chain_with_parser():
    """範例4: 使用輸出解析器的鏈"""
    print("\n範例4: 使用輸出解析器的鏈")
    print("=" * 50)

    llm = Ollama(model="gemma3:1b")

    # 建立提示模板
    prompt = PromptTemplate(
        input_variables=["topic"],
        template="""列出關於{topic}的3個重點，每個重點用一句話說明。
請按照以下格式輸出：
1. 第一個重點
2. 第二個重點
3. 第三個重點

重點列表："""
    )

    # 建立輸出解析器
    parser = StrOutputParser()

    # 使用 | 操作符建立鏈（LCEL語法）
    chain = prompt | llm | parser

    # 執行鏈
    result = chain.invoke({"topic": "電子商務"})

    print(f"主題：電子商務")
    print(f"生成的重點：\n{result}")

    # 解析輸出為列表
    lines = [line.strip() for line in result.split('\n') if line.strip()]
    print("\n解析後的重點列表：")
    for line in lines[:3]:  # 只取前3行
        print(f"  - {line}")

def example5_custom_chain_function():
    """範例5: 自定義鏈函數"""
    print("\n範例5: 自定義鏈函數")
    print("=" * 50)

    class ProductAnalyzer:
        """產品分析鏈"""

        def __init__(self, model="gemma3:1b"):
            self.llm = Ollama(model=model)

            # 優點分析鏈
            self.pros_prompt = PromptTemplate(
                input_variables=["product"],
                template="列出{product}的三個主要優點："
            )

            # 缺點分析鏈
            self.cons_prompt = PromptTemplate(
                input_variables=["product"],
                template="列出{product}的三個主要缺點："
            )

            # 建議鏈
            self.suggestion_prompt = PromptTemplate(
                input_variables=["product", "pros", "cons"],
                template="""根據以下分析，給出{product}的改進建議：
優點：{pros}
缺點：{cons}
改進建議："""
            )

        def analyze(self, product: str) -> Dict[str, Any]:
            """分析產品"""
            # 獲取優點
            pros_chain = LLMChain(llm=self.llm, prompt=self.pros_prompt)
            pros = pros_chain.invoke({"product": product})['text']

            # 獲取缺點
            cons_chain = LLMChain(llm=self.llm, prompt=self.cons_prompt)
            cons = cons_chain.invoke({"product": product})['text']

            # 生成建議
            suggestion_chain = LLMChain(llm=self.llm, prompt=self.suggestion_prompt)
            suggestion = suggestion_chain.invoke({
                "product": product,
                "pros": pros,
                "cons": cons
            })['text']

            return {
                "product": product,
                "pros": pros,
                "cons": cons,
                "suggestion": suggestion
            }

    # 使用產品分析器
    analyzer = ProductAnalyzer()
    result = analyzer.analyze("電動汽車")

    print("產品分析報告：電動汽車")
    print("-" * 40)
    print(f"優點：\n{result['pros']}")
    print(f"\n缺點：\n{result['cons']}")
    print(f"\n改進建議：\n{result['suggestion']}")

def example6_error_handling():
    """範例6: 錯誤處理與重試"""
    print("\n範例6: 錯誤處理與重試")
    print("=" * 50)

    from time import sleep

    def safe_chain_invoke(chain, inputs, max_retries=3):
        """安全執行鏈，包含重試機制"""
        for attempt in range(max_retries):
            try:
                result = chain.invoke(inputs)
                return result
            except Exception as e:
                print(f"  嘗試 {attempt + 1} 失敗: {str(e)}")
                if attempt < max_retries - 1:
                    print(f"  等待 2 秒後重試...")
                    sleep(2)
                else:
                    print(f"  已達最大重試次數")
                    return None

    # 建立鏈
    llm = Ollama(model="gemma3:1b", temperature=0.7)
    prompt = PromptTemplate(
        input_variables=["question"],
        template="回答這個問題：{question}"
    )
    chain = LLMChain(llm=llm, prompt=prompt)

    # 測試錯誤處理
    result = safe_chain_invoke(
        chain,
        {"question": "什麼是雲端運算？"}
    )

    if result:
        print(f"成功獲得回答：\n{result['text'][:200]}...")
    else:
        print("無法獲得回答")

def main():
    """主程式"""
    print("="*60)
    print("Week 5 - Lesson 1: LangChain 基礎與鏈")
    print("="*60)

    # 檢查 Ollama
    try:
        test_llm = Ollama(model="gemma3:1b")
        test_llm.invoke("測試")
        print("✓ Ollama 連接成功")
    except Exception as e:
        print(f"✗ 請先啟動 Ollama: ollama serve")
        print(f"  錯誤: {e}")
        return

    # 執行所有範例
    example1_first_langchain()
    example2_prompt_and_chain()
    example3_simple_sequential_chain()
    example4_chain_with_parser()
    example5_custom_chain_function()
    example6_error_handling()

    print("\n" + "="*60)
    print("課程重點總結：")
    print("1. LangChain 提供了結構化的方式來建立 LLM 應用")
    print("2. 提示模板讓我們可以重複使用和管理提示詞")
    print("3. 鏈（Chain）可以串接多個操作")
    print("4. LCEL (LangChain Expression Language) 提供了優雅的鏈組合方式")
    print("5. 適當的錯誤處理確保應用的穩定性")
    print("="*60)

if __name__ == "__main__":
    main()