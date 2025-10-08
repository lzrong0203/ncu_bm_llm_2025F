#!/usr/bin/env python3
"""
Week 5 - Lesson 4: LangChain + Ollama 深度整合
探索 Ollama 與 LangChain 的進階功能
"""

from langchain_community.llms import Ollama
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import LLMChain, ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.base import BaseCallbackHandler
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import time
import asyncio
from typing import List, Dict, Any, Optional
import json

def example1_ollama_vs_chatollama():
    """範例1: Ollama vs ChatOllama 的差異"""
    print("\n範例1: Ollama vs ChatOllama 的差異")
    print("=" * 50)

    prompt_text = "解釋什麼是機器學習，用50字以內"

    # 使用 Ollama (基本 LLM)
    print("使用 Ollama (基本 LLM):")
    llm = Ollama(model="gemma3:1b")
    response1 = llm.invoke(prompt_text)
    print(f"回應: {response1}\n")

    # 使用 ChatOllama (對話模型)
    print("使用 ChatOllama (對話模型):")
    chat_model = ChatOllama(model="gemma3:1b")

    messages = [
        SystemMessage(content="你是一個簡潔的AI助手"),
        HumanMessage(content=prompt_text)
    ]

    response2 = chat_model.invoke(messages)
    print(f"回應: {response2.content}\n")

    # 比較特性
    print("主要差異：")
    print("1. Ollama: 接受字串，返回字串")
    print("2. ChatOllama: 接受訊息列表，返回訊息物件")
    print("3. ChatOllama: 更適合多輪對話和角色扮演")

def example2_model_parameters():
    """範例2: 模型參數調整"""
    print("\n範例2: 模型參數調整")
    print("=" * 50)

    base_prompt = "寫一個關於AI的短句"

    # 不同溫度設定
    temperatures = [0, 0.5, 1.0]

    for temp in temperatures:
        print(f"\nTemperature = {temp}:")
        llm = Ollama(
            model="gemma3:1b",
            temperature=temp,
            top_p=0.9,
            num_predict=50  # 最大生成長度
        )
        response = llm.invoke(base_prompt)
        print(f"  {response}")

    # 其他參數示例
    print("\n進階參數設定:")
    llm_advanced = Ollama(
        model="gemma3:1b",
        temperature=0.7,
        top_k=40,           # Top-K 取樣
        top_p=0.9,          # Top-P 取樣
        repeat_penalty=1.1,  # 重複懲罰
        num_ctx=2048,       # 上下文長度
        num_predict=100,    # 預測長度
        seed=42            # 隨機種子（可重現結果）
    )

    response = llm_advanced.invoke("列出3個使用AI的好處")
    print(f"進階設定結果:\n{response}")

def example3_streaming_output():
    """範例3: 串流輸出"""
    print("\n範例3: 串流輸出")
    print("=" * 50)

    # 自定義串流處理器
    class CustomStreamHandler(BaseCallbackHandler):
        def __init__(self):
            self.tokens = []
            self.start_time = None

        def on_llm_start(self, *args, **kwargs):
            self.start_time = time.time()
            print("開始生成...\n", end="")

        def on_llm_new_token(self, token: str, **kwargs):
            print(token, end="", flush=True)
            self.tokens.append(token)

        def on_llm_end(self, *args, **kwargs):
            elapsed = time.time() - self.start_time
            print(f"\n\n生成完成！耗時: {elapsed:.2f}秒")
            print(f"總共 {len(self.tokens)} 個 token")

    # 使用串流
    print("串流輸出示例：")
    llm_stream = Ollama(
        model="gemma3:1b",
        callbacks=[CustomStreamHandler()],
        streaming=True
    )

    llm_stream.invoke("寫一個關於未來科技的短故事（100字）")

def example4_multi_model_comparison():
    """範例4: 多模型比較"""
    print("\n範例4: 多模型比較")
    print("=" * 50)

    class ModelComparator:
        """模型比較器"""

        def __init__(self, models: List[str]):
            self.models = {}
            for model_name in models:
                try:
                    self.models[model_name] = Ollama(
                        model=model_name,
                        temperature=0.7
                    )
                    print(f"✓ 載入模型: {model_name}")
                except Exception as e:
                    print(f"✗ 無法載入 {model_name}: {e}")

        def compare(self, prompt: str) -> Dict[str, Any]:
            """比較不同模型的回應"""
            results = {}

            for model_name, llm in self.models.items():
                start_time = time.time()
                try:
                    response = llm.invoke(prompt)
                    elapsed = time.time() - start_time

                    results[model_name] = {
                        "response": response,
                        "time": elapsed,
                        "length": len(response)
                    }
                except Exception as e:
                    results[model_name] = {
                        "error": str(e)
                    }

            return results

        def print_comparison(self, prompt: str):
            """列印比較結果"""
            print(f"提示詞: {prompt}\n")
            results = self.compare(prompt)

            for model_name, result in results.items():
                print(f"\n模型: {model_name}")
                if "error" in result:
                    print(f"  錯誤: {result['error']}")
                else:
                    print(f"  耗時: {result['time']:.2f}秒")
                    print(f"  長度: {result['length']}字")
                    print(f"  回應: {result['response'][:100]}...")

    # 比較不同大小的模型
    available_models = ["gemma3:270m", "gemma3:1b"]  # 根據實際安裝調整

    comparator = ModelComparator(available_models)
    comparator.print_comparison("什麼是深度學習？用一句話解釋")

def example5_async_operations():
    """範例5: 非同步操作"""
    print("\n範例5: 非同步操作")
    print("=" * 50)

    async def async_generate(llm, prompt: str, model_name: str):
        """非同步生成"""
        start = time.time()
        # 注意：目前 Ollama 的非同步支援有限，這裡模擬非同步操作
        response = await asyncio.get_event_loop().run_in_executor(
            None, llm.invoke, prompt
        )
        elapsed = time.time() - start
        return {
            "model": model_name,
            "response": response,
            "time": elapsed
        }

    async def parallel_generation():
        """並行生成多個回應"""
        prompts = [
            "解釋什麼是雲端運算",
            "解釋什麼是物聯網",
            "解釋什麼是區塊鏈"
        ]

        llm = Ollama(model="gemma3:1b")

        tasks = [
            async_generate(llm, prompt, f"Task-{i}")
            for i, prompt in enumerate(prompts)
        ]

        print("並行處理多個請求...")
        start = time.time()
        results = await asyncio.gather(*tasks)
        total_time = time.time() - start

        print(f"\n總耗時: {total_time:.2f}秒")
        for result in results:
            print(f"\n{result['model']} ({result['time']:.2f}秒):")
            print(f"  {result['response'][:80]}...")

    # 執行非同步操作
    asyncio.run(parallel_generation())

def example6_custom_ollama_chain():
    """範例6: 自定義 Ollama 鏈"""
    print("\n範例6: 自定義 Ollama 鏈")
    print("=" * 50)

    class OllamaToolChain:
        """整合工具的 Ollama 鏈"""

        def __init__(self, model="gemma3:1b"):
            self.llm = Ollama(model=model, temperature=0)
            self.chat_model = ChatOllama(model=model)

        def extract_keywords(self, text: str) -> List[str]:
            """提取關鍵詞"""
            prompt = f"從以下文字提取3-5個關鍵詞，用逗號分隔：\n{text}\n關鍵詞："
            response = self.llm.invoke(prompt)
            keywords = [k.strip() for k in response.split(',')]
            return keywords[:5]  # 最多5個

        def generate_summary(self, text: str) -> str:
            """生成摘要"""
            prompt = f"用一段話總結以下內容：\n{text}\n摘要："
            return self.llm.invoke(prompt)

        def answer_question(self, context: str, question: str) -> str:
            """基於上下文回答問題"""
            messages = [
                SystemMessage(content="根據提供的上下文回答問題"),
                HumanMessage(content=f"上下文：{context}\n\n問題：{question}")
            ]
            response = self.chat_model.invoke(messages)
            return response.content

        def process_document(self, document: str) -> Dict[str, Any]:
            """處理文件：提取關鍵詞、生成摘要、準備問答"""
            print("處理文件中...")

            # 提取關鍵詞
            print("  1. 提取關鍵詞...")
            keywords = self.extract_keywords(document)

            # 生成摘要
            print("  2. 生成摘要...")
            summary = self.generate_summary(document)

            # 生成可能的問題
            print("  3. 生成相關問題...")
            questions_prompt = f"根據以下內容，生成3個相關問題：\n{summary}\n問題："
            questions = self.llm.invoke(questions_prompt)

            return {
                "keywords": keywords,
                "summary": summary,
                "suggested_questions": questions,
                "original_length": len(document)
            }

    # 測試自定義鏈
    tool_chain = OllamaToolChain()

    test_document = """
    人工智慧（AI）正在改變我們的世界。從醫療診斷到自動駕駛，
    AI應用無處不在。機器學習讓電腦能從數據中學習，
    深度學習則模仿人腦神經網路。這些技術帶來便利，
    但也引發隱私和就業的擔憂。
    """

    result = tool_chain.process_document(test_document)

    print("\n文件處理結果：")
    print(f"原始長度: {result['original_length']} 字")
    print(f"關鍵詞: {', '.join(result['keywords'])}")
    print(f"摘要: {result['summary']}")
    print(f"建議問題:\n{result['suggested_questions']}")

    # 測試問答
    question = "AI有什麼應用？"
    answer = tool_chain.answer_question(test_document, question)
    print(f"\n問答測試:")
    print(f"問題: {question}")
    print(f"回答: {answer}")

def example7_model_switching():
    """範例7: 動態模型切換"""
    print("\n範例7: 動態模型切換")
    print("=" * 50)

    class AdaptiveAgent:
        """自適應代理：根據任務選擇模型"""

        def __init__(self):
            self.models = {
                "fast": "gemma3:270m",    # 快速回應
                "balanced": "gemma3:1b",  # 平衡選擇
                "quality": "gemma3:1b"    # 品質優先（實際可用更大模型）
            }
            self.current_mode = "balanced"

        def set_mode(self, mode: str):
            """設定模式"""
            if mode in self.models:
                self.current_mode = mode
                print(f"切換到 {mode} 模式 (使用 {self.models[mode]})")
            else:
                print(f"未知模式: {mode}")

        def process(self, task_type: str, content: str) -> str:
            """根據任務類型處理"""
            # 根據任務自動選擇模式
            if task_type == "quick_answer":
                self.set_mode("fast")
            elif task_type == "analysis":
                self.set_mode("quality")
            else:
                self.set_mode("balanced")

            # 建立 LLM
            llm = Ollama(
                model=self.models[self.current_mode],
                temperature=0.7
            )

            # 處理任務
            prompts = {
                "quick_answer": f"簡短回答：{content}",
                "analysis": f"詳細分析：{content}",
                "translation": f"翻譯成英文：{content}",
                "summary": f"總結要點：{content}"
            }

            prompt = prompts.get(task_type, content)
            start = time.time()
            response = llm.invoke(prompt)
            elapsed = time.time() - start

            print(f"處理完成 (耗時 {elapsed:.2f}秒)")
            return response

    # 測試自適應代理
    agent = AdaptiveAgent()

    # 不同類型的任務
    tasks = [
        ("quick_answer", "1+1等於多少？"),
        ("analysis", "AI對社會的影響"),
        ("translation", "機器學習很有趣"),
        ("summary", "雲端運算是透過網路提供運算資源的技術")
    ]

    for task_type, content in tasks:
        print(f"\n任務: {task_type}")
        print(f"輸入: {content}")
        result = agent.process(task_type, content)
        print(f"輸出: {result[:100]}...")

def main():
    """主程式"""
    print("="*60)
    print("Week 5 - Lesson 4: LangChain + Ollama 深度整合")
    print("="*60)

    # 檢查 Ollama
    try:
        test_llm = Ollama(model="gemma3:1b")
        test_llm.invoke("測試")
        print("✓ Ollama 連接成功\n")
    except Exception as e:
        print(f"✗ 請先啟動 Ollama: ollama serve")
        print(f"  錯誤: {e}")
        return

    # 執行範例
    example1_ollama_vs_chatollama()
    example2_model_parameters()
    example3_streaming_output()
    example4_multi_model_comparison()
    example5_async_operations()
    example6_custom_ollama_chain()
    example7_model_switching()

    print("\n" + "="*60)
    print("課程重點總結：")
    print("1. Ollama 和 ChatOllama 適用於不同場景")
    print("2. 參數調整影響生成品質和速度")
    print("3. 串流輸出提供更好的用戶體驗")
    print("4. 多模型比較幫助選擇最適合的模型")
    print("5. 非同步操作提升系統效能")
    print("6. 自定義鏈整合多種功能")
    print("7. 動態切換模型優化資源使用")
    print("="*60)

if __name__ == "__main__":
    main()