#!/usr/bin/env python3
"""
Week 2 - Lesson 4: OpenAI API 基礎
簡化範例 - 展示本地模型與 OpenAI API 的基本用法
"""

import os
import json
from datetime import datetime

# 檢查 openai 套件
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("提示：OpenAI 套件未安裝，將只使用本地模型範例")

class SimpleAgent:
    """簡單的 AI 對話代理"""

    def __init__(self, api_key=None, system="", model="gpt-4o-mini"):
        if not OPENAI_AVAILABLE:
            raise ImportError("需要安裝 openai: pip install openai")

        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.model = model
        self.messages = []

        if system:
            self.messages.append({"role": "system", "content": system})

    def chat(self, message):
        """進行對話"""
        self.messages.append({"role": "user", "content": message})

        completion = self.client.chat.completions.create(
            model=self.model,
            messages=self.messages
        )

        response = completion.choices[0].message.content
        self.messages.append({"role": "assistant", "content": response})

        return response

class LocalAgent:
    """本地模型版本的 Agent（使用 Ollama）"""

    def __init__(self, system="", model="gemma:2b"):
        import ollama
        self.client = ollama
        self.model = model
        self.messages = []

        if system:
            self.messages.append({"role": "system", "content": system})

    def chat(self, message):
        """進行對話"""
        self.messages.append({"role": "user", "content": message})

        response = self.client.chat(
            model=self.model,
            messages=self.messages
        )

        result = response['message']['content']
        self.messages.append({"role": "assistant", "content": result})

        return result

def example1_local_model():
    """範例1: 使用本地模型"""
    print("\n範例1: 本地模型 (Ollama)")
    print("=" * 50)

    # 建立本地 Agent
    agent = LocalAgent(
        system="你是一個友善的助理，用繁體中文回答",
        model="gemma:2b"
    )

    # 簡單對話
    questions = [
        "什麼是機器學習？",
        "舉一個例子",
        "如何開始學習？"
    ]

    for q in questions:
        print(f"\n問: {q}")
        response = agent.chat(q)
        print(f"答: {response[:150]}...")

def example2_openai_demo():
    """範例2: OpenAI API 示範（需要 API Key）"""
    print("\n範例2: OpenAI API 示範")
    print("=" * 50)

    if not OPENAI_AVAILABLE:
        print("OpenAI 套件未安裝，跳過此範例")
        return

    # 這裡展示程式碼結構，實際使用需要 API Key
    print("""
使用方式：
```python
# 建立 OpenAI Agent
agent = SimpleAgent(
    api_key="你的API金鑰",
    system="你是一個數學老師",
    model="gpt-4o-mini"
)

# 進行對話
response = agent.chat("什麼是質數？")
print(response)
```
    """)

def example3_compare_models():
    """範例3: 比較不同模型回應"""
    print("\n範例3: 模型回應比較")
    print("=" * 50)

    system_prompt = "你是一個簡潔的助理"
    test_prompt = "用一句話解釋什麼是 AI"

    # 測試本地模型
    print("本地模型回應：")
    local_agent = LocalAgent(system=system_prompt)
    local_response = local_agent.chat(test_prompt)
    print(f"  {local_response}")

    print("\nOpenAI 模型回應：")
    print("  （需要 API Key 才能執行）")

    # 比較表
    print("\n模型比較：")
    print("┌─────────────┬───────────────┬──────────────┐")
    print("│ 特性        │ 本地模型      │ OpenAI API   │")
    print("├─────────────┼───────────────┼──────────────┤")
    print("│ 費用        │ 免費          │ 按使用計費   │")
    print("│ 隱私        │ 完全私密      │ 資料傳到雲端 │")
    print("│ 速度        │ 依硬體而定    │ 通常較快     │")
    print("│ 能力        │ 基本到中等    │ 較強大       │")
    print("└─────────────┴───────────────┴──────────────┘")

def example4_conversation_history():
    """範例4: 對話歷史管理"""
    print("\n範例4: 對話歷史管理")
    print("=" * 50)

    agent = LocalAgent(system="你是一個記憶力很好的助理")

    # 建立有上下文的對話
    agent.chat("我叫小明，我喜歡打籃球")
    agent.chat("我住在台北")
    response = agent.chat("我的名字和愛好是什麼？")

    print("對話歷史：")
    for msg in agent.messages:
        if msg['role'] != 'system':
            role = "用戶" if msg['role'] == 'user' else "助理"
            print(f"{role}: {msg['content'][:50]}")

    print(f"\n最後回應: {response}")

def example5_save_conversation():
    """範例5: 儲存對話記錄"""
    print("\n範例5: 儲存對話記錄")
    print("=" * 50)

    agent = LocalAgent()

    # 進行一些對話
    agent.chat("今天天氣如何？")
    agent.chat("適合做什麼活動？")

    # 儲存對話
    filename = f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(agent.messages, f, ensure_ascii=False, indent=2)

    print(f"對話已儲存到: {filename}")
    print("對話內容預覽：")
    for msg in agent.messages[:3]:
        print(f"  {msg['role']}: {msg['content'][:30]}...")

def main():
    """執行所有範例"""
    print("OpenAI Agent 基礎範例")
    print("=" * 50)

    # 檢查 Ollama
    try:
        import ollama
        ollama.list()
        print("已連接 Ollama")
    except Exception:
        print("請先啟動 Ollama: ollama serve")
        return

    # 執行範例
    example1_local_model()
    example2_openai_demo()
    example3_compare_models()
    example4_conversation_history()
    example5_save_conversation()

    print("\n所有範例執行完成！")

if __name__ == "__main__":
    main()