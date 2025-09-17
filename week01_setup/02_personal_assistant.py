#!/usr/bin/env python3
"""
Week 1 - Lab: Personal AI Assistant
建立個人 AI 助理 - 具有記憶功能的對話機器人範例
"""

import ollama
import json
from datetime import datetime

class PersonalAssistant:
    """個人 AI 助理類別"""

    def __init__(self, model="gemma:2b", name="Gemma"):
        self.model = model
        self.name = name
        self.conversation_history = []

        # 系統提示詞
        self.system_prompt = f"""你是一個友善的 AI 助理，名字叫 {self.name}。
你會用繁體中文回答問題，並且記住對話的上下文。
請保持回答簡潔明瞭。"""

        self.conversation_history.append({
            'role': 'system',
            'content': self.system_prompt
        })

    def chat(self, user_input):
        """處理用戶輸入並生成回應"""
        # 添加用戶消息
        self.conversation_history.append({
            'role': 'user',
            'content': user_input
        })

        # 獲取模型回應
        response = ollama.chat(
            model=self.model,
            messages=self.conversation_history
        )

        assistant_message = response['message']['content']

        # 添加助理回應到歷史
        self.conversation_history.append({
            'role': 'assistant',
            'content': assistant_message
        })

        return assistant_message

    def save_session(self, filename="session.json"):
        """儲存對話記錄"""
        session_data = {
            'model': self.model,
            'name': self.name,
            'history': self.conversation_history,
            'timestamp': datetime.now().isoformat()
        }

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(session_data, f, ensure_ascii=False, indent=2)
        print(f"對話已儲存到 {filename}")

def example1_basic_assistant():
    """範例1: 基本助理對話"""
    print("\n範例1: 基本助理對話")
    print("=" * 50)

    assistant = PersonalAssistant(model="gemma:2b", name="小幫手")

    # 第一輪對話
    response1 = assistant.chat("我叫小明，我喜歡程式設計")
    print(f"用戶: 我叫小明，我喜歡程式設計")
    print(f"助理: {response1}")

    # 第二輪對話 - 測試記憶功能
    response2 = assistant.chat("我的名字是什麼？")
    print(f"\n用戶: 我的名字是什麼？")
    print(f"助理: {response2}")

def example2_continuous_conversation():
    """範例2: 連續對話範例"""
    print("\n範例2: 連續對話與記憶")
    print("=" * 50)

    assistant = PersonalAssistant()

    conversations = [
        "今天天氣真好",
        "我想學習 Python",
        "你可以推薦學習資源嗎？"
    ]

    for user_input in conversations:
        response = assistant.chat(user_input)
        print(f"\n用戶: {user_input}")
        print(f"助理: {response}")

def example3_save_and_load():
    """範例3: 儲存對話記錄"""
    print("\n範例3: 儲存對話記錄")
    print("=" * 50)

    assistant = PersonalAssistant(name="記錄助手")

    # 進行幾輪對話
    assistant.chat("請記住：我的生日是 5 月 15 日")
    assistant.chat("我最喜歡的顏色是藍色")

    # 儲存對話
    assistant.save_session("my_conversation.json")

    # 顯示對話歷史
    print("對話歷史：")
    for msg in assistant.conversation_history[1:]:  # 跳過系統提示
        role = "用戶" if msg['role'] == 'user' else "助理"
        print(f"{role}: {msg['content']}")

def main():
    """執行所有範例"""
    print("Personal AI Assistant 範例程式")
    print("=" * 50)

    # 檢查 Ollama
    try:
        models = ollama.list()
        print(f"已連接 Ollama，共 {len(models['models'])} 個模型可用")
    except Exception as e:
        print(f"請先啟動 Ollama: ollama serve")
        return

    # 執行範例
    example1_basic_assistant()
    example2_continuous_conversation()
    example3_save_and_load()

    print("\n所有範例執行完成！")

if __name__ == "__main__":
    main()