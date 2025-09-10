#!/usr/bin/env python3
"""
Week 2 - Lesson 4: OpenAI API 基礎
學習如何使用 OpenAI API（選修）
適合想要比較本地模型與雲端服務的學生
"""

import os
from typing import List, Dict, Optional
import json
from datetime import datetime

# 檢查是否安裝 openai 套件
try:
    from openai import OpenAI
except ImportError:
    print("❌ 需要安裝 OpenAI 套件")
    print("請執行: pip install openai")
    exit(1)

class SimpleAgent:
    """
    簡單的 AI 對話代理
    這個類別幫助你與 AI 進行多輪對話
    """
    
    def __init__(self, api_key: str = None, system: str = "", model: str = "gpt-4o-mini"):
        """
        初始化 Agent
        
        參數說明：
        - api_key: OpenAI API 金鑰
        - system: 系統提示詞（定義 AI 的角色和行為）
        - model: 使用的模型（gpt-4o-mini 比較便宜）
        """
        # 檢查 API 金鑰
        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                print("⚠️  請設定 OpenAI API 金鑰")
                print("方法1: 直接傳入 api_key 參數")
                print("方法2: 設定環境變數 OPENAI_API_KEY")
                raise ValueError("Missing API Key")
        
        # 建立 OpenAI 客戶端
        self.client = OpenAI(api_key=api_key)
        self.model = model
        
        # 初始化對話歷史
        self.system = system
        self.messages = []
        
        # 如果有系統提示詞，加入到對話歷史
        if self.system:
            self.messages.append({
                "role": "system", 
                "content": system
            })
        
        print(f"✅ Agent 初始化成功")
        print(f"📦 使用模型: {self.model}")
    
    def execute(self) -> str:
        """
        執行 API 呼叫，取得 AI 回應
        
        返回：
        - AI 的回應文字
        """
        try:
            # 呼叫 OpenAI API
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=self.messages,
                temperature=0.7  # 控制創造力（0=保守, 1=創意）
            )
            
            # 取得回應內容
            return completion.choices[0].message.content
            
        except Exception as e:
            print(f"❌ API 呼叫失敗: {e}")
            return f"錯誤: {str(e)}"
    
    def __call__(self, message: str) -> str:
        """
        讓 Agent 可以像函數一樣被呼叫
        
        參數：
        - message: 用戶的訊息
        
        返回：
        - AI 的回應
        """
        # 加入用戶訊息
        self.messages.append({
            "role": "user", 
            "content": message
        })
        
        # 執行並取得回應
        result = self.execute()
        
        # 加入 AI 回應到歷史
        self.messages.append({
            "role": "assistant", 
            "content": result
        })
        
        return result
    
    def clear_history(self):
        """清除對話歷史（保留系統提示詞）"""
        self.messages = []
        if self.system:
            self.messages.append({
                "role": "system", 
                "content": self.system
            })
        print("✅ 對話歷史已清除")
    
    def get_history(self) -> List[Dict]:
        """取得完整對話歷史"""
        return self.messages
    
    def save_history(self, filename: str = None):
        """儲存對話歷史到檔案"""
        if filename is None:
            filename = f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.messages, f, ensure_ascii=False, indent=2)
        
        print(f"✅ 對話歷史已儲存到 {filename}")


class LocalAgent(SimpleAgent):
    """
    本地模型版本的 Agent（使用 Ollama）
    讓你可以用相同的方式使用本地模型
    """
    
    def __init__(self, system: str = "", model: str = "gemma2:9b-instruct-q4_0"):
        """
        初始化本地 Agent
        
        參數：
        - system: 系統提示詞
        - model: Ollama 模型名稱
        """
        import ollama
        
        self.client = ollama
        self.model = model
        self.system = system
        self.messages = []
        
        if self.system:
            self.messages.append({
                "role": "system",
                "content": system
            })
        
        print(f"✅ 本地 Agent 初始化成功")
        print(f"📦 使用模型: {self.model}")
    
    def execute(self) -> str:
        """執行本地模型推理"""
        try:
            response = self.client.chat(
                model=self.model,
                messages=self.messages
            )
            return response['message']['content']
        except Exception as e:
            print(f"❌ 本地模型呼叫失敗: {e}")
            return f"錯誤: {str(e)}"


def compare_models_demo():
    """
    示範：比較 OpenAI 和本地模型
    """
    print("=" * 60)
    print("🔬 模型比較實驗")
    print("=" * 60)
    
    # 相同的系統提示詞
    system_prompt = """你是一個友善的助理。
請用繁體中文回答，保持簡潔但資訊豐富。
回答時要有條理，可以使用列點或編號。"""
    
    # 測試問題
    test_questions = [
        "什麼是機器學習？用 100 字以內解釋",
        "列出學習 Python 的 5 個建議",
        "解釋什麼是 API"
    ]
    
    print("\n1️⃣ 測試本地模型 (Ollama)")
    print("-" * 40)
    
    try:
        local_agent = LocalAgent(system=system_prompt)
        
        for i, question in enumerate(test_questions, 1):
            print(f"\n問題 {i}: {question}")
            response = local_agent(question)
            print(f"回答: {response[:200]}...")
            
    except Exception as e:
        print(f"❌ 本地模型測試失敗: {e}")
    
    print("\n2️⃣ 測試 OpenAI API")
    print("-" * 40)
    
    api_key = input("\n請輸入 OpenAI API Key (或按 Enter 跳過): ").strip()
    
    if api_key:
        try:
            openai_agent = SimpleAgent(
                api_key=api_key,
                system=system_prompt,
                model="gpt-4o-mini"  # 使用較便宜的模型
            )
            
            for i, question in enumerate(test_questions, 1):
                print(f"\n問題 {i}: {question}")
                response = openai_agent(question)
                print(f"回答: {response[:200]}...")
                
        except Exception as e:
            print(f"❌ OpenAI API 測試失敗: {e}")
    
    print("\n" + "=" * 60)
    print("📊 比較總結")
    print("=" * 60)
    print("""
本地模型優點：
- 免費使用
- 資料隱私
- 離線運作

OpenAI API 優點：
- 更強大的能力
- 更快的回應
- 支援最新功能
    """)


def basic_usage_demo():
    """
    基礎使用示範
    """
    print("=" * 60)
    print("📚 Agent 基礎使用教學")
    print("=" * 60)
    
    # 步驟 1: 建立 Agent
    print("\n步驟 1: 建立 Agent")
    print("-" * 40)
    print("""
# 使用本地模型
agent = LocalAgent(
    system="你是一個數學老師"
)

# 或使用 OpenAI (需要 API Key)
agent = SimpleAgent(
    api_key="你的API金鑰",
    system="你是一個數學老師"
)
    """)
    
    # 步驟 2: 對話
    print("\n步驟 2: 進行對話")
    print("-" * 40)
    print("""
# 方法 1: 直接呼叫
response = agent("什麼是質數？")
print(response)

# 方法 2: 多輪對話
agent("什麼是質數？")
agent("舉例說明")
agent("如何判斷一個數是否為質數？")
    """)
    
    # 步驟 3: 管理對話
    print("\n步驟 3: 管理對話歷史")
    print("-" * 40)
    print("""
# 查看歷史
history = agent.get_history()

# 清除歷史
agent.clear_history()

# 儲存歷史
agent.save_history("my_chat.json")
    """)


def interactive_chat():
    """
    互動式聊天
    """
    print("=" * 60)
    print("💬 互動式聊天")
    print("=" * 60)
    
    print("\n選擇模型：")
    print("1. 本地模型 (Ollama)")
    print("2. OpenAI API")
    
    choice = input("\n選擇 (1-2): ").strip()
    
    if choice == "1":
        # 使用本地模型
        model_name = input("輸入模型名稱 [預設: gemma2:9b-instruct-q4_0]: ").strip()
        if not model_name:
            model_name = "gemma2:9b-instruct-q4_0"
        
        system = input("輸入系統提示詞 (可選): ").strip()
        
        agent = LocalAgent(system=system, model=model_name)
        
    elif choice == "2":
        # 使用 OpenAI
        api_key = input("輸入 OpenAI API Key: ").strip()
        if not api_key:
            print("❌ 需要 API Key")
            return
        
        system = input("輸入系統提示詞 (可選): ").strip()
        
        agent = SimpleAgent(api_key=api_key, system=system)
    
    else:
        print("❌ 無效選擇")
        return
    
    print("\n開始對話（輸入 /exit 結束, /clear 清除歷史, /save 儲存）")
    print("-" * 60)
    
    while True:
        user_input = input("\n👤 你: ").strip()
        
        if user_input == "/exit":
            print("👋 再見！")
            break
        elif user_input == "/clear":
            agent.clear_history()
        elif user_input == "/save":
            agent.save_history()
        elif user_input:
            response = agent(user_input)
            print(f"\n🤖 AI: {response}")


def main():
    """主程式"""
    print("=" * 60)
    print("🎓 Week 2 - OpenAI Agent 基礎")
    print("=" * 60)
    
    while True:
        print("\n選擇功能：")
        print("1. 基礎教學")
        print("2. 模型比較")
        print("3. 互動聊天")
        print("0. 結束")
        
        choice = input("\n選擇 (0-3): ").strip()
        
        if choice == "0":
            break
        elif choice == "1":
            basic_usage_demo()
        elif choice == "2":
            compare_models_demo()
        elif choice == "3":
            interactive_chat()
        else:
            print("❌ 無效選擇")
        
        input("\n按 Enter 繼續...")
    
    print("\n👋 課程結束！")


if __name__ == "__main__":
    main()