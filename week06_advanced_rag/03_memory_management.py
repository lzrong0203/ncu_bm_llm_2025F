#!/usr/bin/env python3
"""
Week 5 - Lesson 3: 記憶體與對話管理
學習 LangChain 的記憶體系統和對話狀態管理
"""

from langchain_community.llms import Ollama
from langchain.memory import (
    ConversationBufferMemory,
    ConversationBufferWindowMemory,
    ConversationSummaryMemory,
    ConversationSummaryBufferMemory,
    ConversationEntityMemory
)
from langchain.chains import ConversationChain, LLMChain
from langchain_core.prompts import PromptTemplate
from langchain.schema import BaseMessage, HumanMessage, AIMessage
import json
from datetime import datetime
from typing import List, Dict, Any

def example1_buffer_memory():
    """範例1: 基本緩衝記憶體"""
    print("\n範例1: 基本緩衝記憶體 (ConversationBufferMemory)")
    print("=" * 50)

    # 初始化 LLM 和記憶體
    llm = Ollama(model="gemma3:1b")
    memory = ConversationBufferMemory()

    # 建立對話鏈
    conversation = ConversationChain(
        llm=llm,
        memory=memory,
        verbose=True  # 顯示詳細過程
    )

    # 進行對話
    print("開始對話...")

    response1 = conversation.predict(input="我叫小明，我喜歡打籃球")
    print(f"AI: {response1}\n")

    response2 = conversation.predict(input="我還喜歡看科幻電影")
    print(f"AI: {response2}\n")

    response3 = conversation.predict(input="我的名字是什麼？我有什麼興趣？")
    print(f"AI: {response3}\n")

    # 查看記憶體內容
    print("\n記憶體內容：")
    print(memory.buffer)

    # 獲取對話歷史
    messages = memory.chat_memory.messages
    print(f"\n對話歷史（{len(messages)} 條訊息）：")
    for msg in messages:
        role = "User" if isinstance(msg, HumanMessage) else "AI"
        print(f"{role}: {msg.content[:50]}...")

def example2_window_memory():
    """範例2: 視窗記憶體（限制對話長度）"""
    print("\n範例2: 視窗記憶體 (ConversationBufferWindowMemory)")
    print("=" * 50)

    # 只保留最近 3 輪對話
    memory = ConversationBufferWindowMemory(k=3)
    llm = Ollama(model="gemma3:1b")

    conversation = ConversationChain(
        llm=llm,
        memory=memory,
        verbose=False
    )

    # 進行多輪對話
    conversations = [
        "我是一個程式設計師",
        "我使用Python和JavaScript",
        "我在一家科技公司工作",
        "我負責後端開發",
        "我的專長是什麼？"  # 這時應該還記得
    ]

    for i, user_input in enumerate(conversations, 1):
        print(f"\n第{i}輪對話：")
        print(f"User: {user_input}")
        response = conversation.predict(input=user_input)
        print(f"AI: {response[:100]}...")

    print("\n視窗記憶體內容（只保留最近3輪）：")
    print(memory.buffer)

def example3_summary_memory():
    """範例3: 摘要記憶體（節省記憶體空間）"""
    print("\n範例3: 摘要記憶體 (ConversationSummaryMemory)")
    print("=" * 50)

    llm = Ollama(model="gemma3:1b")

    # 使用摘要記憶體
    memory = ConversationSummaryMemory(llm=llm)

    conversation = ConversationChain(
        llm=llm,
        memory=memory,
        verbose=False
    )

    # 長對話測試
    long_conversation = [
        "我想開始學習機器學習",
        "我已經有Python基礎，也懂一些數學",
        "請推薦一些入門資源",
        "深度學習和機器學習有什麼區別？",
        "哪些框架比較適合初學者？"
    ]

    for user_input in long_conversation:
        print(f"User: {user_input}")
        response = conversation.predict(input=user_input)
        print(f"AI: {response[:80]}...\n")

    print("摘要記憶體內容：")
    print(memory.buffer)

def example4_summary_buffer_memory():
    """範例4: 摘要緩衝記憶體（混合模式）"""
    print("\n範例4: 摘要緩衝記憶體 (ConversationSummaryBufferMemory)")
    print("=" * 50)

    llm = Ollama(model="gemma3:1b")

    # 保留最近的對話，較舊的對話轉為摘要
    memory = ConversationSummaryBufferMemory(
        llm=llm,
        max_token_limit=200  # 超過200個token後開始摘要
    )

    conversation = ConversationChain(
        llm=llm,
        memory=memory
    )

    # 模擬客服對話
    customer_service = [
        "我想詢問訂單#12345的狀態",
        "我是上週三下單的",
        "商品是一台筆電和滑鼠",
        "可以修改送貨地址嗎？",
        "新地址是台北市信義區XXX路"
    ]

    for query in customer_service:
        print(f"客戶: {query}")
        response = conversation.predict(input=query)
        print(f"客服: {response[:100]}...\n")

    print("混合記憶體狀態：")
    print(f"摘要部分: {memory.moving_summary_buffer}")
    print(f"最近對話: {[msg.content[:30] for msg in memory.chat_memory.messages[-4:]]}")

def example5_custom_memory_handler():
    """範例5: 自定義記憶體處理"""
    print("\n範例5: 自定義記憶體處理")
    print("=" * 50)

    class ConversationManager:
        """對話管理器"""

        def __init__(self, model="gemma3:1b", save_history=True):
            self.llm = Ollama(model=model)
            self.memory = ConversationBufferMemory()
            self.save_history = save_history
            self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.conversation = ConversationChain(
                llm=self.llm,
                memory=self.memory
            )

        def chat(self, user_input: str) -> str:
            """進行對話"""
            response = self.conversation.predict(input=user_input)

            # 儲存對話
            if self.save_history:
                self._save_turn(user_input, response)

            return response

        def _save_turn(self, user_input: str, response: str):
            """儲存單次對話"""
            turn = {
                "timestamp": datetime.now().isoformat(),
                "user": user_input,
                "assistant": response
            }

            filename = f"conversation_{self.session_id}.json"
            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    history = json.load(f)
            except FileNotFoundError:
                history = []

            history.append(turn)

            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(history, f, ensure_ascii=False, indent=2)

        def get_summary(self) -> str:
            """獲取對話摘要"""
            if not self.memory.chat_memory.messages:
                return "尚無對話記錄"

            prompt = PromptTemplate(
                input_variables=["history"],
                template="請總結以下對話的重點：\n{history}\n\n摘要："
            )

            chain = LLMChain(llm=self.llm, prompt=prompt)
            summary = chain.run(history=self.memory.buffer)
            return summary

        def clear_memory(self):
            """清除記憶體"""
            self.memory.clear()
            print("記憶體已清除")

        def export_memory(self) -> Dict[str, Any]:
            """匯出記憶體"""
            return {
                "session_id": self.session_id,
                "messages": [
                    {
                        "type": "human" if isinstance(msg, HumanMessage) else "ai",
                        "content": msg.content
                    }
                    for msg in self.memory.chat_memory.messages
                ],
                "timestamp": datetime.now().isoformat()
            }

    # 使用自定義管理器
    print("建立對話管理器...")
    manager = ConversationManager()

    # 模擬對話
    test_conversation = [
        "你好，我想了解你們的產品",
        "有哪些付款方式？",
        "可以貨到付款嗎？"
    ]

    for user_input in test_conversation:
        print(f"\nUser: {user_input}")
        response = manager.chat(user_input)
        print(f"AI: {response[:100]}...")

    # 獲取摘要
    print("\n對話摘要：")
    summary = manager.get_summary()
    print(summary)

    # 匯出記憶體
    exported = manager.export_memory()
    print(f"\n匯出的對話數量: {len(exported['messages'])} 條")

def example6_multi_user_memory():
    """範例6: 多用戶記憶體管理"""
    print("\n範例6: 多用戶記憶體管理")
    print("=" * 50)

    class MultiUserChatbot:
        """多用戶聊天機器人"""

        def __init__(self, model="gemma3:1b"):
            self.llm = Ollama(model=model)
            self.user_memories = {}  # 儲存每個用戶的記憶體

        def get_or_create_memory(self, user_id: str) -> ConversationBufferMemory:
            """獲取或建立用戶記憶體"""
            if user_id not in self.user_memories:
                self.user_memories[user_id] = ConversationBufferMemory()
                print(f"為用戶 {user_id} 建立新的記憶體")
            return self.user_memories[user_id]

        def chat(self, user_id: str, message: str) -> str:
            """與特定用戶對話"""
            memory = self.get_or_create_memory(user_id)

            conversation = ConversationChain(
                llm=self.llm,
                memory=memory,
                verbose=False
            )

            return conversation.predict(input=message)

        def get_user_history(self, user_id: str) -> List[str]:
            """獲取用戶對話歷史"""
            if user_id not in self.user_memories:
                return []

            memory = self.user_memories[user_id]
            messages = memory.chat_memory.messages
            return [msg.content for msg in messages]

        def clear_user_memory(self, user_id: str):
            """清除特定用戶的記憶體"""
            if user_id in self.user_memories:
                self.user_memories[user_id].clear()
                print(f"已清除用戶 {user_id} 的記憶體")

    # 測試多用戶系統
    chatbot = MultiUserChatbot()

    # 用戶A的對話
    print("用戶A的對話：")
    print("A: 我是Alice，我是工程師")
    response = chatbot.chat("user_a", "我是Alice，我是工程師")
    print(f"Bot: {response}\n")

    # 用戶B的對話
    print("用戶B的對話：")
    print("B: 我是Bob，我是設計師")
    response = chatbot.chat("user_b", "我是Bob，我是設計師")
    print(f"Bot: {response}\n")

    # 用戶A繼續對話
    print("用戶A繼續：")
    print("A: 我的職業是什麼？")
    response = chatbot.chat("user_a", "我的職業是什麼？")
    print(f"Bot: {response}\n")

    # 用戶B繼續對話
    print("用戶B繼續：")
    print("B: 我叫什麼名字？")
    response = chatbot.chat("user_b", "我叫什麼名字？")
    print(f"Bot: {response}\n")

    print("系統中的用戶數量:", len(chatbot.user_memories))

def example7_persistent_memory():
    """範例7: 持久化記憶體"""
    print("\n範例7: 持久化記憶體")
    print("=" * 50)

    import pickle
    import os

    class PersistentConversation:
        """可持久化的對話系統"""

        def __init__(self, session_file="conversation_session.pkl"):
            self.session_file = session_file
            self.llm = Ollama(model="gemma3:1b")
            self.memory = self.load_memory()
            self.conversation = ConversationChain(
                llm=self.llm,
                memory=self.memory
            )

        def load_memory(self) -> ConversationBufferMemory:
            """載入記憶體"""
            if os.path.exists(self.session_file):
                try:
                    with open(self.session_file, 'rb') as f:
                        memory_data = pickle.load(f)
                    print(f"已載入先前的對話（{len(memory_data)} 條訊息）")

                    memory = ConversationBufferMemory()
                    # 重建記憶體
                    for msg in memory_data:
                        if msg['type'] == 'human':
                            memory.chat_memory.add_user_message(msg['content'])
                        else:
                            memory.chat_memory.add_ai_message(msg['content'])
                    return memory
                except Exception as e:
                    print(f"載入失敗: {e}")

            print("開始新的對話")
            return ConversationBufferMemory()

        def save_memory(self):
            """儲存記憶體"""
            memory_data = []
            for msg in self.memory.chat_memory.messages:
                memory_data.append({
                    'type': 'human' if isinstance(msg, HumanMessage) else 'ai',
                    'content': msg.content
                })

            with open(self.session_file, 'wb') as f:
                pickle.dump(memory_data, f)
            print(f"對話已儲存（{len(memory_data)} 條訊息）")

        def chat(self, user_input: str) -> str:
            """對話並自動儲存"""
            response = self.conversation.predict(input=user_input)
            self.save_memory()  # 每次對話後自動儲存
            return response

        def show_stats(self):
            """顯示統計資訊"""
            messages = self.memory.chat_memory.messages
            human_msgs = sum(1 for msg in messages if isinstance(msg, HumanMessage))
            ai_msgs = len(messages) - human_msgs

            print(f"\n對話統計：")
            print(f"  總訊息數: {len(messages)}")
            print(f"  用戶訊息: {human_msgs}")
            print(f"  AI回應: {ai_msgs}")

    # 測試持久化對話
    print("測試持久化對話系統...")
    persistent_chat = PersistentConversation()

    # 進行對話
    test_inputs = [
        "記住這個數字：42",
        "我最喜歡的顏色是藍色"
    ]

    for user_input in test_inputs:
        print(f"\nUser: {user_input}")
        response = persistent_chat.chat(user_input)
        print(f"AI: {response[:100]}...")

    # 顯示統計
    persistent_chat.show_stats()

    print("\n提示：對話已儲存，下次執行時會自動載入")

def main():
    """主程式"""
    print("="*60)
    print("Week 5 - Lesson 3: 記憶體與對話管理")
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
    example1_buffer_memory()
    example2_window_memory()
    example3_summary_memory()
    example4_summary_buffer_memory()
    example5_custom_memory_handler()
    example6_multi_user_memory()
    example7_persistent_memory()

    print("\n" + "="*60)
    print("課程重點總結：")
    print("1. ConversationBufferMemory 保存完整對話歷史")
    print("2. ConversationBufferWindowMemory 限制記憶體大小")
    print("3. ConversationSummaryMemory 使用摘要節省空間")
    print("4. 混合記憶體策略平衡效能和完整性")
    print("5. 自定義記憶體處理滿足特殊需求")
    print("6. 多用戶系統需要獨立的記憶體管理")
    print("7. 持久化確保對話的連續性")
    print("="*60)

if __name__ == "__main__":
    main()