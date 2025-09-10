#!/usr/bin/env python3
"""
Week 1 - Lab: Personal AI Assistant
建立個人 AI 助理 - 具有記憶和特殊指令的對話機器人
"""

import ollama
import json
import os
from datetime import datetime
from typing import List, Dict, Optional
import pickle

class PersonalAssistant:
    """個人 AI 助理類別"""
    
    def __init__(self, model: str = "gemma:2b", name: str = "Gemma"):
        """
        初始化助理
        
        Args:
            model: 使用的模型名稱
            name: 助理的名字
        """
        self.model = model
        self.name = name
        self.conversation_history = []
        self.temperature = 0.7
        self.max_history = 10  # 保留最近 N 輪對話
        self.session_file = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # 系統提示詞
        self.system_prompt = f"""你是一個友善的 AI 助理，名字叫 {self.name}。
你會用繁體中文回答問題，並且記住對話的上下文。
請保持回答簡潔明瞭，但也要友善和有幫助。"""
        
        # 初始化對話歷史
        self.conversation_history.append({
            'role': 'system',
            'content': self.system_prompt
        })
    
    def chat(self, user_input: str) -> str:
        """
        處理用戶輸入並生成回應
        
        Args:
            user_input: 用戶的輸入
            
        Returns:
            助理的回應
        """
        # 添加用戶消息到歷史
        self.conversation_history.append({
            'role': 'user',
            'content': user_input
        })
        
        # 限制歷史長度
        if len(self.conversation_history) > self.max_history * 2 + 1:
            # 保留系統提示詞和最近的對話
            self.conversation_history = [self.conversation_history[0]] + \
                                       self.conversation_history[-(self.max_history * 2):]
        
        try:
            # 獲取模型回應
            response = ollama.chat(
                model=self.model,
                messages=self.conversation_history,
                options={
                    'temperature': self.temperature,
                }
            )
            
            assistant_message = response['message']['content']
            
            # 添加助理回應到歷史
            self.conversation_history.append({
                'role': 'assistant',
                'content': assistant_message
            })
            
            return assistant_message
            
        except Exception as e:
            return f"❌ 發生錯誤: {e}"
    
    def streaming_chat(self, user_input: str) -> None:
        """
        串流方式處理對話（打字機效果）
        
        Args:
            user_input: 用戶的輸入
        """
        # 添加用戶消息到歷史
        self.conversation_history.append({
            'role': 'user',
            'content': user_input
        })
        
        # 限制歷史長度
        if len(self.conversation_history) > self.max_history * 2 + 1:
            self.conversation_history = [self.conversation_history[0]] + \
                                       self.conversation_history[-(self.max_history * 2):]
        
        try:
            # 串流獲取回應
            stream = ollama.chat(
                model=self.model,
                messages=self.conversation_history,
                options={
                    'temperature': self.temperature,
                },
                stream=True
            )
            
            full_response = ""
            for chunk in stream:
                content = chunk['message']['content']
                print(content, end='', flush=True)
                full_response += content
            
            print()  # 換行
            
            # 添加完整回應到歷史
            self.conversation_history.append({
                'role': 'assistant',
                'content': full_response
            })
            
        except Exception as e:
            print(f"❌ 發生錯誤: {e}")
    
    def set_temperature(self, temp: float) -> None:
        """設定 temperature 參數"""
        if 0 <= temp <= 1:
            self.temperature = temp
            print(f"✅ Temperature 已設定為 {temp}")
        else:
            print("❌ Temperature 必須在 0 到 1 之間")
    
    def clear_history(self) -> None:
        """清除對話歷史"""
        self.conversation_history = [{
            'role': 'system',
            'content': self.system_prompt
        }]
        print("✅ 對話歷史已清除")
    
    def save_session(self, filename: Optional[str] = None) -> None:
        """儲存對話記錄"""
        if filename is None:
            filename = self.session_file
        
        session_data = {
            'model': self.model,
            'name': self.name,
            'temperature': self.temperature,
            'history': self.conversation_history,
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(session_data, f, ensure_ascii=False, indent=2)
            print(f"✅ 對話已儲存到 {filename}")
        except Exception as e:
            print(f"❌ 儲存失敗: {e}")
    
    def load_session(self, filename: str) -> None:
        """載入對話記錄"""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                session_data = json.load(f)
            
            self.model = session_data['model']
            self.name = session_data['name']
            self.temperature = session_data['temperature']
            self.conversation_history = session_data['history']
            
            print(f"✅ 已載入對話記錄 from {filename}")
            print(f"   模型: {self.model}")
            print(f"   時間: {session_data['timestamp']}")
        except Exception as e:
            print(f"❌ 載入失敗: {e}")
    
    def show_history(self) -> None:
        """顯示對話歷史"""
        print("\n📜 對話歷史：")
        print("-" * 50)
        for i, msg in enumerate(self.conversation_history[1:], 1):  # 跳過系統提示
            role = "👤 用戶" if msg['role'] == 'user' else "🤖 助理"
            print(f"{i}. {role}: {msg['content'][:100]}...")
        print("-" * 50)
    
    def get_stats(self) -> Dict:
        """獲取對話統計"""
        user_msgs = sum(1 for m in self.conversation_history if m['role'] == 'user')
        assistant_msgs = sum(1 for m in self.conversation_history if m['role'] == 'assistant')
        
        return {
            'total_messages': len(self.conversation_history) - 1,  # 排除系統提示
            'user_messages': user_msgs,
            'assistant_messages': assistant_msgs,
            'temperature': self.temperature,
            'model': self.model
        }

def print_help():
    """顯示幫助資訊"""
    print("""
╔══════════════════════════════════════════════════════╗
║                    指令列表                           ║
╠══════════════════════════════════════════════════════╣
║ /help        - 顯示此幫助訊息                        ║
║ /clear       - 清除對話歷史                          ║
║ /history     - 顯示對話歷史                          ║
║ /save [檔名] - 儲存對話記錄                          ║
║ /load 檔名   - 載入對話記錄                          ║
║ /temp 數值   - 設定 temperature (0-1)                ║
║ /stats       - 顯示統計資訊                          ║
║ /model       - 顯示當前模型                          ║
║ /exit        - 結束程式                              ║
╚══════════════════════════════════════════════════════╝
    """)

def main():
    """主程式"""
    print("=" * 60)
    print("🤖 個人 AI 助理 v1.0")
    print("=" * 60)
    
    # 檢查 Ollama 連接
    try:
        models = ollama.list()
        print("✅ 已連接到 Ollama")
        print("📦 可用模型：")
        for model in models['models']:
            print(f"   - {model['name']}")
    except Exception as e:
        print(f"❌ 無法連接到 Ollama: {e}")
        return
    
    # 選擇模型
    model_name = input("\n請輸入要使用的模型名稱 [預設: gemma:2b]: ").strip()
    if not model_name:
        model_name = "gemma:2b"
    
    # 建立助理
    assistant_name = input("請為你的助理取個名字 [預設: Gemma]: ").strip()
    if not assistant_name:
        assistant_name = "Gemma"
    
    assistant = PersonalAssistant(model=model_name, name=assistant_name)
    
    print(f"\n✨ {assistant_name} 已準備就緒！")
    print("💡 輸入 /help 查看可用指令")
    print("-" * 60)
    
    # 主要對話迴圈
    while True:
        try:
            # 獲取用戶輸入
            user_input = input("\n👤 你: ").strip()
            
            if not user_input:
                continue
            
            # 處理特殊指令
            if user_input.startswith('/'):
                command_parts = user_input.split(maxsplit=1)
                command = command_parts[0].lower()
                args = command_parts[1] if len(command_parts) > 1 else None
                
                if command == '/exit':
                    # 詢問是否儲存
                    save_choice = input("是否要儲存對話記錄？(y/n): ").lower()
                    if save_choice == 'y':
                        assistant.save_session()
                    print(f"👋 再見！感謝使用 {assistant_name}")
                    break
                
                elif command == '/help':
                    print_help()
                
                elif command == '/clear':
                    assistant.clear_history()
                
                elif command == '/history':
                    assistant.show_history()
                
                elif command == '/save':
                    assistant.save_session(args)
                
                elif command == '/load':
                    if args:
                        assistant.load_session(args)
                    else:
                        print("❌ 請提供檔案名稱")
                
                elif command == '/temp':
                    if args:
                        try:
                            temp = float(args)
                            assistant.set_temperature(temp)
                        except ValueError:
                            print("❌ 請輸入有效的數字")
                    else:
                        print(f"📊 當前 Temperature: {assistant.temperature}")
                
                elif command == '/stats':
                    stats = assistant.get_stats()
                    print("\n📊 統計資訊：")
                    for key, value in stats.items():
                        print(f"   {key}: {value}")
                
                elif command == '/model':
                    print(f"🤖 當前模型: {assistant.model}")
                
                else:
                    print(f"❌ 未知指令: {command}")
                    print("💡 輸入 /help 查看可用指令")
            
            else:
                # 一般對話
                print(f"\n🤖 {assistant_name}: ", end='')
                assistant.streaming_chat(user_input)
        
        except KeyboardInterrupt:
            print("\n\n⚠️  收到中斷信號")
            save_choice = input("是否要儲存對話記錄？(y/n): ").lower()
            if save_choice == 'y':
                assistant.save_session()
            print(f"👋 再見！")
            break
        
        except Exception as e:
            print(f"\n❌ 發生錯誤: {e}")
            continue

if __name__ == "__main__":
    main()