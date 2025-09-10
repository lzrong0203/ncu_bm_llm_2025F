#!/usr/bin/env python3
"""
Week 1 - Lesson 1: Hello LLM
第一個本地 LLM 程式
"""

import ollama
import sys
from typing import Optional

def check_ollama_installation() -> bool:
    """檢查 Ollama 是否已安裝並運行"""
    try:
        models = ollama.list()
        print("✅ Ollama 已成功連接！")
        print(f"📦 已安裝的模型：")
        for model in models['models']:
            print(f"   - {model['name']} ({model['size'] / 1024 / 1024:.1f} MB)")
        return True
    except Exception as e:
        print(f"❌ 無法連接到 Ollama: {e}")
        print("\n請確認：")
        print("1. Ollama 已安裝：https://ollama.com/download")
        print("2. Ollama 服務正在運行：ollama serve")
        return False

def download_model(model_name: str = "gemma2:9b-instruct-q4_0") -> bool:
    """下載指定的模型"""
    try:
        print(f"📥 正在下載模型 {model_name}...")
        ollama.pull(model_name)
        print(f"✅ 模型 {model_name} 下載完成！")
        return True
    except Exception as e:
        print(f"❌ 下載失敗: {e}")
        return False

def simple_chat(model: str = "gemma2:9b-instruct-q4_0", prompt: str = None) -> None:
    """簡單的對話功能"""
    if prompt is None:
        prompt = "介紹一下你自己，用繁體中文回答"
    
    print(f"\n🤖 使用模型: {model}")
    print(f"📝 提示詞: {prompt}")
    print("-" * 50)
    
    try:
        # 方法1: 簡單對話（一次性回應）
        response = ollama.chat(
            model=model,
            messages=[{
                'role': 'user',
                'content': prompt
            }]
        )
        print("💬 回應：")
        print(response['message']['content'])
        
        # 顯示統計資訊
        if 'total_duration' in response:
            duration_sec = response['total_duration'] / 1_000_000_000
            print(f"\n⏱️  生成時間: {duration_sec:.2f} 秒")
        
    except Exception as e:
        print(f"❌ 對話失敗: {e}")

def streaming_chat(model: str = "gemma2:9b-instruct-q4_0", prompt: str = None) -> None:
    """串流對話（打字機效果）"""
    if prompt is None:
        prompt = "什麼是機器學習？請用簡單的方式解釋"
    
    print(f"\n🤖 使用模型: {model}")
    print(f"📝 提示詞: {prompt}")
    print("-" * 50)
    print("💬 回應：")
    
    try:
        # 方法2: 串流回應（打字機效果）
        stream = ollama.chat(
            model=model,
            messages=[{
                'role': 'user',
                'content': prompt
            }],
            stream=True
        )
        
        for chunk in stream:
            print(chunk['message']['content'], end='', flush=True)
        print("\n")
        
    except Exception as e:
        print(f"❌ 串流對話失敗: {e}")

def explore_model_parameters(model: str = "gemma2:9b-instruct-q4_0") -> None:
    """探索模型參數的影響"""
    prompt = "寫一句關於 AI 的句子"
    temperatures = [0.1, 0.5, 0.9]
    
    print(f"\n🔬 實驗：Temperature 對輸出的影響")
    print(f"📝 提示詞: {prompt}")
    print("=" * 50)
    
    for temp in temperatures:
        print(f"\n🌡️  Temperature = {temp}")
        response = ollama.generate(
            model=model,
            prompt=prompt,
            options={
                'temperature': temp,
                'seed': 42  # 固定種子以便比較
            }
        )
        print(f"   {response['response']}")

def main():
    """主程式"""
    print("=" * 50)
    print("🚀 Week 1 - Lesson 1: Hello LLM")
    print("=" * 50)
    
    # Step 1: 檢查 Ollama
    if not check_ollama_installation():
        sys.exit(1)
    
    # Step 2: 確認模型（支援多種選擇）
    print("\n🤖 可用模型選項：")
    print("1. gemma2:2b - 超輕量 (1.4GB, 8GB RAM)")
    print("2. gemma2:9b-instruct-q4_0 - 推薦 (5.5GB, 16GB RAM)")
    print("3. llama3.2:3b - Meta 最新 (2GB, 8GB RAM)")
    print("4. qwen2.5:7b - 中文最強 (4.7GB, 16GB RAM)")
    
    choice = input("\n選擇模型 (1-4) [預設: 2]: ").strip()
    
    model_options = {
        "1": "gemma2:2b",
        "2": "gemma2:9b-instruct-q4_0",
        "3": "llama3.2:3b",
        "4": "qwen2.5:7b"
    }
    
    model_name = model_options.get(choice, "gemma2:9b-instruct-q4_0")
    
    # 檢查模型是否已存在
    models = ollama.list()
    model_exists = any(model_name in m['name'] for m in models['models'])
    
    if not model_exists:
        print(f"\n⚠️  模型 {model_name} 未安裝")
        response = input("是否要下載？(y/n): ")
        if response.lower() == 'y':
            if not download_model(model_name):
                sys.exit(1)
        else:
            print(f"請先下載模型：ollama pull {model_name}")
            sys.exit(1)
    
    # Step 3: 執行不同的對話模式
    print("\n" + "=" * 50)
    print("📚 示範 1: 簡單對話")
    print("=" * 50)
    simple_chat(model_name)
    
    input("\n按 Enter 繼續...")
    
    print("\n" + "=" * 50)
    print("📚 示範 2: 串流對話")
    print("=" * 50)
    streaming_chat(model_name)
    
    input("\n按 Enter 繼續...")
    
    print("\n" + "=" * 50)
    print("📚 示範 3: Temperature 實驗")
    print("=" * 50)
    explore_model_parameters(model_name)
    
    print("\n✅ 課程完成！")
    print("📝 練習建議：")
    print("1. 嘗試不同的提示詞")
    print("2. 測試不同的 temperature 值")
    print("3. 比較不同模型的回應速度和品質")

if __name__ == "__main__":
    main()