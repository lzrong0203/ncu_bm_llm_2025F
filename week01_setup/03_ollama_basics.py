#!/usr/bin/env python3
"""
Week 1 - Lesson 3: Ollama API 基礎
深入了解 Ollama API 的各種功能
"""

import ollama
import json
import time
from typing import Dict, List, Any

class OllamaExplorer:
    """Ollama API 探索工具"""
    
    def __init__(self):
        self.client = ollama
    
    def list_models(self) -> List[Dict]:
        """列出所有已安裝的模型"""
        print("\n📦 已安裝的模型：")
        print("=" * 60)
        
        models = self.client.list()
        model_list = []
        
        for model in models['models']:
            size_mb = model['size'] / (1024 * 1024)
            modified = model['modified_at']
            
            print(f"📌 {model['name']}")
            print(f"   大小: {size_mb:.1f} MB")
            print(f"   修改時間: {modified}")
            print(f"   摘要: {model['digest'][:20]}...")
            print("-" * 40)
            
            model_list.append({
                'name': model['name'],
                'size_mb': size_mb,
                'modified': modified
            })
        
        return model_list
    
    def show_model_info(self, model_name: str) -> Dict:
        """顯示模型詳細資訊"""
        print(f"\n🔍 模型資訊: {model_name}")
        print("=" * 60)
        
        try:
            info = self.client.show(model_name)
            
            # 基本資訊
            print("📊 基本資訊:")
            print(f"   模型家族: {info.get('details', {}).get('family', 'N/A')}")
            print(f"   參數量: {info.get('details', {}).get('parameter_size', 'N/A')}")
            print(f"   量化等級: {info.get('details', {}).get('quantization_level', 'N/A')}")
            
            # 模型卡片 (如果有的話)
            if 'modelfile' in info:
                print("\n📝 Modelfile 片段:")
                lines = info['modelfile'].split('\n')[:5]
                for line in lines:
                    print(f"   {line}")
            
            # 授權資訊
            if 'license' in info:
                print(f"\n📜 授權: {info['license'][:100]}...")
            
            return info
            
        except Exception as e:
            print(f"❌ 無法獲取模型資訊: {e}")
            return {}
    
    def test_generate_api(self, model: str = "gemma:2b") -> None:
        """測試 generate API (原始文字生成)"""
        print(f"\n🔬 測試 Generate API - {model}")
        print("=" * 60)
        
        prompts = [
            "完成這個句子：人工智慧的未來",
            "寫一個關於機器人的短故事，50字以內：",
            "解釋什麼是深度學習："
        ]
        
        for prompt in prompts:
            print(f"\n📝 Prompt: {prompt}")
            
            start_time = time.time()
            response = self.client.generate(
                model=model,
                prompt=prompt,
                options={
                    'temperature': 0.7,
                    'max_tokens': 100
                }
            )
            elapsed = time.time() - start_time
            
            print(f"💬 Response: {response['response']}")
            print(f"⏱️  生成時間: {elapsed:.2f} 秒")
            
            if 'context' in response:
                print(f"📊 Context 長度: {len(response['context'])}")
    
    def test_chat_api(self, model: str = "gemma:2b") -> None:
        """測試 chat API (對話模式)"""
        print(f"\n🔬 測試 Chat API - {model}")
        print("=" * 60)
        
        # 多輪對話測試
        messages = [
            {'role': 'system', 'content': '你是一個友善的助理，用繁體中文回答'},
            {'role': 'user', 'content': '請記住我最喜歡的顏色是藍色'},
            {'role': 'assistant', 'content': '好的，我記住了你最喜歡的顏色是藍色。'},
            {'role': 'user', 'content': '我最喜歡什麼顏色？'}
        ]
        
        print("📝 對話歷史：")
        for msg in messages:
            role_icon = "🤖" if msg['role'] == 'assistant' else "👤"
            if msg['role'] != 'system':
                print(f"{role_icon} {msg['role']}: {msg['content']}")
        
        print("\n正在生成回應...")
        response = self.client.chat(
            model=model,
            messages=messages
        )
        
        print(f"🤖 助理: {response['message']['content']}")
        
        # 顯示 token 使用情況
        if 'prompt_eval_count' in response:
            print(f"\n📊 Token 統計:")
            print(f"   輸入 tokens: {response.get('prompt_eval_count', 0)}")
            print(f"   輸出 tokens: {response.get('eval_count', 0)}")
    
    def test_embeddings(self, model: str = "gemma:2b") -> None:
        """測試 embeddings API"""
        print(f"\n🔬 測試 Embeddings API - {model}")
        print("=" * 60)
        
        texts = [
            "機器學習是人工智慧的一個分支",
            "深度學習使用神經網路",
            "今天天氣很好"
        ]
        
        embeddings = []
        for text in texts:
            print(f"\n📝 文字: {text}")
            
            try:
                response = self.client.embeddings(
                    model=model,
                    prompt=text
                )
                
                embedding = response['embedding']
                embeddings.append(embedding)
                
                print(f"   向量維度: {len(embedding)}")
                print(f"   前5個值: {embedding[:5]}")
                
            except Exception as e:
                print(f"   ❌ 錯誤: {e}")
        
        # 計算相似度（餘弦相似度）
        if len(embeddings) >= 2:
            print("\n📐 相似度分析：")
            import math
            
            def cosine_similarity(v1, v2):
                dot_product = sum(a * b for a, b in zip(v1, v2))
                norm1 = math.sqrt(sum(a * a for a in v1))
                norm2 = math.sqrt(sum(b * b for b in v2))
                return dot_product / (norm1 * norm2)
            
            for i in range(len(texts)):
                for j in range(i + 1, len(texts)):
                    similarity = cosine_similarity(embeddings[i], embeddings[j])
                    print(f"   '{texts[i][:20]}...' vs '{texts[j][:20]}...': {similarity:.3f}")
    
    def benchmark_performance(self, model: str = "gemma:2b") -> Dict:
        """效能基準測試"""
        print(f"\n⚡ 效能基準測試 - {model}")
        print("=" * 60)
        
        test_prompts = [
            "Hello, how are you?",
            "寫一個10字的句子",
            "1+1等於多少？"
        ]
        
        results = []
        
        for prompt in test_prompts:
            print(f"\n測試: {prompt}")
            
            # 測試生成速度
            start = time.time()
            response = self.client.generate(
                model=model,
                prompt=prompt,
                options={'max_tokens': 50}
            )
            elapsed = time.time() - start
            
            tokens = response.get('eval_count', 0)
            tokens_per_sec = tokens / elapsed if elapsed > 0 else 0
            
            print(f"   時間: {elapsed:.2f}秒")
            print(f"   Tokens: {tokens}")
            print(f"   速度: {tokens_per_sec:.1f} tokens/秒")
            
            results.append({
                'prompt': prompt,
                'time': elapsed,
                'tokens': tokens,
                'tokens_per_sec': tokens_per_sec
            })
        
        # 總結
        avg_speed = sum(r['tokens_per_sec'] for r in results) / len(results)
        print(f"\n📊 平均速度: {avg_speed:.1f} tokens/秒")
        
        return {
            'model': model,
            'average_tokens_per_sec': avg_speed,
            'results': results
        }
    
    def test_model_options(self, model: str = "gemma:2b") -> None:
        """測試各種模型選項"""
        print(f"\n🎛️ 測試模型選項 - {model}")
        print("=" * 60)
        
        prompt = "寫一個關於 AI 的句子"
        
        # 測試不同的選項組合
        options_list = [
            {'temperature': 0.1, 'seed': 42},
            {'temperature': 0.9, 'seed': 42},
            {'temperature': 0.5, 'top_p': 0.9},
            {'temperature': 0.5, 'top_k': 40},
            {'temperature': 0.5, 'repeat_penalty': 1.5}
        ]
        
        for i, options in enumerate(options_list, 1):
            print(f"\n測試 {i}: {options}")
            response = self.client.generate(
                model=model,
                prompt=prompt,
                options=options
            )
            print(f"   結果: {response['response']}")

def print_menu():
    """顯示選單"""
    print("\n" + "=" * 60)
    print("📚 Ollama API 探索工具")
    print("=" * 60)
    print("1. 列出已安裝模型")
    print("2. 顯示模型詳細資訊")
    print("3. 測試 Generate API")
    print("4. 測試 Chat API")
    print("5. 測試 Embeddings")
    print("6. 效能基準測試")
    print("7. 測試模型選項")
    print("8. 執行所有測試")
    print("0. 結束")
    print("-" * 60)

def main():
    """主程式"""
    explorer = OllamaExplorer()
    
    # 檢查連接
    try:
        models = ollama.list()
        if not models['models']:
            print("⚠️  沒有找到已安裝的模型")
            print("請先安裝模型：ollama pull gemma:2b")
            return
    except Exception as e:
        print(f"❌ 無法連接到 Ollama: {e}")
        return
    
    default_model = "gemma:2b"
    
    while True:
        print_menu()
        
        choice = input("請選擇 (0-8): ").strip()
        
        if choice == '0':
            print("👋 再見！")
            break
        
        elif choice == '1':
            explorer.list_models()
        
        elif choice == '2':
            model = input(f"輸入模型名稱 [{default_model}]: ").strip() or default_model
            explorer.show_model_info(model)
        
        elif choice == '3':
            model = input(f"輸入模型名稱 [{default_model}]: ").strip() or default_model
            explorer.test_generate_api(model)
        
        elif choice == '4':
            model = input(f"輸入模型名稱 [{default_model}]: ").strip() or default_model
            explorer.test_chat_api(model)
        
        elif choice == '5':
            model = input(f"輸入模型名稱 [{default_model}]: ").strip() or default_model
            explorer.test_embeddings(model)
        
        elif choice == '6':
            model = input(f"輸入模型名稱 [{default_model}]: ").strip() or default_model
            explorer.benchmark_performance(model)
        
        elif choice == '7':
            model = input(f"輸入模型名稱 [{default_model}]: ").strip() or default_model
            explorer.test_model_options(model)
        
        elif choice == '8':
            model = input(f"輸入模型名稱 [{default_model}]: ").strip() or default_model
            print("\n🚀 執行所有測試...")
            explorer.list_models()
            explorer.show_model_info(model)
            explorer.test_generate_api(model)
            explorer.test_chat_api(model)
            explorer.test_embeddings(model)
            explorer.benchmark_performance(model)
            explorer.test_model_options(model)
            print("\n✅ 所有測試完成！")
        
        else:
            print("❌ 無效的選擇")
        
        input("\n按 Enter 繼續...")

if __name__ == "__main__":
    main()