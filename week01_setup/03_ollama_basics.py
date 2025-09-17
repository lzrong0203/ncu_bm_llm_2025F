#!/usr/bin/env python3
"""
Week 1 - Lesson 3: Ollama API 基礎
深入了解 Ollama API 的各種功能範例
"""

import ollama
import time
import math

def example1_list_models():
    """範例1: 列出所有已安裝的模型"""
    print("\n範例1: 列出已安裝的模型")
    print("=" * 50)

    models = ollama.list()
    for model in models['models']:
        size_mb = model['size'] / (1024 * 1024)
        print(f"模型: {model['name']}")
        print(f"  大小: {size_mb:.1f} MB")
        print(f"  修改時間: {model['modified_at'][:10]}")

def example2_generate_api():
    """範例2: 測試 generate API (原始文字生成)"""
    print("\n範例2: Generate API 文字生成")
    print("=" * 50)

    model = "gemma:2b"
    prompts = [
        "完成這個句子：人工智慧的未來",
        "解釋什麼是深度學習："
    ]

    for prompt in prompts:
        print(f"\nPrompt: {prompt}")
        response = ollama.generate(
            model=model,
            prompt=prompt,
            options={'temperature': 0.7, 'max_tokens': 50}
        )
        print(f"Response: {response['response']}")

def example3_chat_api():
    """範例3: 測試 chat API (對話模式)"""
    print("\n範例3: Chat API 對話模式")
    print("=" * 50)

    model = "gemma:2b"

    # 建立對話歷史
    messages = [
        {'role': 'system', 'content': '你是一個友善的助理'},
        {'role': 'user', 'content': '請記住我最喜歡的顏色是藍色'},
        {'role': 'assistant', 'content': '好的，我記住了你最喜歡的顏色是藍色。'},
        {'role': 'user', 'content': '我最喜歡什麼顏色？'}
    ]

    response = ollama.chat(model=model, messages=messages)

    print("對話歷史：")
    for msg in messages[1:]:  # 跳過 system
        role = "助理" if msg['role'] == 'assistant' else "用戶"
        print(f"{role}: {msg['content']}")

    print(f"助理回應: {response['message']['content']}")

def example4_embeddings():
    """範例4: 測試 embeddings API"""
    print("\n範例4: Embeddings 向量生成")
    print("=" * 50)

    model = "gemma:2b"
    texts = [
        "機器學習是人工智慧的一個分支",
        "深度學習使用神經網路",
        "今天天氣很好"
    ]

    embeddings = []
    for text in texts:
        response = ollama.embeddings(model=model, prompt=text)
        embedding = response['embedding']
        embeddings.append(embedding)
        print(f"文字: {text}")
        print(f"  向量維度: {len(embedding)}")

    # 計算相似度
    def cosine_similarity(v1, v2):
        dot_product = sum(a * b for a, b in zip(v1, v2))
        norm1 = math.sqrt(sum(a * a for a in v1))
        norm2 = math.sqrt(sum(b * b for b in v2))
        return dot_product / (norm1 * norm2)

    print("\n相似度分析：")
    similarity_01 = cosine_similarity(embeddings[0], embeddings[1])
    similarity_02 = cosine_similarity(embeddings[0], embeddings[2])
    print(f"  句子1 vs 句子2: {similarity_01:.3f}")
    print(f"  句子1 vs 句子3: {similarity_02:.3f}")

def example5_model_options():
    """範例5: 測試不同的模型參數"""
    print("\n範例5: 不同 Temperature 的效果")
    print("=" * 50)

    model = "gemma:2b"
    prompt = "寫一個關於 AI 的句子"

    temperatures = [0.1, 0.5, 0.9]
    for temp in temperatures:
        response = ollama.generate(
            model=model,
            prompt=prompt,
            options={'temperature': temp, 'seed': 42}
        )
        print(f"Temperature {temp}: {response['response']}")

def example6_performance():
    """範例6: 效能測試"""
    print("\n範例6: 效能基準測試")
    print("=" * 50)

    model = "gemma:2b"
    prompt = "Hello, how are you?"

    start = time.time()
    response = ollama.generate(
        model=model,
        prompt=prompt,
        options={'max_tokens': 30}
    )
    elapsed = time.time() - start

    tokens = response.get('eval_count', 0)
    tokens_per_sec = tokens / elapsed if elapsed > 0 else 0

    print(f"Prompt: {prompt}")
    print(f"Response: {response['response']}")
    print(f"生成時間: {elapsed:.2f} 秒")
    print(f"生成 tokens: {tokens}")
    print(f"速度: {tokens_per_sec:.1f} tokens/秒")

def main():
    """執行所有範例"""
    print("Ollama API 基礎範例")
    print("=" * 50)

    # 檢查 Ollama
    try:
        models = ollama.list()
        if not models['models']:
            print("請先安裝模型：ollama pull gemma:2b")
            return
        print(f"已連接 Ollama，共 {len(models['models'])} 個模型可用")
    except Exception as e:
        print(f"請先啟動 Ollama: ollama serve")
        return

    # 執行範例
    example1_list_models()
    example2_generate_api()
    example3_chat_api()
    example4_embeddings()
    example5_model_options()
    example6_performance()

    print("\n所有範例執行完成！")

if __name__ == "__main__":
    main()