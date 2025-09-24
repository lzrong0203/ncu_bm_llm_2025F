#!/usr/bin/env python3
"""
Week 1 - Lesson 1: Hello LLM
第一個本地 LLM 程式範例
"""

import ollama

def example1_simple_chat():
    """範例1: 簡單對話"""
    print("\n範例1: 簡單對話")
    print("=" * 50)

    model = "gemma3:1b"  # 使用輕量模型作為範例
    prompt = "介紹一下你自己，用繁體中文回答"

    response = ollama.chat(
        model=model,
        messages=[{
            'role': 'user',
            'content': prompt
        }]
    )

    print(f"提示詞: {prompt}")
    print(f"回應: {response['message']['content']}")
    print(f"生成時間: {response['total_duration'] / 1_000_000_000:.2f} 秒")

def example2_streaming_chat():
    """範例2: 串流對話（打字機效果）"""
    print("\n範例2: 串流對話")
    print("=" * 50)

    model = "gemma3:1b"
    prompt = "什麼是機器學習？請用一句話解釋"

    print(f"提示詞: {prompt}")
    print("回應: ", end="")

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
    print()

def example3_temperature_effect():
    """範例3: Temperature 參數效果"""
    print("\n範例3: Temperature 參數對輸出的影響")
    print("=" * 50)

    model = "gemma3:1b"
    prompt = "寫一句關於 AI 的句子"
    temperatures = [0.1, 0.5, 0.9]

    print(f"提示詞: {prompt}\n")

    for temp in temperatures:
        response = ollama.generate(
            model=model,
            prompt=prompt,
            options={'temperature': temp}
        )
        print(f"Temperature = {temp}: {response['response']}")

def main():
    """主程式 - 執行所有範例"""
    print("Week 1: Hello LLM 範例程式")
    print("=" * 50)

    # 檢查 Ollama 連接
    try:
        models = ollama.list()
        print(f"已連接 Ollama，共 {len(models['models'])} 個模型可用")
    except Exception as e:
        print(f"請先啟動 Ollama: ollama serve")
        return

    # 執行範例
    example1_simple_chat()
    example2_streaming_chat()
    example3_temperature_effect()

    print("\n所有範例執行完成！")

if __name__ == "__main__":
    main()