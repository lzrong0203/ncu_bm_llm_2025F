#!/usr/bin/env python3
"""
Week 2 - Lesson 1: Prompt Engineering 基礎
Zero-shot, Few-shot, Chain-of-Thought 技巧範例
"""

import ollama
import time

def example1_zero_shot():
    """範例1: Zero-shot Prompting - 直接給任務，無範例"""
    print("\n範例1: Zero-shot Prompting")
    print("=" * 50)

    model = "gemma:2b"

    # 情感分析
    prompt1 = """判斷以下評論的情感（正面/負面/中性）：

評論：這家餐廳的服務很好，但食物普通，價格偏高。
情感："""

    response = ollama.generate(model=model, prompt=prompt1)
    print("情感分析任務：")
    print(f"  評論：這家餐廳的服務很好，但食物普通，價格偏高")
    print(f"  模型判斷：{response['response']}")

    # 文字分類
    prompt2 = """將以下新聞標題分類（科技/體育/娛樂/政治）：

標題：新款 iPhone 發表，搭載更強大的 AI 功能
類別："""

    response = ollama.generate(model=model, prompt=prompt2)
    print("\n文字分類任務：")
    print(f"  標題：新款 iPhone 發表，搭載更強大的 AI 功能")
    print(f"  模型分類：{response['response']}")

def example2_few_shot():
    """範例2: Few-shot Prompting - 提供範例讓模型學習"""
    print("\n範例2: Few-shot Prompting")
    print("=" * 50)

    model = "gemma:2b"

    # 格式轉換範例
    prompt = """將日期轉換為標準格式 (YYYY-MM-DD)：

輸入：3月15日2024年
輸出：2024-03-15

輸入：2023年12月1日
輸出：2023-12-01

輸入：2024年5月8日
輸出："""

    response = ollama.generate(model=model, prompt=prompt)
    print("日期格式轉換：")
    print("  輸入：2024年5月8日")
    print(f"  輸出：{response['response']}")

    # 情感評分範例
    prompt2 = """根據評論給出情感分數（1-5分，5分最正面）：

評論：太棒了！完全超出預期！
分數：5

評論：還可以，沒什麼特別的
分數：3

評論：完全不推薦，浪費錢
分數：1

評論：品質很好，但價格有點高
分數："""

    response = ollama.generate(model=model, prompt=prompt2)
    print("\n情感評分：")
    print("  評論：品質很好，但價格有點高")
    print(f"  分數：{response['response']}")

def example3_chain_of_thought():
    """範例3: Chain-of-Thought - 引導逐步思考"""
    print("\n範例3: Chain-of-Thought Prompting")
    print("=" * 50)

    model = "gemma:2b"

    prompt = """解決以下數學問題，請一步步思考：

問題：小明有 45 元，買了 3 個蘋果，每個蘋果 8 元，又買了一瓶水 12 元。
請問小明還剩多少錢？

讓我們一步步計算：
1. 首先計算蘋果的總價：
2. 然後計算總花費：
3. 最後計算剩餘的錢：

答案："""

    response = ollama.generate(model=model, prompt=prompt)
    print("數學問題（逐步推理）：")
    print("  問題：小明有45元，買3個蘋果(每個8元)和一瓶水(12元)")
    print(f"  推理過程：\n{response['response']}")

def example4_zero_shot_cot():
    """範例4: Zero-shot Chain-of-Thought - 魔法句子"""
    print("\n範例4: Zero-shot Chain-of-Thought")
    print("=" * 50)

    model = "gemma:2b"

    prompt = """一家餐廳有 12 張桌子。每張桌子可坐 4 人。
今天有 3 個 15 人的團體預約。
請問餐廳還能接待多少散客？

Let's think step by step."""

    response = ollama.generate(model=model, prompt=prompt)
    print("使用魔法句子 'Let's think step by step'：")
    print("  問題：計算餐廳剩餘座位")
    print(f"  模型推理：\n{response['response'][:300]}...")

def example5_technique_comparison():
    """範例5: 比較不同技巧的效果"""
    print("\n範例5: 技巧效果比較")
    print("=" * 50)

    model = "gemma:2b"
    task = "分析文字：遠端工作提供彈性但也帶來溝通挑戰"

    # Zero-shot
    prompt1 = "分析這段文字的主要觀點：遠端工作提供彈性但也帶來溝通挑戰。\n主要觀點："
    start = time.time()
    response1 = ollama.generate(model=model, prompt=prompt1)
    time1 = time.time() - start

    # With CoT
    prompt2 = """分析這段文字的主要觀點：遠端工作提供彈性但也帶來溝通挑戰。
Let's think step by step:"""
    start = time.time()
    response2 = ollama.generate(model=model, prompt=prompt2)
    time2 = time.time() - start

    print(f"Zero-shot (耗時 {time1:.2f}秒):")
    print(f"  {response1['response'][:100]}...")
    print(f"\nWith CoT (耗時 {time2:.2f}秒):")
    print(f"  {response2['response'][:100]}...")

def main():
    """執行所有範例"""
    print("Prompt Engineering 基礎技巧範例")
    print("=" * 50)

    # 檢查 Ollama
    try:
        ollama.list()
        print("已連接到 Ollama")
    except Exception as e:
        print(f"請先啟動 Ollama: ollama serve")
        return

    # 執行範例
    example1_zero_shot()
    example2_few_shot()
    example3_chain_of_thought()
    example4_zero_shot_cot()
    example5_technique_comparison()

    print("\n所有範例執行完成！")

if __name__ == "__main__":
    main()