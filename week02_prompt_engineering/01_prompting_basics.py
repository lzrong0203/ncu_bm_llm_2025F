#!/usr/bin/env python3
"""
Week 2 - Lesson 1: Prompt Engineering 基礎
掌握 Zero-shot, Few-shot, Chain-of-Thought 技巧
"""

import ollama
import json
from typing import List, Dict, Any
from dataclasses import dataclass
import time

@dataclass
class PromptResult:
    """Prompt 執行結果"""
    prompt: str
    response: str
    technique: str
    execution_time: float
    tokens_used: int = 0

class PromptEngineeringBasics:
    """Prompt Engineering 基礎技巧示範"""
    
    def __init__(self, model: str = "gemma:2b"):
        self.model = model
        self.results = []
    
    def zero_shot_demo(self) -> List[PromptResult]:
        """Zero-shot Prompting 示範"""
        print("\n" + "=" * 60)
        print("🎯 Zero-shot Prompting")
        print("=" * 60)
        print("定義：直接給出任務描述，不提供範例\n")
        
        examples = [
            {
                "task": "情感分析",
                "prompt": """判斷以下評論的情感（正面/負面/中性）：

評論：這家餐廳的服務很好，但食物普通，價格偏高。
情感："""
            },
            {
                "task": "文字分類",
                "prompt": """將以下新聞標題分類（科技/體育/娛樂/政治）：

標題：新款 iPhone 發表，搭載更強大的 AI 功能
類別："""
            },
            {
                "task": "摘要生成",
                "prompt": """用一句話總結以下段落：

人工智慧正在改變我們的生活方式。從智慧型手機的語音助理，
到自動駕駛汽車，再到醫療診斷系統，AI 技術已經滲透到各個領域。
這些應用不僅提高了效率，也為人類帶來了前所未有的便利。

總結："""
            }
        ]
        
        results = []
        for example in examples:
            print(f"📝 任務: {example['task']}")
            print(f"Prompt:\n{example['prompt']}")
            
            start = time.time()
            response = ollama.generate(
                model=self.model,
                prompt=example['prompt'],
                options={'temperature': 0.3}  # 低溫度for一致性
            )
            elapsed = time.time() - start
            
            result = PromptResult(
                prompt=example['prompt'],
                response=response['response'],
                technique="Zero-shot",
                execution_time=elapsed,
                tokens_used=response.get('eval_count', 0)
            )
            
            print(f"💬 回應: {result.response}")
            print(f"⏱️  時間: {elapsed:.2f}秒")
            print("-" * 40)
            
            results.append(result)
            self.results.append(result)
        
        return results
    
    def few_shot_demo(self) -> List[PromptResult]:
        """Few-shot Prompting 示範"""
        print("\n" + "=" * 60)
        print("🎯 Few-shot Prompting")
        print("=" * 60)
        print("定義：提供 2-5 個範例，讓模型學習模式\n")
        
        examples = [
            {
                "task": "格式轉換",
                "prompt": """將日期轉換為標準格式 (YYYY-MM-DD)：

輸入：3月15日2024年
輸出：2024-03-15

輸入：2023年12月1日
輸出：2023-12-01

輸入：7月20日2025年
輸出：2025-07-20

輸入：2024年5月8日
輸出："""
            },
            {
                "task": "情感分數評分",
                "prompt": """根據評論給出情感分數（1-5分，5分最正面）：

評論：太棒了！完全超出預期！
分數：5

評論：還可以，沒什麼特別的
分數：3

評論：完全不推薦，浪費錢
分數：1

評論：相當不錯，物超所值
分數：4

評論：品質很好，但價格有點高
分數："""
            },
            {
                "task": "實體識別",
                "prompt": """從句子中提取人名、地點和組織：

句子：張三在微軟台北辦公室工作
結果：人名[張三], 地點[台北], 組織[微軟]

句子：李四昨天去了Google總部
結果：人名[李四], 地點[總部], 組織[Google]

句子：王五明天要去蘋果公司面試
結果：人名[王五], 地點[], 組織[蘋果公司]

句子：陳六在台積電新竹廠區上班
結果："""
            }
        ]
        
        results = []
        for example in examples:
            print(f"📝 任務: {example['task']}")
            print(f"Prompt:\n{example['prompt'][:200]}...")  # 顯示部分
            
            start = time.time()
            response = ollama.generate(
                model=self.model,
                prompt=example['prompt'],
                options={'temperature': 0.2}
            )
            elapsed = time.time() - start
            
            result = PromptResult(
                prompt=example['prompt'],
                response=response['response'],
                technique="Few-shot",
                execution_time=elapsed,
                tokens_used=response.get('eval_count', 0)
            )
            
            print(f"💬 回應: {result.response}")
            print(f"⏱️  時間: {elapsed:.2f}秒")
            print("-" * 40)
            
            results.append(result)
            self.results.append(result)
        
        return results
    
    def chain_of_thought_demo(self) -> List[PromptResult]:
        """Chain-of-Thought (CoT) Prompting 示範"""
        print("\n" + "=" * 60)
        print("🎯 Chain-of-Thought (CoT) Prompting")
        print("=" * 60)
        print("定義：引導模型逐步思考，展示推理過程\n")
        
        examples = [
            {
                "task": "數學問題",
                "prompt": """解決以下數學問題，請一步步思考：

問題：小明有 45 元，買了 3 個蘋果，每個蘋果 8 元，又買了一瓶水 12 元。
請問小明還剩多少錢？

讓我們一步步計算：
1. 首先計算蘋果的總價：
2. 然後計算總花費：
3. 最後計算剩餘的錢：

答案："""
            },
            {
                "task": "邏輯推理",
                "prompt": """請一步步分析這個邏輯問題：

問題：所有的貓都有尾巴。咪咪是一隻貓。湯姆有尾巴。
請問：我們能確定湯姆是貓嗎？

讓我們逐步分析：
1. 已知條件整理：
2. 邏輯關係分析：
3. 得出結論：

答案："""
            },
            {
                "task": "決策分析",
                "prompt": """幫助做出決策，請詳細分析：

情況：一家公司要決定是否推出新產品。
- 開發成本：100萬
- 預計第一年銷售：50萬
- 預計第二年銷售：80萬
- 市場競爭激烈
- 公司現金流充足

請一步步分析是否應該推出：
1. 成本效益分析：
2. 風險評估：
3. 機會成本：
4. 建議：

決策："""
            }
        ]
        
        results = []
        for example in examples:
            print(f"📝 任務: {example['task']}")
            print(f"Prompt:\n{example['prompt'][:150]}...")
            
            start = time.time()
            response = ollama.generate(
                model=self.model,
                prompt=example['prompt'],
                options={'temperature': 0.3}
            )
            elapsed = time.time() - start
            
            result = PromptResult(
                prompt=example['prompt'],
                response=response['response'],
                technique="Chain-of-Thought",
                execution_time=elapsed,
                tokens_used=response.get('eval_count', 0)
            )
            
            print(f"💬 回應:\n{result.response[:300]}...")
            print(f"⏱️  時間: {elapsed:.2f}秒")
            print("-" * 40)
            
            results.append(result)
            self.results.append(result)
        
        return results
    
    def zero_shot_cot_demo(self) -> List[PromptResult]:
        """Zero-shot Chain-of-Thought 示範"""
        print("\n" + "=" * 60)
        print("🎯 Zero-shot Chain-of-Thought")
        print("=" * 60)
        print("定義：使用魔法句子 'Let's think step by step'\n")
        
        examples = [
            {
                "task": "複雜計算",
                "prompt": """一家餐廳有 12 張桌子。每張桌子可坐 4 人。
今天有 3 個 15 人的團體預約。
請問餐廳還能接待多少散客？

Let's think step by step."""
            },
            {
                "task": "邏輯謎題",
                "prompt": """有三個盒子，標籤分別寫著「蘋果」、「橘子」、「蘋果和橘子」。
但所有標籤都貼錯了。你只能從一個盒子拿出一個水果來看。
要如何確定每個盒子的真實內容？

Let's think step by step."""
            }
        ]
        
        results = []
        for example in examples:
            print(f"📝 任務: {example['task']}")
            print(f"Prompt:\n{example['prompt']}")
            
            start = time.time()
            response = ollama.generate(
                model=self.model,
                prompt=example['prompt'],
                options={'temperature': 0.3}
            )
            elapsed = time.time() - start
            
            result = PromptResult(
                prompt=example['prompt'],
                response=response['response'],
                technique="Zero-shot-CoT",
                execution_time=elapsed,
                tokens_used=response.get('eval_count', 0)
            )
            
            print(f"💬 回應:\n{result.response[:400]}...")
            print(f"⏱️  時間: {elapsed:.2f}秒")
            print("-" * 40)
            
            results.append(result)
            self.results.append(result)
        
        return results
    
    def compare_techniques(self, task: str) -> Dict[str, PromptResult]:
        """比較不同 Prompting 技巧在同一任務上的表現"""
        print("\n" + "=" * 60)
        print("🔬 技巧比較實驗")
        print("=" * 60)
        print(f"任務: {task}\n")
        
        base_task = "分析這段文字的主要觀點：\n\n" + \
                   "遠端工作正在改變傳統的辦公模式。雖然提供了更大的彈性和工作生活平衡，" + \
                   "但也帶來了溝通挑戰和團隊凝聚力的問題。企業需要找到適當的平衡點。"
        
        techniques = {
            "Zero-shot": base_task + "\n\n主要觀點：",
            
            "Few-shot": """分析文字的主要觀點：

文字：社交媒體改變了人們的溝通方式，帶來便利但也造成隱私問題。
主要觀點：社交媒體有利有弊，便利性vs隱私權的權衡。

文字：""" + base_task.split('：\n\n')[1] + """
主要觀點：""",
            
            "Chain-of-Thought": base_task + """

請一步步分析：
1. 識別關鍵詞：
2. 找出正面觀點：
3. 找出負面觀點：
4. 總結主要論點：

主要觀點：""",
            
            "Zero-shot-CoT": base_task + "\n\nLet's think step by step to identify the main points:"
        }
        
        comparison = {}
        
        for technique, prompt in techniques.items():
            print(f"\n📌 {technique}")
            print(f"Prompt: {prompt[:100]}...")
            
            start = time.time()
            response = ollama.generate(
                model=self.model,
                prompt=prompt,
                options={'temperature': 0.3}
            )
            elapsed = time.time() - start
            
            result = PromptResult(
                prompt=prompt,
                response=response['response'],
                technique=technique,
                execution_time=elapsed,
                tokens_used=response.get('eval_count', 0)
            )
            
            comparison[technique] = result
            
            print(f"💬 回應: {result.response[:200]}...")
            print(f"⏱️  時間: {elapsed:.2f}秒")
            print(f"📊 Tokens: {result.tokens_used}")
        
        # 分析比較結果
        print("\n" + "=" * 60)
        print("📊 效能比較")
        print("=" * 60)
        
        for technique, result in comparison.items():
            print(f"{technique:15} | 時間: {result.execution_time:.2f}s | Tokens: {result.tokens_used}")
        
        return comparison
    
    def save_results(self, filename: str = "prompt_results.json") -> None:
        """儲存實驗結果"""
        results_dict = []
        for result in self.results:
            results_dict.append({
                'technique': result.technique,
                'prompt': result.prompt[:200],  # 儲存部分prompt
                'response': result.response[:500],  # 儲存部分response
                'execution_time': result.execution_time,
                'tokens_used': result.tokens_used
            })
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results_dict, f, ensure_ascii=False, indent=2)
        
        print(f"\n✅ 結果已儲存到 {filename}")

def main():
    """主程式"""
    print("=" * 60)
    print("🎓 Prompt Engineering 基礎技巧")
    print("=" * 60)
    
    # 檢查 Ollama
    try:
        models = ollama.list()
        print("✅ 已連接到 Ollama")
    except Exception as e:
        print(f"❌ 無法連接到 Ollama: {e}")
        return
    
    # 選擇模型
    model = input("\n選擇模型 [預設: gemma:2b]: ").strip() or "gemma:2b"
    
    # 建立示範物件
    demo = PromptEngineeringBasics(model=model)
    
    while True:
        print("\n" + "=" * 60)
        print("選擇示範項目：")
        print("1. Zero-shot Prompting")
        print("2. Few-shot Prompting")
        print("3. Chain-of-Thought")
        print("4. Zero-shot CoT")
        print("5. 技巧比較")
        print("6. 執行所有示範")
        print("7. 儲存結果")
        print("0. 結束")
        print("-" * 60)
        
        choice = input("選擇 (0-7): ").strip()
        
        if choice == '0':
            break
        elif choice == '1':
            demo.zero_shot_demo()
        elif choice == '2':
            demo.few_shot_demo()
        elif choice == '3':
            demo.chain_of_thought_demo()
        elif choice == '4':
            demo.zero_shot_cot_demo()
        elif choice == '5':
            demo.compare_techniques("文字分析")
        elif choice == '6':
            demo.zero_shot_demo()
            demo.few_shot_demo()
            demo.chain_of_thought_demo()
            demo.zero_shot_cot_demo()
            demo.compare_techniques("文字分析")
        elif choice == '7':
            demo.save_results()
        else:
            print("❌ 無效選擇")
        
        if choice != '0':
            input("\n按 Enter 繼續...")
    
    print("\n👋 課程結束！")

if __name__ == "__main__":
    main()