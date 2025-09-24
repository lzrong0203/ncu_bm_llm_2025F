#!/usr/bin/env python3
"""
Week 5 - Lesson 5: 商業聊天機器人綜合專案
整合所有概念，建立完整的客服系統
"""

import streamlit as st
from langchain_community.llms import Ollama
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain.chains import LLMChain, ConversationChain
from langchain.memory import ConversationBufferMemory, ConversationSummaryBufferMemory
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import json
import random
from pathlib import Path
from enum import Enum

# 資料模型定義
class OrderStatus(Enum):
    """訂單狀態"""
    PENDING = "待處理"
    PROCESSING = "處理中"
    SHIPPED = "已出貨"
    DELIVERED = "已送達"
    CANCELLED = "已取消"

class Product(BaseModel):
    """產品模型"""
    id: str
    name: str
    category: str
    price: float
    stock: int
    description: str
    features: List[str]

class Order(BaseModel):
    """訂單模型"""
    order_id: str
    customer_name: str
    products: List[Dict[str, Any]]
    total_amount: float
    status: str
    order_date: str
    expected_delivery: str

class CustomerQuery(BaseModel):
    """客戶查詢分類"""
    category: str = Field(description="查詢類別：產品/訂單/技術支援/其他")
    intent: str = Field(description="具體意圖")
    entities: List[str] = Field(description="提取的實體（如產品名、訂單號）")
    urgency: str = Field(description="緊急程度：高/中/低")

class BusinessChatbot:
    """商業客服聊天機器人"""

    def __init__(self, model="gemma3:1b"):
        """初始化聊天機器人"""
        self.llm = Ollama(model=model, temperature=0.7)
        self.chat_model = ChatOllama(model=model)
        self.memory = ConversationBufferMemory()
        self.load_data()
        self.init_chains()

    def load_data(self):
        """載入商業資料"""
        # 載入產品資料
        products_file = Path("week05_langchain/data/products.json")
        if products_file.exists():
            with open(products_file, 'r', encoding='utf-8') as f:
                self.products = json.load(f)
        else:
            # 使用預設資料
            self.products = self.get_default_products()

        # 載入FAQ
        faqs_file = Path("week05_langchain/data/faqs.json")
        if faqs_file.exists():
            with open(faqs_file, 'r', encoding='utf-8') as f:
                self.faqs = json.load(f)
        else:
            self.faqs = self.get_default_faqs()

        # 模擬訂單資料
        self.orders = self.generate_sample_orders()

    def get_default_products(self) -> List[Dict]:
        """預設產品資料"""
        return [
            {
                "id": "LAPTOP001",
                "name": "ProBook 15",
                "category": "筆記型電腦",
                "price": 35999,
                "stock": 50,
                "description": "高效能商務筆電",
                "features": ["Intel i7", "16GB RAM", "512GB SSD", "15.6吋螢幕"]
            },
            {
                "id": "PHONE001",
                "name": "SmartPhone X",
                "category": "智慧手機",
                "price": 25999,
                "stock": 100,
                "description": "旗艦級智慧手機",
                "features": ["5G", "128GB", "三鏡頭", "快速充電"]
            },
            {
                "id": "WATCH001",
                "name": "FitWatch Pro",
                "category": "智慧手錶",
                "price": 8999,
                "stock": 75,
                "description": "健康監測智慧手錶",
                "features": ["心率監測", "GPS", "防水", "7天續航"]
            },
            {
                "id": "EARPHONE001",
                "name": "SoundBuds Pro",
                "category": "無線耳機",
                "price": 4999,
                "stock": 200,
                "description": "主動降噪無線耳機",
                "features": ["ANC降噪", "30小時續航", "快速配對", "防水IPX4"]
            }
        ]

    def get_default_faqs(self) -> List[Dict]:
        """預設FAQ資料"""
        return [
            {
                "question": "如何退換貨？",
                "answer": "商品收到7天內，保持完整包裝可申請退換貨。請聯繫客服或在會員中心申請。"
            },
            {
                "question": "運費如何計算？",
                "answer": "滿2000元免運費，未滿收取80元運費。偏遠地區另計。"
            },
            {
                "question": "有哪些付款方式？",
                "answer": "支援信用卡、貨到付款、銀行轉帳、行動支付（Line Pay、街口支付）。"
            },
            {
                "question": "保固期多久？",
                "answer": "電子產品享有一年原廠保固，配件類三個月保固。"
            }
        ]

    def generate_sample_orders(self) -> List[Dict]:
        """生成示範訂單"""
        orders = []
        statuses = list(OrderStatus)
        names = ["張小明", "李美玲", "王大華", "陳雅婷", "林志豪"]

        for i in range(5):
            order_date = datetime.now() - timedelta(days=random.randint(1, 30))
            orders.append({
                "order_id": f"ORD{str(i+1).zfill(5)}",
                "customer_name": random.choice(names),
                "products": [random.choice(self.products)],
                "total_amount": random.randint(5000, 50000),
                "status": random.choice(statuses).value,
                "order_date": order_date.strftime("%Y-%m-%d"),
                "expected_delivery": (order_date + timedelta(days=3)).strftime("%Y-%m-%d")
            })

        return orders

    def init_chains(self):
        """初始化處理鏈"""
        # 查詢分類鏈
        self.classifier_prompt = PromptTemplate(
            input_variables=["query"],
            template="""分析以下客戶查詢，判斷類別和意圖：

客戶查詢：{query}

請分析：
1. 類別（產品諮詢/訂單查詢/技術支援/其他）
2. 具體意圖
3. 關鍵實體（產品名、訂單號等）
4. 緊急程度（高/中/低）

回應格式：
類別：
意圖：
實體：
緊急度："""
        )

        # 回應生成鏈
        self.response_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""你是一個專業的客服代表。
請友善、專業地回答客戶問題。
如果需要具體資料，會提供給你。
使用繁體中文回答。"""),
            HumanMessage(content="{query}\n\n相關資料：{context}")
        ])

    def classify_query(self, query: str) -> Dict[str, Any]:
        """分類客戶查詢"""
        chain = LLMChain(llm=self.llm, prompt=self.classifier_prompt)
        result = chain.invoke({"query": query})

        # 簡單解析（實際應使用結構化輸出）
        lines = result['text'].strip().split('\n')
        classification = {
            "category": "其他",
            "intent": "一般詢問",
            "entities": [],
            "urgency": "中"
        }

        for line in lines:
            if "類別" in line:
                classification["category"] = line.split("：")[-1].strip()
            elif "意圖" in line:
                classification["intent"] = line.split("：")[-1].strip()
            elif "實體" in line:
                entities = line.split("：")[-1].strip()
                classification["entities"] = [e.strip() for e in entities.split(",")]
            elif "緊急" in line:
                classification["urgency"] = line.split("：")[-1].strip()

        return classification

    def search_products(self, criteria: str) -> List[Dict]:
        """搜尋產品"""
        results = []
        criteria_lower = criteria.lower()

        for product in self.products:
            if (criteria_lower in product['name'].lower() or
                criteria_lower in product['category'].lower() or
                criteria_lower in product['description'].lower()):
                results.append(product)

        return results

    def search_order(self, order_id: str) -> Optional[Dict]:
        """搜尋訂單"""
        for order in self.orders:
            if order['order_id'].lower() == order_id.lower():
                return order
        return None

    def search_faq(self, query: str) -> List[Dict]:
        """搜尋FAQ"""
        results = []
        query_lower = query.lower()

        for faq in self.faqs:
            if (query_lower in faq['question'].lower() or
                query_lower in faq['answer'].lower()):
                results.append(faq)

        return results[:3]  # 最多返回3個

    def handle_product_query(self, query: str, entities: List[str]) -> str:
        """處理產品查詢"""
        # 搜尋相關產品
        products = []
        for entity in entities:
            products.extend(self.search_products(entity))

        if not products and len(entities) > 0:
            products = self.search_products(entities[0])

        if products:
            context = "找到以下產品：\n"
            for p in products[:3]:  # 最多顯示3個
                context += f"- {p['name']} ({p['category']})\n"
                context += f"  價格：${p['price']}\n"
                context += f"  特色：{', '.join(p['features'][:3])}\n"
                context += f"  庫存：{p['stock']}件\n\n"
        else:
            context = "沒有找到相關產品。"

        return self.generate_response(query, context)

    def handle_order_query(self, query: str, entities: List[str]) -> str:
        """處理訂單查詢"""
        order = None
        for entity in entities:
            if "ORD" in entity.upper():
                order = self.search_order(entity)
                if order:
                    break

        if order:
            context = f"""訂單資訊：
訂單編號：{order['order_id']}
客戶姓名：{order['customer_name']}
訂單金額：${order['total_amount']}
訂單狀態：{order['status']}
下單日期：{order['order_date']}
預計送達：{order['expected_delivery']}"""
        else:
            context = "找不到訂單資料。請確認訂單編號是否正確。"

        return self.generate_response(query, context)

    def handle_support_query(self, query: str) -> str:
        """處理技術支援查詢"""
        # 搜尋相關FAQ
        faqs = self.search_faq(query)

        if faqs:
            context = "相關說明：\n"
            for faq in faqs:
                context += f"Q: {faq['question']}\n"
                context += f"A: {faq['answer']}\n\n"
        else:
            context = "請詳細描述您遇到的問題，我們會盡快為您解答。"

        return self.generate_response(query, context)

    def generate_response(self, query: str, context: str) -> str:
        """生成回應"""
        messages = [
            SystemMessage(content="你是專業的客服代表，請友善且專業地回答。"),
            HumanMessage(content=f"客戶問題：{query}\n\n相關資料：{context}")
        ]

        response = self.chat_model.invoke(messages)
        return response.content

    def chat(self, user_input: str) -> Tuple[str, Dict[str, Any]]:
        """主要對話介面"""
        # 分類查詢
        classification = self.classify_query(user_input)

        # 根據分類處理
        if "產品" in classification["category"]:
            response = self.handle_product_query(user_input, classification["entities"])
        elif "訂單" in classification["category"]:
            response = self.handle_order_query(user_input, classification["entities"])
        elif "技術" in classification["category"] or "支援" in classification["category"]:
            response = self.handle_support_query(user_input)
        else:
            # 一般對話
            response = self.generate_response(user_input, "")

        # 儲存到記憶體
        self.memory.chat_memory.add_user_message(user_input)
        self.memory.chat_memory.add_ai_message(response)

        return response, classification

def create_streamlit_app():
    """建立 Streamlit 應用程式"""
    st.set_page_config(
        page_title="智慧客服系統",
        page_icon="🤖",
        layout="wide"
    )

    st.title("🤖 智慧客服聊天機器人")
    st.markdown("整合 LangChain + Ollama 的商業客服系統")

    # 初始化 session state
    if "chatbot" not in st.session_state:
        with st.spinner("初始化系統..."):
            st.session_state.chatbot = BusinessChatbot()
            st.session_state.messages = []

    # 側邊欄
    with st.sidebar:
        st.header("系統資訊")

        # 顯示產品列表
        st.subheader("📦 產品目錄")
        for product in st.session_state.chatbot.products:
            st.write(f"• {product['name']} - ${product['price']}")

        # 顯示訂單範例
        st.subheader("📋 訂單範例")
        for order in st.session_state.chatbot.orders[:3]:
            st.write(f"• {order['order_id']} - {order['status']}")

        # 常見問題
        st.subheader("❓ 常見問題")
        for faq in st.session_state.chatbot.faqs[:3]:
            st.write(f"• {faq['question']}")

        # 清除對話
        if st.button("清除對話記錄"):
            st.session_state.messages = []
            st.session_state.chatbot.memory.clear()
            st.rerun()

    # 主要對話區
    chat_container = st.container()

    # 顯示對話歷史
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])
                if "classification" in message:
                    with st.expander("分析結果"):
                        st.json(message["classification"])

    # 輸入區
    user_input = st.chat_input("請輸入您的問題...")

    if user_input:
        # 顯示用戶訊息
        with st.chat_message("user"):
            st.write(user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})

        # 生成回應
        with st.chat_message("assistant"):
            with st.spinner("思考中..."):
                response, classification = st.session_state.chatbot.chat(user_input)
                st.write(response)

                # 顯示分析結果
                with st.expander("查詢分析"):
                    st.json(classification)

        st.session_state.messages.append({
            "role": "assistant",
            "content": response,
            "classification": classification
        })

def test_chatbot():
    """測試聊天機器人（非 Streamlit 模式）"""
    print("="*60)
    print("測試商業聊天機器人")
    print("="*60)

    chatbot = BusinessChatbot()

    test_queries = [
        "你們有什麼筆電？",
        "我要查詢訂單ORD00001",
        "如何退貨？",
        "ProBook 15的規格是什麼？",
        "有哪些付款方式？"
    ]

    for query in test_queries:
        print(f"\n用戶: {query}")
        response, classification = chatbot.chat(query)
        print(f"分類: {classification['category']} / {classification['intent']}")
        print(f"客服: {response[:200]}...")
        print("-" * 40)

def main():
    """主程式"""
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--streamlit":
        # 執行 Streamlit 應用
        create_streamlit_app()
    else:
        # 執行測試模式
        print("="*60)
        print("Week 5 - Lesson 5: 商業聊天機器人綜合專案")
        print("="*60)

        # 檢查 Ollama
        try:
            test_llm = Ollama(model="gemma3:1b")
            test_llm.invoke("測試")
            print("✓ Ollama 連接成功\n")
        except Exception as e:
            print(f"✗ 請先啟動 Ollama: ollama serve")
            print(f"  錯誤: {e}")
            return

        # 執行測試
        test_chatbot()

        print("\n" + "="*60)
        print("專案功能總結：")
        print("1. 智慧查詢分類：自動識別客戶意圖")
        print("2. 產品查詢：搜尋和推薦產品")
        print("3. 訂單追蹤：查詢訂單狀態")
        print("4. FAQ支援：回答常見問題")
        print("5. 對話記憶：保持上下文連貫")
        print("6. Web介面：Streamlit整合")
        print("\n執行 'streamlit run 05_business_chatbot.py' 啟動Web介面")
        print("="*60)

if __name__ == "__main__":
    main()