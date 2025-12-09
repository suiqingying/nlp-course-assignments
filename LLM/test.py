import requests
import json

# ================= 配置 =================
API_KEY = "sk_317d87cb3cf64fde228486c6d3d397b181eee1c7b42865a3ae5f9e1395f991d3"
BASE_URL = "http://www.claudecodeserver.top/api/v1/messages"

def test_bypass_403():
    print(f"🕵️ 正在尝试伪装成 Claude Code 绕过 403...")
    
    # 核心策略：尝试模拟官方工具的特征
    # 我们轮询几个可能的 User-Agent，看看哪个能骗过服务器
    user_agents_to_try = [
        # 1. 模拟 Claude Code 命令行工具 (最可能的通行证)
        "claude-code/0.1.0 (darwin-x64; node-v20.10.0)", 
        # 2. 模拟浏览器 (通用伪装)
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
        # 3. 模拟 Anthropic 官方 Python SDK
        "anthropic-python/0.15.0"
    ]

    payload = {
        "model": "claude-3-5-sonnet-20241022",
        "max_tokens": 1024,
        "messages": [
            {"role": "user", "content": "Hello"}
        ]
    }

    for ua in user_agents_to_try:
        print(f"\n👉 尝试伪装 User-Agent: {ua}")
        
        headers = {
            "content-type": "application/json",
            "anthropic-version": "2023-06-01",
            "x-api-key": API_KEY,  # Anthropic 协议用这个头
            "User-Agent": ua       # 【关键】替换身份标识
        }

        try:
            response = requests.post(BASE_URL, headers=headers, json=payload, timeout=10)
            
            print(f"   状态码: {response.status_code}")
            
            if response.status_code == 200:
                print("   ✅ 成功绕过！服务器被骗过去了。")
                print("   回复:", response.json().get('content', [{}])[0].get('text', '无内容'))
                return # 成功就停止
            elif response.status_code == 403:
                print("   🚫 依然被拦截 (403)")
            else:
                print(f"   ⚠️ 其他状态: {response.text[:100]}")

        except Exception as e:
            print(f"   💥 请求报错: {e}")

if __name__ == "__main__":
    test_bypass_403()