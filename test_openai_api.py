"""
测试 OpenAI 兼容的 OCR API

使用示例:
1. 启动服务: python ocr_openai_api.py
2. 运行测试: python test_openai_api.py
"""

import requests
import base64
import json

# API 配置
API_BASE_URL = "http://127.0.0.1:8866"


def test_health():
    """测试健康检查"""
    print("\n=== 测试健康检查 ===")
    response = requests.get(f"{API_BASE_URL}/health")
    print(f"状态码: {response.status_code}")
    print(f"响应: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")

def test_list_models():
    """测试列出模型"""
    print("\n=== 测试列出模型 ===")
    response = requests.get(f"{API_BASE_URL}/v1/models")
    print(f"状态码: {response.status_code}")
    print(f"响应: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")

def test_chat_completion_with_image(image_path):
    """测试 OpenAI 兼容的聊天接口 - 图片+文本"""
    print("\n=== 测试 Chat Completions API (图片+文本) ===")
    
    # 读取图片并转换为 base64
    with open(image_path, 'rb') as f:
        image_data = base64.b64encode(f.read()).decode('utf-8')
    
    # 构造 OpenAI 格式的请求
    payload = {
        "model": "paddleocr-v5",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "请识别这张图片中的文字"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_data}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 1000
    }
    
    response = requests.post(
        f"{API_BASE_URL}/v1/chat/completions",
        json=payload,
        headers={"Content-Type": "application/json"}
    )
    
    print(f"状态码: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"模型: {result['model']}")
        print(f"识别结果:\n{result['choices'][0]['message']['content']}")
        if 'metadata' in result['choices'][0]['message']:
            print(f"\n结构化数据: {json.dumps(result['choices'][0]['message']['metadata'], indent=2, ensure_ascii=False)}")
    else:
        print(f"错误: {response.text}")

def test_chat_only_text():
    """测试纯文本输入"""
    print("\n=== 测试纯文本输入 ===")
    
    payload = {
        "model": "paddleocr-v5",
        "messages": [
            {
                "role": "user",
                "content": "这是一段测试文本,没有图片"
            }
        ]
    }
    
    response = requests.post(
        f"{API_BASE_URL}/v1/chat/completions",
        json=payload,
        headers={"Content-Type": "application/json"}
    )
    
    print(f"状态码: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"识别结果:\n{result['choices'][0]['message']['content']}")
        if 'metadata' in result['choices'][0]['message']:
            print(f"\n结构化数据: {json.dumps(result['choices'][0]['message']['metadata'], indent=2, ensure_ascii=False)}")
    else:
        print(f"错误: {response.text}")

def test_chat_only_image(image_path):
    """测试纯图片输入"""
    print("\n=== 测试纯图片输入 ===")
    
    with open(image_path, 'rb') as f:
        image_data = base64.b64encode(f.read()).decode('utf-8')
    
    payload = {
        "model": "paddleocr-v5",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_data}"
                        }
                    }
                ]
            }
        ]
    }
    
    response = requests.post(
        f"{API_BASE_URL}/v1/chat/completions",
        json=payload,
        headers={"Content-Type": "application/json"}
    )
    
    print(f"状态码: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"识别结果:\n{result['choices'][0]['message']['content']}")
        if 'metadata' in result['choices'][0]['message']:
            print(f"\n结构化数据: {json.dumps(result['choices'][0]['message']['metadata'], indent=2, ensure_ascii=False)}")
    else:
        print(f"错误: {response.text}")



if __name__ == "__main__":
    print("=" * 60)
    print("PaddleOCR OpenAI-Compatible API 测试")
    print("=" * 60)
    
    # 测试健康检查和模型列表
    test_health()
    test_list_models()
    
    # 查找测试图片
    import os
    test_images = ["test_image.png", "test.png", "sample.jpg", "test.jpg"]
    test_image = None
    
    for img in test_images:
        if os.path.exists(img):
            test_image = img
            break
    
    if test_image:
        print(f"\n使用测试图片: {test_image}")
        test_chat_completion_with_image(test_image)
        test_chat_only_text()
        test_chat_only_image(test_image)
        
    else:
        print(f"\n⚠️  测试图片不存在")
        print("请准备一张包含文字的图片并命名为以下之一:")
        for img in test_images:
            print(f"  - {img}")
        print("\n✅ 基础 API 测试通过!")
        print("   服务已正常运行,可以开始使用了")
    
    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)
