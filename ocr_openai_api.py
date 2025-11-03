from flask import Flask, request, jsonify
from paddleocr import PaddleOCR
import base64
import numpy as np
import cv2
import time
import uuid

app = Flask(__name__)

# 初始化 PaddleOCR
print("正在初始化 PaddleOCR...")
ocr = PaddleOCR(lang="ch")
print("PaddleOCR 初始化完成!")

# 模拟的模型列表
MODELS = {
    "paddleocr-v5": {
        "id": "paddleocr-v5",
        "object": "model",
        "created": 1677610602,
        "owned_by": "paddleocr"
    }
}

@app.route('/v1/models', methods=['GET'])
def list_models():
    """列出可用模型 (兼容 OpenAI API)"""
    return jsonify({
        "object": "list",
        "data": list(MODELS.values())
    })

@app.route('/v1/models/<model_id>', methods=['GET'])
def get_model(model_id):
    """获取模型信息 (兼容 OpenAI API)"""
    if model_id in MODELS:
        return jsonify(MODELS[model_id])
    return jsonify({"error": "Model not found"}), 404



@app.route('/v1/chat/completions', methods=['POST'])
def chat_completions():
    """
    兼容 OpenAI Chat Completions API 的 OCR 接口
    
    请求格式:
    {
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
                            "url": "data:image/png;base64,iVBORw0KG..."
                        }
                    }
                ]
            }
        ]
    }
    """
    try:
        data = request.json
        messages = data.get('messages', [])
        model = data.get('model', 'paddleocr-v5')
        
        # 提取图片和文本
        image_data = None
        input_text = None
        
        for message in messages:
            content = message.get('content')
            
            # 处理字符串格式的 content
            if isinstance(content, str):
                input_text = content
            # 处理列表格式的 content
            elif isinstance(content, list):
                for item in content:
                    if item.get('type') == 'text':
                        input_text = item.get('text', '')
                    elif item.get('type') == 'image_url':
                        image_url = item.get('image_url', {}).get('url', '')
                        if image_url.startswith('data:image'):
                            # 提取 base64 数据
                            image_data = image_url.split(',')[1] if ',' in image_url else image_url
                        else:
                            image_data = image_url
        
        # OCR 识别结果
        ocr_results = []
        
        # 如果有图片,进行 OCR 识别
        if image_data:
            try:
                # 解码图片
                img_bytes = base64.b64decode(image_data)
                nparr = np.frombuffer(img_bytes, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if img is not None:
                    # OCR识别
                    result = ocr.ocr(img)
                    
                    print(f"[DEBUG] OCR原始结果类型: {type(result)}")
                    
                    # 格式化结果 - 处理多种返回格式
                    if result:
                        ocr_result = result[0] if isinstance(result, list) and len(result) > 0 else result
                        
                        print(f"[DEBUG] OCR result[0] 类型: {type(ocr_result)}")
                        
                        # 方式1: 字典格式 (新版PaddleOCR)
                        if isinstance(ocr_result, dict):
                            print(f"[DEBUG] 字典格式，键: {ocr_result.keys()}")
                            
                            if 'rec_texts' in ocr_result and 'rec_scores' in ocr_result:
                                rec_texts = ocr_result['rec_texts']
                                rec_scores = ocr_result['rec_scores']
                                
                                print(f"[DEBUG] 找到 rec_texts，数量: {len(rec_texts)}")
                                
                                prev_text = None
                                for text, score in zip(rec_texts, rec_scores):
                                    if text and text != prev_text:  # 去重
                                        ocr_results.append({
                                            "text": text,
                                            "confidence": float(score)
                                        })
                                        prev_text = text
                        
                        # 方式2: 对象属性格式
                        elif hasattr(ocr_result, 'rec_texts') and hasattr(ocr_result, 'rec_scores'):
                            rec_texts = ocr_result.rec_texts
                            rec_scores = ocr_result.rec_scores
                            
                            print(f"[DEBUG] 对象属性格式 - 文本数: {len(rec_texts)}")
                            
                            prev_text = None
                            for text, score in zip(rec_texts, rec_scores):
                                if text and text != prev_text:  # 去重
                                    ocr_results.append({
                                        "text": text,
                                        "confidence": float(score)
                                    })
                                    prev_text = text
                        
                        # 方式3: 标准列表格式 [[[box], (text, score)], ...]
                        elif isinstance(ocr_result, list):
                            print(f"[DEBUG] 列表格式 - 行数: {len(ocr_result)}")
                            
                            prev_text = None
                            for line in ocr_result:
                                if line and len(line) >= 2:
                                    text_info = line[1]
                                    if isinstance(text_info, (tuple, list)) and len(text_info) >= 2:
                                        text = text_info[0]
                                        score = text_info[1]
                                        
                                        if text and text != prev_text:  # 去重
                                            ocr_results.append({
                                                "text": text,
                                                "confidence": float(score)
                                            })
                                            prev_text = text
                    
                    print(f"[DEBUG] 最终OCR结果数: {len(ocr_results)}")
            except Exception as e:
                print(f"[DEBUG] 图片处理错误: {e}")
        
        print(f"[DEBUG] 准备返回 - OCR结果数: {len(ocr_results)}, 输入文本: {input_text is not None}")
        
        # 直接返回纯文本，不要任何Markdown或特殊格式
        content_lines = []
        
        # 方式1：只返回识别的文本（推荐，最简洁）
        if ocr_results:
            # 直接输出每行文本，不带序号和置信度
            for item in ocr_results:
                content_lines.append(item['text'])
        
        # 如果有输入文本但没有OCR结果
        if input_text and not ocr_results:
            content_lines.append(input_text)
        
        # 如果什么都没有
        if not content_lines:
            content_lines.append("未识别到文字")
        
        recognized_text = "\n".join(content_lines)
        
        print(f"[DEBUG] 最终文本长度: {len(recognized_text)} 字符")
        print(f"[DEBUG] 文本行数: {len(content_lines)}")
        if len(recognized_text) > 200:
            print(f"[DEBUG] 文本前100字符: {recognized_text[:100]}")
            print(f"[DEBUG] 文本后100字符: {recognized_text[-100:]}")
        else:
            print(f"[DEBUG] 完整文本: {recognized_text}")
        
        # 返回标准 OpenAI 格式的响应
        response = {
            "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": recognized_text
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": len(recognized_text),
                "total_tokens": 100 + len(recognized_text)
            }
        }
        
        print(f"[DEBUG] 返回response - content长度: {len(response['choices'][0]['message']['content'])}")
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({
            "error": {
                "message": str(e),
                "type": "internal_error",
                "code": "ocr_error"
            }
        }), 500

@app.route('/', methods=['GET'])
def index():
    return jsonify({
        "name": "PaddleOCR OpenAI-Compatible API",
        "version": "1.0.0",
        "endpoints": {
            "models": "/v1/models",
            "chat": "/v1/chat/completions"
        },
        "description": "OpenAI-compatible OCR service powered by PaddleOCR"
    })

@app.route('/health', methods=['GET'])
def health():
    """健康检查"""
    return jsonify({
        "status": "healthy",
        "service": "paddleocr",
        "timestamp": int(time.time())
    })

if __name__ == '__main__':
    print("=" * 60)
    print("PaddleOCR OpenAI-Compatible API 服务启动中...")
    print("服务地址: http://127.0.0.1:8866")
    print("=" * 60)
    print("\n可用端点:")
    print("  - GET  /v1/models              - 列出模型")
    print("  - POST /v1/chat/completions    - OpenAI 兼容的聊天接口(OCR)")
    print("  - GET  /health                 - 健康检查")
    print("=" * 60)
    app.run(host='0.0.0.0', port=8866, debug=False, threaded=True)
