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

def deduplicate_ocr_results(results, threshold=0.8):
    """
    去重 OCR 结果
    - 移除完全相同的文本
    - 移除高度相似的文本(可选)
    """
    if not results:
        return results
    
    deduplicated = []
    seen_texts = set()
    
    for item in results:
        text = item['text']
        
        # 跳过完全相同的文本
        if text in seen_texts:
            continue
        
        # 跳过空文本
        if not text or not text.strip():
            continue
        
        seen_texts.add(text)
        deduplicated.append(item)
    
    return deduplicated


def preprocess_image(img):
    """对输入图片进行预处理以提高 OCR 识别率。

    处理步骤：
    - 转为灰度
    - 去噪（非局部均值去噪）
    - 自适应直方图均衡（CLAHE）提升对比度
    - 自适应阈值二值化
    - 形态学开运算去小噪点

    返回处理后的 BGR 图片（如果处理失败则返回原图）。
    """
    try:
        # 灰度
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 去噪
        denoised = cv2.fastNlMeansDenoising(gray, None, h=10)

        # CLAHE 提升对比度
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)

        # 自适应阈值二值化
        th = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 25, 10)

        # 形态学开运算去除小噪点
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        opened = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=1)

        # 将单通道转回三通道 BGR 返回（PaddleOCR 接受彩色或灰度，但保持一致）
        processed = cv2.cvtColor(opened, cv2.COLOR_GRAY2BGR)
        return processed
    except Exception as e:
        print(f"[DEBUG] 预处理失败,使用原图: {e}")
        return img


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
                    # 先做图片预处理以提升 OCR 效果
                    img_proc = preprocess_image(img)
                    # OCR识别
                    result = ocr.ocr(img_proc)
                    
                    # 格式化结果
                    if result and len(result) > 0:
                        ocr_result = result[0]
                        
                        # 新版 PaddleOCR 返回的是字典对象
                        try:
                            if isinstance(ocr_result, dict) or hasattr(ocr_result, '__getitem__'):
                                rec_texts = ocr_result['rec_texts'] if 'rec_texts' in ocr_result else None
                                rec_scores = ocr_result['rec_scores'] if 'rec_scores' in ocr_result else None
                            else:
                                rec_texts = getattr(ocr_result, 'rec_texts', None)
                                rec_scores = getattr(ocr_result, 'rec_scores', None)
                            
                            if rec_texts and rec_scores:
                                # 去重处理 - 移除连续重复的文本
                                prev_text = None
                                for text, score in zip(rec_texts, rec_scores):
                                    # 跳过与前一行完全相同的文本
                                    if text != prev_text:
                                        ocr_results.append({
                                            "text": text,
                                            "confidence": float(score)
                                        })
                                        prev_text = text
                        except Exception as e:
                            print(f"[DEBUG] OCR 解析错误: {e}")
            except Exception as e:
                print(f"[DEBUG] 图片处理错误: {e}")
        
        # 去重处理
        ocr_results = deduplicate_ocr_results(ocr_results)
        
        # 构建响应内容
        response_data = {
            "input_text": input_text if input_text else None,
            "ocr_results": ocr_results if ocr_results else []
        }
        
        # 格式化为文本输出
        content_lines = []
        
        if input_text:
            content_lines.append("【输入文本】")
            content_lines.append(input_text)
            content_lines.append("")
        
        if ocr_results:
            content_lines.append("【OCR识别结果】")
            for idx, item in enumerate(ocr_results, 1):
                content_lines.append(f"{idx}. {item['text']} (置信度: {item['confidence']:.2f})")
        elif not input_text:
            content_lines.append("未识别到文字,也未提供输入文本")
        
        recognized_text = "\n".join(content_lines) if content_lines else "无内容"
        
        # 返回 OpenAI 格式的响应
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
                        "content": recognized_text,
                        "metadata": response_data  # 结构化数据
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
