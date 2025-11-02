from flask import Flask, request, jsonify
from paddleocr import PaddleOCR
import base64
import numpy as np
import cv2

app = Flask(__name__)

# 初始化 PaddleOCR (使用最简参数)
print("正在初始化 PaddleOCR...")
ocr = PaddleOCR(lang="ch")
print("PaddleOCR 初始化完成!")

@app.route('/predict/paddleocr', methods=['POST'])
def predict():
    try:
        data = request.json
        images = data.get('images', [])
        
        if not images:
            return jsonify({"status": "101", "msg": "No images provided"})
        
        # 解码第一张图片
        img_data = base64.b64decode(images[0])
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # OCR识别 (新版本不需要 cls 参数)
        result = ocr.ocr(img)
        
        # 格式化结果
        formatted_results = []
        if result and result[0]:
            for line in result[0]:
                box = line[0]  # 坐标
                text_info = line[1]  # (文字, 置信度)
                formatted_results.append({
                    "box": box,
                    "text": text_info[0],
                    "confidence": text_info[1]
                })
        
        return jsonify({
            "status": "000",
            "msg": "Success",
            "results": [{"data": formatted_results}]
        })
        
    except Exception as e:
        return jsonify({"status": "500", "msg": str(e)})

@app.route('/', methods=['GET'])
def index():
    return "PaddleOCR Service is running!"

if __name__ == '__main__':
    print("=" * 60)
    print("PaddleOCR 服务启动中...")
    print("服务地址: http://127.0.0.1:8866")
    print("=" * 60)
    app.run(host='0.0.0.0', port=8866, debug=False, threaded=True)