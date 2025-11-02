"# PaddleOCR PDF 处理工具

使用 PaddleOCR 将扫描版 PDF 转换为可搜索的 PDF 文件。

## 功能特点

- ✅ 支持大批量 PDF 处理
- ✅ 多线程并发处理,速度快
- ✅ 使用 PaddleOCR 进行中文识别
- ✅ 生成可搜索、可复制文字的 PDF
- ✅ 保留原始图片质量
- ✅ 支持自定义 DPI 和线程数

## 环境要求

- Python 3.8+
- Windows/Linux/MacOS

## 安装步骤

### 1. 创建虚拟环境

```bash
# 使用 conda (推荐)
conda create -n ocr_env python=3.9
conda activate ocr_env

# 或使用 venv
python -m venv ocr_env
# Windows
ocr_env\Scripts\activate
# Linux/Mac
source ocr_env/bin/activate
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 首次运行会自动下载模型

第一次运行时,PaddleOCR 会自动下载所需模型(约 200MB),请耐心等待。

## 使用方法

### 方式1: 直接处理 PDF (推荐)

```bash
python paddle_ocr.py
```

配置说明(在 `paddle_ocr.py` 中修改):
```python
INPUT_PDF_PATH = "input.pdf"      # 输入的 PDF 文件路径
OUTPUT_PDF_PATH = "output_searchable.pdf"  # 输出的 PDF 文件路径
DPI = 200                          # 图片分辨率 (150-400,越高越清晰但越慢)
MAX_WORKERS = 4                    # 并发线程数 (2-8,根据 CPU 调整)
```

### 方式2: 使用 OCR 服务

启动服务:
```bash
python ocr_server.py
```

服务地址: `http://127.0.0.1:8866`

API 使用示例:
```python
import requests
import base64

# 读取图片
with open("image.png", "rb") as f:
    img_base64 = base64.b64encode(f.read()).decode('utf-8')

# 调用 API
response = requests.post(
    "http://127.0.0.1:8866/predict/paddleocr",
    json={"images": [img_base64]}
)

# 获取结果
result = response.json()
print(result)
```

### 方式3: 使用 OpenAI 兼容的 API (推荐用于集成)

启动 OpenAI 兼容服务:
```bash
python ocr_openai_api.py
```

服务地址: `http://127.0.0.1:8866`

#### 3.1 使用 OpenAI SDK

```python
from openai import OpenAI
import base64

# 创建客户端
client = OpenAI(
    api_key="dummy-key",  # 当前版本不需要真实 key
    base_url="http://127.0.0.1:8866/v1"
)

# 读取图片
with open("image.png", "rb") as f:
    image_data = base64.b64encode(f.read()).decode('utf-8')

# 调用 OCR
response = client.chat.completions.create(
    model="paddleocr-v5",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "请识别图片中的文字"
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{image_data}"
                    }
                }
            ]
        }
    ]
)

print(response.choices[0].message.content)
```

#### 3.2 使用简化的 OCR 接口

```python
import requests
import base64

with open("image.png", "rb") as f:
    img_base64 = base64.b64encode(f.read()).decode('utf-8')

response = requests.post(
    "http://127.0.0.1:8866/v1/ocr",
    json={"image": img_base64}
)

result = response.json()
for item in result['results']:
    print(f"{item['text']} (置信度: {item['confidence']:.2f})")
```

#### 3.3 可用端点

- `GET /v1/models` - 列出可用模型
- `POST /v1/chat/completions` - OpenAI 兼容的聊天接口
- `POST /v1/ocr` - 简化的 OCR 接口
- `GET /health` - 健康检查

## 性能参数调优

### 速度优化

如果处理速度慢,可以:

1. **降低 DPI**:
   ```python
   DPI = 150  # 从 200 降到 150,速度提升约 2 倍
   ```

2. **增加线程数**:
   ```python
   MAX_WORKERS = 6  # 从 4 增加到 6,速度提升约 1.5 倍
   ```

3. **使用 GPU** (需要安装 PaddlePaddle GPU 版本):
   ```bash
   pip install paddlepaddle-gpu
   ```

### 质量优化

如果 OCR 识别质量不佳,可以:

1. **提高 DPI**:
   ```python
   DPI = 300  # 提高到 300,识别更准确
   ```

2. **使用更大的模型** (在 PaddleOCR 初始化时配置)

## 性能参考

测试环境: Intel i5 CPU, 8GB RAM

| DPI | 线程数 | 速度 (页/秒) | 395页耗时 |
| --- | ------ | ------------ | --------- |
| 150 | 2      | ~1.5         | ~4分钟    |
| 150 | 4      | ~2.5         | ~3分钟    |
| 200 | 2      | ~1.0         | ~7分钟    |
| 200 | 4      | ~1.8         | ~4分钟    |
| 300 | 2      | ~0.6         | ~11分钟   |
| 300 | 4      | ~1.0         | ~7分钟    |

## 项目结构

```
.
├── paddle_ocr.py           # 主处理脚本 (PDF批量处理)
├── ocr_server.py           # PaddleHub 格式的 OCR 服务
├── ocr_openai_api.py       # OpenAI 兼容的 OCR 服务 (推荐)
├── test_openai_api.py      # OpenAI API 测试脚本
├── requirements.txt        # Python 依赖
├── README.md               # 项目说明
├── input.pdf               # 输入文件 (需要自行准备)
└── output_searchable.pdf   # 输出文件 (运行后生成)
```

## 常见问题

### 1. 安装 PyMuPDF 失败

确保安装的是 `PyMuPDF` 而不是 `fitz`:
```bash
pip uninstall fitz
pip install PyMuPDF
```

### 2. OCR 识别不准确

- 提高 DPI 设置 (200-300)
- 确保原始 PDF 图片清晰
- 检查是否为中文内容 (`lang="ch"`)

### 3. 处理速度太慢

- 降低 DPI (150)
- 增加线程数 (4-8)
- 使用 GPU 版本 PaddlePaddle

### 4. 内存不足

- 减少线程数 (MAX_WORKERS = 2)
- 降低 DPI
- 分批处理大文件

## 技术栈

- **PaddleOCR**: 百度开源的 OCR 工具
- **PyMuPDF (fitz)**: PDF 处理库
- **OpenCV**: 图像处理
- **Flask**: Web 服务框架
- **tqdm**: 进度条显示

## 许可证

MIT License

## 作者

Liucong-JunZi

## 参考链接

- [PaddleOCR GitHub](https://github.com/PaddlePaddle/PaddleOCR)
- [PyMuPDF 文档](https://pymupdf.readthedocs.io/)
" 
