"""
创建一个简单的测试图片,包含中文文字
用于测试 OCR 功能
"""

import cv2
import numpy as np

# 创建白色背景
img = np.ones((400, 600, 3), dtype=np.uint8) * 255

# 添加文字 (使用 OpenCV 的默认字体,不支持中文)
# 所以我们使用 PIL 来添加中文
from PIL import Image, ImageDraw, ImageFont

# 转换为 PIL Image
pil_img = Image.fromarray(img)
draw = ImageDraw.Draw(pil_img)

# 尝试使用系统字体
try:
    # Windows 系统字体
    font = ImageFont.truetype("C:/Windows/Fonts/simhei.ttf", 40)
except:
    try:
        # 如果找不到黑体,使用宋体
        font = ImageFont.truetype("C:/Windows/Fonts/simsun.ttc", 40)
    except:
        # 如果都找不到,使用默认字体
        font = ImageFont.load_default()

# 添加文字
text_lines = [
    "PaddleOCR 测试",
    "这是一段中文文字",
    "用于测试 OCR 识别功能",
    "Test 123"
]

y_position = 50
for line in text_lines:
    draw.text((50, y_position), line, font=font, fill=(0, 0, 0))
    y_position += 80

# 转换回 OpenCV 格式
img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

# 保存图片
cv2.imwrite("test_image.png", img)
print("✅ 测试图片已生成: test_image.png")
print("内容:")
for line in text_lines:
    print(f"  {line}")
