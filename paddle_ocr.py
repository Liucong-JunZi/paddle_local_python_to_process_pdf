import fitz  # PyMuPDF
import requests
import base64
import os
from tqdm import tqdm

# --- 1. 请在这里配置您的信息 ---

# 您本地PaddleOCR服务的API地址
PADDLE_OCR_API_URL = "http://127.0.0.1:8866/predict/paddleocr"

# 输入的PDF文件路径 (需要进行OCR的PDF)
# Windows路径示例: "C:\\Users\\YourUser\\Documents\\scan.pdf"
# Mac/Linux路径示例: "/home/user/docs/scan.pdf"
INPUT_PDF_PATH = "input.pdf"

# 输出的带文字的PDF文件路径
OUTPUT_PDF_PATH = "output_searchable.pdf"

# 渲染PDF页面的DPI（分辨率），300 DPI是印刷质量，对于OCR来说足够清晰
# 如果识别效果不好，可以尝试提高到400
DPI = 400

# --- 2. 核心功能函数 ---

def call_paddle_ocr(image_bytes: bytes) -> list:
    """调用PaddleOCR API并返回识别结果"""
    try:
        # 将图片字节流编码为Base64
        base64_data = base64.b64encode(image_bytes).decode('utf-8')
        
        # 构建请求体
        payload = {"images": [base64_data]}
        
        # 发送POST请求
        response = requests.post(PADDLE_OCR_API_URL, json=payload)
        response.raise_for_status()  # 如果请求失败则抛出异常
        
        result = response.json()
        
        # 检查API返回状态是否成功
        if result.get("status") == "000" and "results" in result and result["results"]:
            # PaddleHub返回的结果结构是 result['results'][0]['data']
            return result["results"][0]["data"]
        else:
            print(f"API返回错误: {result.get('msg', '未知错误')}")
            return []
            
    except requests.exceptions.RequestException as e:
        print(f"调用OCR API时发生网络错误: {e}")
        return []
    except Exception as e:
        print(f"处理OCR响应时发生未知错误: {e}")
        return []


def create_searchable_pdf(input_path: str, output_path: str):
    """读取PDF，进行OCR，并创建可搜索的PDF"""
    
    if not os.path.exists(input_path):
        print(f"错误: 输入文件不存在 -> {input_path}")
        return

    print(f"正在打开PDF文件: {input_path}")
    doc = fitz.open(input_path)
    
    # 创建一个新的空PDF用于输出
    out_pdf = fitz.open()

    print("开始逐页进行OCR处理...")
    # 使用tqdm创建进度条
    for page_num in tqdm(range(len(doc)), desc="PDF处理进度"):
        page = doc.load_page(page_num)
        
        # 1. 将PDF页面渲染成高分辨率图片
        pix = page.get_pixmap(dpi=DPI)
        img_bytes = pix.tobytes("png")
        
        # 2. 调用OCR服务
        ocr_results = call_paddle_ocr(img_bytes)
        
        # 3. 创建新页面并添加图片和不可见的文字
        new_page = out_pdf.new_page(width=page.rect.width, height=page.rect.height)
        
        # 将原始图片作为背景插入
        new_page.insert_image(page.rect, stream=img_bytes)
        
        if ocr_results:
            for item in ocr_results:
                text = item.get("text", "")
                box = item.get("box") #  box是四个点的坐标列表
                
                # 计算一个简单的边界框
                x_coords = [p[0] for p in box]
                y_coords = [p[1] for p in box]
                bbox = fitz.Rect(min(x_coords), min(y_coords), max(x_coords), max(y_coords))
                
                # 插入不可见的文字 (render_mode=3)
                # 这使得文字可以被搜索和复制，但肉眼看不见
                new_page.insert_textbox(
                    bbox,
                    text,
                    fontsize=12,      # 字体大小不重要，因为它不可见
                    fontname="helv",  # 使用一个标准字体
                    render_mode=3   # 关键：设置文字渲染模式为不可见
                )

    print("OCR处理完成，正在保存新的PDF文件...")
    try:
        # 保存文件，使用优化选项减小文件大小
        out_pdf.save(output_path, garbage=4, deflate=True, clean=True)
        print(f"成功！可搜索的PDF已保存至: {output_path}")
    except Exception as e:
        print(f"保存PDF时发生错误: {e}")
    finally:
        doc.close()
        out_pdf.close()


# --- 3. 运行脚本 ---
if __name__ == "__main__":
    # 确保您的本地PaddleOCR服务已经启动！
    print("--- 开始将PDF转换为可搜索PDF ---")
    create_searchable_pdf(INPUT_PDF_PATH, OUTPUT_PDF_PATH)
    print("--- 任务结束 ---")