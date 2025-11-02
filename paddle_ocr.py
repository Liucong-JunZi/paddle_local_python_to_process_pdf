import fitz  # PyMuPDF
from paddleocr import PaddleOCR
import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import cv2
import numpy as np

# --- 配置 ---
INPUT_PDF_PATH = "input.pdf"
OUTPUT_PDF_PATH = "output_searchable.pdf"
DPI = 200
MAX_WORKERS = 4  # OCR本身很占资源,建议2个线程

# 初始化 PaddleOCR (第一次运行会自动下载模型)
print("正在初始化 PaddleOCR (首次运行会下载模型,请稍候)...")
ocr_engine = PaddleOCR(lang="ch")
print("PaddleOCR 初始化完成!")

# --- 核心函数 ---

def call_paddle_ocr_direct(image_bytes: bytes) -> list:
    """直接调用 PaddleOCR 进行识别"""
    try:
        # 将字节流转换为图片
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # 检查图片是否成功解码
        if img is None:
            return []
        
        # OCR识别 (新版本不需要 cls 参数)
        result = ocr_engine.ocr(img)
        
        # 格式化结果
        formatted_results = []
        if result and len(result) > 0 and result[0]:
            for line in result[0]:
                try:
                    if line and len(line) >= 2:
                        box = line[0]
                        text_info = line[1]
                        if isinstance(text_info, (list, tuple)) and len(text_info) >= 2:
                            formatted_results.append({
                                "box": box,
                                "text": text_info[0],
                                "confidence": text_info[1]
                            })
                except Exception:
                    # 忽略单个文本行的解析错误
                    continue
        
        return formatted_results
        
    except Exception as e:
        # 不打印错误,只是返回空结果继续处理
        return []


def process_page(doc_path: str, page_num: int, dpi: int) -> dict:
    """处理单个页面"""
    doc = fitz.open(doc_path)
    page = doc.load_page(page_num)
    
    # 渲染页面
    pix = page.get_pixmap(dpi=dpi)
    img_bytes = pix.tobytes("png")
    
    # OCR识别
    ocr_results = call_paddle_ocr_direct(img_bytes)
    
    result = {
        'page_num': page_num,
        'width': page.rect.width,
        'height': page.rect.height,
        'img_bytes': img_bytes,
        'ocr_results': ocr_results
    }
    
    doc.close()
    return result


def create_searchable_pdf(input_path: str, output_path: str):
    """创建可搜索的PDF"""
    
    if not os.path.exists(input_path):
        print(f"错误: 输入文件不存在 -> {input_path}")
        return

    print(f"正在打开PDF文件: {input_path}")
    doc = fitz.open(input_path)
    total_pages = len(doc)
    doc.close()
    
    print(f"PDF共有 {total_pages} 页")
    print(f"使用 {MAX_WORKERS} 个线程并发处理...")
    print(f"DPI设置: {DPI}")
    
    page_results = [None] * total_pages
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(process_page, input_path, page_num, DPI): page_num 
            for page_num in range(total_pages)
        }
        
        with tqdm(total=total_pages, desc="OCR处理进度", unit="页") as pbar:
            for future in as_completed(futures):
                page_num = futures[future]
                try:
                    result = future.result()
                    page_results[page_num] = result
                except Exception as e:
                    print(f"\n页面 {page_num} 处理失败: {e}")
                    page_results[page_num] = None
                pbar.update(1)
    
    print("\n正在生成可搜索PDF...")
    out_pdf = fitz.open()
    
    for result in tqdm(page_results, desc="组装PDF", unit="页"):
        if result is None:
            continue
            
        new_page = out_pdf.new_page(width=result['width'], height=result['height'])
        new_page.insert_image(
            fitz.Rect(0, 0, result['width'], result['height']), 
            stream=result['img_bytes']
        )
        
        if result['ocr_results']:
            for item in result['ocr_results']:
                text = item.get("text", "")
                box = item.get("box")
                
                if box and text:
                    x_coords = [p[0] for p in box]
                    y_coords = [p[1] for p in box]
                    bbox = fitz.Rect(min(x_coords), min(y_coords), max(x_coords), max(y_coords))
                    
                    try:
                        new_page.insert_textbox(
                            bbox, text,
                            fontsize=10,
                            fontname="helv",
                            render_mode=3
                        )
                    except:
                        pass

    print("正在保存PDF文件...")
    out_pdf.save(output_path, garbage=4, deflate=True, clean=True)
    out_pdf.close()
    print(f"\n✅ 成功！可搜索的PDF已保存至: {output_path}")


if __name__ == "__main__":
    start_time = time.time()
    print("=" * 60)
    print("开始将PDF转换为可搜索PDF")
    print("=" * 60)
    
    create_searchable_pdf(INPUT_PDF_PATH, OUTPUT_PDF_PATH)
    
    elapsed = time.time() - start_time
    print(f"\n总耗时: {elapsed/60:.1f} 分钟 ({elapsed:.1f} 秒)")
    print("=" * 60)