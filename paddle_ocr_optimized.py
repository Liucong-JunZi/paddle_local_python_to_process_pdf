import fitz  # PyMuPDF
from paddleocr import PaddleOCR
import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import cv2
import numpy as np
import threading

# --- 配置 ---
INPUT_PDF_PATH = "input.pdf"
OUTPUT_TEXT_PATH = "output_ocr_text.txt"  # 输出文本文件
OUTPUT_PDF_PATH = "output_searchable.pdf"  # 可选：同时输出PDF
DPI = 200  # 推荐200，300会更清晰但慢很多
MAX_WORKERS = 4  # 线程数，根据CPU核心数调整
SAVE_TEXT_ONLY = True  # True=只保存文本, False=同时保存文本和PDF

# 初始化 PaddleOCR (第一次运行会自动下载模型)
print("正在初始化 PaddleOCR (首次运行会下载模型,请稍候)...")
ocr_engine = PaddleOCR(lang="ch", show_log=False)
print("PaddleOCR 初始化完成!")

# 线程局部存储，每个线程维护自己的文档对象
thread_local = threading.local()

# --- 核心函数 ---

def get_thread_doc(doc_path: str):
    """获取线程局部的PDF文档对象"""
    if not hasattr(thread_local, 'doc'):
        thread_local.doc = fitz.open(doc_path)
    return thread_local.doc


def call_paddle_ocr_direct(image_bytes: bytes) -> list:
    """直接调用 PaddleOCR 进行识别"""
    try:
        # 将字节流转换为图片
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # 检查图片是否成功解码
        if img is None:
            return []
        
        # OCR识别（使用新版 predict 方法）
        try:
            result = ocr_engine.predict(img)
        except AttributeError:
            # 如果 predict 不存在，回退到 ocr 方法
            result = ocr_engine.ocr(img)
        
        # 格式化结果 - 处理多种返回格式
        formatted_results = []
        
        if result:
            ocr_result = result[0] if isinstance(result, list) and len(result) > 0 else result
            
            # 方式1: 字典格式 (新版PaddleOCR)
            if isinstance(ocr_result, dict):
                if 'rec_texts' in ocr_result and 'rec_scores' in ocr_result:
                    rec_texts = ocr_result['rec_texts']
                    rec_scores = ocr_result['rec_scores']
                    rec_boxes = ocr_result.get('rec_boxes', [None] * len(rec_texts))
                    
                    for text, score, box in zip(rec_texts, rec_scores, rec_boxes):
                        if text and text.strip():
                            formatted_results.append({
                                "box": box,
                                "text": text,
                                "confidence": float(score)
                            })
            
            # 方式2: 对象属性格式
            elif hasattr(ocr_result, 'rec_texts') and hasattr(ocr_result, 'rec_scores'):
                rec_texts = ocr_result.rec_texts
                rec_scores = ocr_result.rec_scores
                rec_boxes = getattr(ocr_result, 'rec_boxes', [None] * len(rec_texts))
                
                for text, score, box in zip(rec_texts, rec_scores, rec_boxes):
                    if text and text.strip():
                        formatted_results.append({
                            "box": box,
                            "text": text,
                            "confidence": float(score)
                        })
            
            # 方式3: 标准列表格式 [[[box], (text, score)], ...]
            elif isinstance(ocr_result, list):
                for line in ocr_result:
                    try:
                        if line and len(line) >= 2:
                            box = line[0]
                            text_info = line[1]
                            if isinstance(text_info, (tuple, list)) and len(text_info) >= 2:
                                text = text_info[0]
                                if text and text.strip():
                                    formatted_results.append({
                                        "box": box,
                                        "text": text,
                                        "confidence": text_info[1]
                                    })
                    except Exception:
                        continue
        
        return formatted_results
        
    except Exception as e:
        # 静默处理错误，返回空结果
        return []


def process_page(doc_path: str, page_num: int, dpi: int) -> dict:
    """处理单个页面（线程安全）"""
    try:
        # 使用线程局部的文档对象
        doc = get_thread_doc(doc_path)
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
            'img_bytes': img_bytes if not SAVE_TEXT_ONLY else None,  # 只保存文本时不需要图片
            'ocr_results': ocr_results
        }
        
        return result
    
    except Exception as e:
        print(f"\n页面 {page_num + 1} 处理出错: {e}")
        return {
            'page_num': page_num,
            'width': 0,
            'height': 0,
            'img_bytes': None,
            'ocr_results': []
        }


def save_as_text(page_results: list, output_path: str):
    """将OCR结果保存为纯文本文件"""
    print(f"\n正在保存为文本文件: {output_path}")
    
    # 过滤掉None，并按page_num排序确保顺序正确
    valid_results = [r for r in page_results if r is not None]
    valid_results.sort(key=lambda x: x['page_num'])
    
    total_text_lines = 0
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for result in tqdm(valid_results, desc="写入文本", unit="页"):
            page_num = result['page_num']
            f.write(f"{'='*60}\n")
            f.write(f"第 {page_num + 1} 页\n")
            f.write(f"{'='*60}\n\n")
            
            if result['ocr_results']:
                for item in result['ocr_results']:
                    text = item.get("text", "").strip()
                    if text:
                        f.write(f"{text}\n")
                        total_text_lines += 1
            else:
                f.write("(此页无文本内容)\n")
            
            f.write("\n")
    
    print(f"✅ 文本文件已保存至: {output_path}")
    print(f"✅ 成功保存 {len(valid_results)} 页内容，共 {total_text_lines} 行文本")


def create_searchable_pdf(input_path: str, output_text_path: str, output_pdf_path: str = None):
    """创建可搜索的PDF或纯文本"""
    
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
    print(f"输出模式: {'仅文本' if SAVE_TEXT_ONLY else '文本+PDF'}")
    
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
                    print(f"\n页面 {page_num + 1} 处理失败: {e}")
                    page_results[page_num] = None
                pbar.update(1)
    
    # 保存为文本文件
    save_as_text(page_results, output_text_path)
    
    # 如果需要，同时生成PDF
    if output_pdf_path and not SAVE_TEXT_ONLY:
        print("\n正在生成可搜索PDF...")
        out_pdf = fitz.open()
        
        # 按页码顺序组装PDF
        sorted_results = sorted([r for r in page_results if r is not None], key=lambda x: x['page_num'])
        
        for result in tqdm(sorted_results, desc="组装PDF", unit="页"):
            if result['img_bytes'] is None:
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
                        try:
                            x_coords = [p[0] for p in box]
                            y_coords = [p[1] for p in box]
                            bbox = fitz.Rect(min(x_coords), min(y_coords), max(x_coords), max(y_coords))
                            
                            new_page.insert_textbox(
                                bbox, text,
                                fontsize=10,
                                fontname="helv",
                                render_mode=3  # 不可见文本
                            )
                        except:
                            pass

        print("正在保存PDF文件...")
        out_pdf.save(output_pdf_path, garbage=4, deflate=True, clean=True)
        out_pdf.close()
        print(f"✅ 可搜索的PDF已保存至: {output_pdf_path}")
    
    print(f"\n✅ 全部处理完成！")


if __name__ == "__main__":
    start_time = time.time()
    print("=" * 60)
    print("PDF OCR 批量识别工具")
    print("=" * 60)
    
    create_searchable_pdf(INPUT_PDF_PATH, OUTPUT_TEXT_PATH, OUTPUT_PDF_PATH)
    
    elapsed = time.time() - start_time
    print(f"\n总耗时: {elapsed/60:.1f} 分钟 ({elapsed:.1f} 秒)")
    print("=" * 60)
