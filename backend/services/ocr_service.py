import os
import random
import json
from pgdb.pgdb_manager import pg_manager

# from paddleocr import PaddleOCR

# # Initialize PaddleOCR
# # use_angle_cls=True allows detecting text direction
# # lang='en' defaults to English, user can change if needed
# _ocr_engine = PaddleOCR(use_angle_cls=True, lang='en')

def save_ocr_result(filename: str, ocr_text: str, raw_data: list, image_path: str):
    try:
        # Read image binary
        with open(image_path, "rb") as f:
            image_binary = f.read()
        
        # Insert into ocr_results
        query_ocr = """
            INSERT INTO repo_ask.ocr_results (filename, ocr_text, image)
            VALUES (%s, %s, %s)
            RETURNING id, created_at;
        """
        result_id = pg_manager.execute_query(query_ocr, (filename, ocr_text, image_binary), fetch=True)
        
        if result_id:
            ocr_id = result_id[0][0]
            created_at = result_id[0][1]
            print(f"DEBUG: OCR Result saved with ID: {ocr_id}")
            
            if raw_data and isinstance(raw_data, list) and len(raw_data) > 0:
                image_result = raw_data[0]
                if image_result:
                    
                    bbox_query = """
                        INSERT INTO repo_ask.bbox (ocr_result_id, text, confidence, x1, y1, x2, y2, x3, y3, x4, y4)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """
                    
                    for i, line in enumerate(image_result):
                        try:
                            coords = line[0] # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                            text_conf = line[1] # (text, confidence)
                            text = text_conf[0]
                            conf = text_conf[1]
                            
                            # Extract coordinates
                            x1, y1 = coords[0]
                            x2, y2 = coords[1]
                            x3, y3 = coords[2]
                            x4, y4 = coords[3]
                            
                            params = (
                                ocr_id, text, conf,
                                x1, y1, x2, y2, x3, y3, x4, y4
                            )
                            pg_manager.execute_query(bbox_query, params)
                        except Exception as bbox_err:
                            print(f"DEBUG: Failed to insert bbox #{i}: {bbox_err}")
                else:
                    print(f"DEBUG: No bbox data in image_result for OCR ID {ocr_id}")
            else:
                 print(f"DEBUG: raw_data is empty or invalid for OCR ID {ocr_id}")
                    
            return ocr_id, created_at
        else:
            print("DEBUG: result_id was empty/None after INSERT")
            return None, None
    except Exception as e:
        print(f"Failed to save Found text from images {e}")
        import traceback
        traceback.print_exc()
        return None, None

def extract_text_from_image(image_path: str):
    """
    Extracts text from an image file using PaddleOCR.
    """
    # Dummy data in PaddleOCR format: [[[[x1,y1],[x2,y2],[x3,y3],[x4,y4]], (text, confidence)], ...]
    # The outer list represents the list of results (usually one per image).
    
    filename = os.path.basename(image_path)
    rand_id = random.randint(1000, 9999)

    result = [
        [
            [[[50.0, 50.0], [300.0, 50.0], [300.0, 100.0], [50.0, 100.0]], (f"Simulated OCR Result for: {filename}", 0.99)],
            [[[50.0, 110.0], [150.0, 110.0], [150.0, 140.0], [50.0, 140.0]], (f"ID: {rand_id}", 0.98)],
            [[[160.0, 110.0], [260.0, 110.0], [260.0, 140.0], [160.0, 140.0]], ("2023-10-27", 0.97)],
            [[[50.0, 200.0], [100.0, 200.0], [100.0, 230.0], [50.0, 230.0]], ("Content: Sample Text", 0.96)],
        ]
    ]

    # result = _ocr_engine.ocr(image_path, cls=True)
    
    # Extract text
    extracted_text = ""
    # result can be None or empty list if no text found
    if result and result[0]:
        extracted_text = "\n".join([line[1][0] for line in result[0]])
    
    ocr_id, created_at = save_ocr_result(filename, extracted_text, result, image_path)

    return extracted_text, ocr_id, created_at
