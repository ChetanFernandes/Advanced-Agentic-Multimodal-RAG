import os
from backend.hybrid_pdf_parser import extract_pdf_elements
from backend.hybrid_docx_parser import extract_docx_elements
from backend.hybrid_excel_parser import extract_excel_elements
from backend.hybrid_pptx_parser import extract_pptx_elements
from backend.hybrid_text_csv_json import txt_file_processing,csv_file_processing,JSON_file_processing
import pandas as pd
from src.logger_config import log

def file_processor(file_name,file_bytes,user_id):
    try:
        file_type = file_name.rsplit(".", 1)[1].lower()
        log.info(f"File_type - {file_type}")

        if not file_bytes:
            log.warning(f"Empty file: {file_name}")
            return [], "Uploaded file is empty."

        if file_type == 'txt':
            log.info("Entering under text  dunction")
            documents,error_msg = txt_file_processing(file_name,file_bytes)

        elif file_type in ['xlsx', 'xls']:
            log.info("Entering under excel dunction")
            documents , error_msg = extract_excel_elements(file_name,file_bytes,user_id)
        
        elif file_type == "pdf":
            log.info("Entering under pdf dunction")
            documents , error_msg = extract_pdf_elements(file_name,file_bytes,user_id)

        elif file_type == "docx":
            log.info("Entering under docx dunction")
            ddocuments , error_msg = extract_docx_elements(file_name,file_bytes,user_id)

        elif file_type == "pptx":
            log.info("Entering under ppt dunction")
            documents , error_msg = extract_pptx_elements(file_name,file_bytes,user_id)

        elif file_type =='csv':
             log.info("Entering under CSV dunction")
             documents , error_msg  = csv_file_processing(file_name,file_bytes)
         

        elif file_type == "json":
             log.info("Entering under JSON dunction")
             documents , error_msg  = JSON_file_processing(file_name,file_bytes)

        else:
            return [] , "No valid file uploaded"

        return documents, error_msg
    
    except Exception as e:
        log.exception(f"File {file_name} handling failed")
        return [], f" Extraction failed: {e}"
