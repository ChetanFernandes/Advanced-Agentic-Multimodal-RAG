from unstructured.partition.xlsx import partition_xlsx
from src.backend.utilis import * 
from src.backend.Image_processing_disk import extract_Image_summaries
from openpyxl import load_workbook
from PIL import Image
import io,os
import streamlit as st
import asyncio
from src.logger_config import log
import re
import tempfile
# Extract only text + tables (no images).
def extract_text_tables(file_bytes):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file_bytes)
            tmp.flush()                  
            tmp_path = tmp.name
        log.info(f"Processing temp PDF: {tmp_path}")
        raw_excel_elements =  partition_xlsx( 
                            filename = tmp_path,
                            find_subtable = True,
                            include_header = False,
                            infer_table_structure = True,
                            starting_page_number = 1)
    
        return raw_excel_elements
    except Exception:
        log.exception("Extracting excel elements failed")
        return []
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        log.info(f"Tmp File removed: {tmp_path}")

def extract_images(file_bytes:bytes,output_dir:str):
    '''
    Function to extract image from excel (if images are present)
    '''
    try:
        wb = load_workbook(io.BytesIO(file_bytes))
        ws = wb.active
        images = getattr(ws, "_images", [])

        saved_files = []


        for idx, img in  enumerate(images):
            image_bytes = img._data()
            image = Image.open(io.BytesIO(image_bytes))
            outpath = os.path.join(output_dir, f"images_{idx}.png")
            image.save(outpath)
            saved_files.append({
                "path": outpath
            })
        return saved_files 
    except Exception:
        log.exception("Extracting image from exce failed")
        return []


def extract_excel_elements(file_name,file_bytes,user_id):
    try:

        if not file_bytes:
            log.warning(f"[Excel_PARSE] Empty file: {file_name}")
            return [], "Uploaded excel is empty."

        # 1. Extract text + tables
        log.info("Entering function to extract raw elements")
        raw_excel_elements = extract_text_tables(file_bytes)
        log.info("Raw elements successfully extracted")

        if not raw_excel_elements:
            log.warning(f"No elements found in {file_name}")
            return [], "No readable text or elements found in the excel."
        
         #2. Get image output directory
        log.info("Entering function to get directory path to store images")
        output_dir = get_doc_image_dir(file_name,user_id)
        log.info(f"Directory path {output_dir}")

        #2. Convert raw excel elements in text
        log.info("Enter processor function to extract text elements from raw elements")
        try:
            Header, Footer, Title , NarrativeText, Text , ListItem , Img , Tables = extract_text_elements(raw_excel_elements)
        except Exception:
            log.exception("[Excel_PARSE] Text extraction failed.")


        # 4. Extract images from Excel
        log.info("Entering function to get image from uploaded excel file")
        excel_image_info  = extract_images(file_bytes,output_dir)
        log.info(f"successfully extracted images -> \n {excel_image_info}")

        Image_summaries = []
        
        if excel_image_info:
            log.info("Extracting image_path")
            image_paths = [img["path"] for img in excel_image_info]
            log.info(f"Image_path extracted -> {image_paths}")
            
            log.info('Entering function to capture image summaries')
            Image_summaries = asyncio.run(extract_Image_summaries(image_paths))
            if Image_summaries:
                 Image_summaries = [re.sub(r'[>]+', '', t).strip() for t in Image_summaries if t.strip()]
                 log.info(f"cleaned image summaries extracted successfully. Count: {len(Image_summaries)}")
            else:
                Image_summaries = []
        
        log.info("Entering function to get final information")
        final = final_doc(Header, Footer, Title , NarrativeText, Text , ListItem , Img , Tables, Image_summaries, file_name)
        documents = final.overall()

        if not documents:
            log.warning(f"excel parsed but no valid text documents found in {file_name}")
            return [], "No readable text detected."
        
        log.info(f"Extracted {len(documents)} document chunks from {file_name}")
        return documents, None  # No error message

    except Exception as e:
        log.exception("Creating documents object failed")
        return [], f"Excel extraction failed: {e}"
   









    






    