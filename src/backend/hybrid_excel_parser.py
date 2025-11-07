from unstructured.partition.xlsx import partition_xlsx
from agentic_rag.backend.utilis import * 
from agentic_rag.backend.Image_processing_disk import extract_Image_summaries
import openpyxl
from openpyxl import load_workbook
from PIL import Image
import io,os
import streamlit as st
import asyncio
from agentic_rag.logger_config import log
import re


# Extract only text + tables (no images).
def extract_text_tables(file_path:str):
    raw_excel_elements =  partition_xlsx( 
                         filename = file_path,
                         find_subtable = True,
                         include_header = False,
                         infer_table_structure = True,
                         starting_page_number = 1)
 
    return raw_excel_elements

def extract_images(file:str,output_dir:str):
    '''
    Function to extract image from excel (if images are present)
    '''
    wb = load_workbook(file)
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


def extract_excel_elements(file_path,file_name):

    
    # 1. Extract text + tables
    log.info("Entering function to extract raw elements")
    raw_excel_elements = extract_text_tables(file_path)
    log.info("Raw elements successfully extracted")

    # 2. Convert raw excel elements in text
    log.info("Enter processor function to extract text elements from raw elements")
    Header, Footer, Title , NarrativeText, Text , ListItem , Img , Tables = extract_text_elements(raw_excel_elements)

    # 3. Get image output directory
    log.info("Entering function to get directory path to store images")
    output_dir = get_doc_image_dir(file_path)
    log.info(f"Directory path {output_dir}")

    # 4. Extract images from Excel
    log.info("Entering function to get image from uploaded excel file")
    excel_image_info  = extract_images(file_path,output_dir)
    log.info(f"successfully extracted images -> \n {excel_image_info}")

    Image_summaries = []

    if excel_image_info:
        log.info("Extracting image_path")
        image_paths = [img["path"] for img in excel_image_info]
        log.info(f"Image_path extracted -> {image_paths}")

        # 5. Convert images to summaries + Document objects
        
        log.info('Entering function to capture image summaries')
        Image_summaries = asyncio.run(extract_Image_summaries(image_paths))
        log.info(f'Image summary extracted {Image_summaries}')

        Image_summaries = [re.sub(r'[>]+', '', t).strip() for t in Image_summaries if t.strip()]
    
    log.info("Entering function to get final information")
    final = final_doc(Header, Footer, Title , NarrativeText, Text , ListItem , Img , Tables, Image_summaries, file_name)
    documents = final.overall()
    return documents












    






    