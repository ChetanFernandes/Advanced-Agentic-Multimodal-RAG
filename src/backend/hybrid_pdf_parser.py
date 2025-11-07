from unstructured.partition.pdf import partition_pdf
from langchain.docstore.document import Document
from agentic_rag.backend.utilis import *
from agentic_rag.backend.Image_processing_disk import extract_Image_summaries
import asyncio
from agentic_rag.logger_config import log
import re,io
import os
import tempfile
import shutil


def pdf_processor(file_bytes,output_dir):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_bytes)
        tmp.flush()                  
        tmp_path = tmp.name
    log.info(f"Processing temp PDF: {tmp_path}")
    
    try:
        raw_pdf_elements = partition_pdf(
        filename=tmp_path,
        extract_images_in_pdf=True,
        infer_table_structure=True,
        chunking_strategy="by_title",
        max_characters=4000,
        new_after_n_chars=3800,
        combine_text_under_n_chars=2000,
        image_output_dir_path=output_dir,
)
        '''
        raw_pdf_elements=partition_pdf(
        filename=tmp_path,   
        #file = io.BytesIO(file_bytes),         # mandatory
        strategy="hi_res",                                 # mandatory to use ``hi_res`` strategy
        extract_images_in_pdf=True,                       # mandatory to set as ``True``
        extract_image_block_types=["Image", "Table"],          # optional
        extract_image_block_to_payload=False,                  # optional
        extract_image_block_output_dir=output_dir)
        '''
        return raw_pdf_elements
    
    except Exception as e:
        log.exception(f"PDF processing error: {e}")
        return []
    
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        log.info(f"Tmp File removed: {tmp_path}")

     
def extract_pdf_elements(file_name,file_bytes):

    log.info("Get the foler to store extracted image from PDF")
    output_dir = get_doc_image_dir(file_name) #Directory to store image

    log.info("Enter processor function to extract raw elements from PDF")
    raw_pdf_elements = pdf_processor(file_bytes,output_dir)
    log.info(f"Raw_pdf_elements {raw_pdf_elements}")

    log.info("Enter processor function to extract text elements from raw elements")
    Header, Footer, Title , NarrativeText, Text , ListItem , Img , Tables = extract_text_elements(raw_pdf_elements)


    Image_summaries = []
     # Check if images re present in output_dir
    images = [f for f in os.listdir(output_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    if images:
        # To extract image summaries
        log.info('Pass extracted images for summarization')
        Image_summaries = asyncio.run(extract_Image_summaries(output_dir))
        log.info(f"Summary extracted from images \n{Image_summaries}")

        Image_summaries = [re.sub(r'[>]+', '', t).strip() for t in Image_summaries if t.strip()]

    shutil.rmtree(output_dir)
    final = final_doc(Header, Footer, Title , NarrativeText, Text , ListItem , Img , Tables, Image_summaries, file_name)
    documents = final.overall()
    return documents


    
    
 

   



