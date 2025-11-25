#from langchain_unstructured import UnstructuredLoader
#from unstructured.cleaners.core import clean_extra_whitespace
import re
from docx import Document as DocxDocument
import os
from src.backend.utilis import get_doc_image_dir, extract_text_elements,final_doc
from src.backend.Image_processing_disk import extract_Image_summaries
import asyncio
from unstructured.partition.docx import partition_docx
from src.logger_config import log
import tempfile,io



def docx_processor(file_bytes):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
            tmp.write(file_bytes)
            tmp.flush()                  
            tmp_path = tmp.name
        log.info(f"Processing temp PDF: {tmp_path}")
        raw_docx_elements = partition_docx(
        filename = tmp_path,
        include_page_breaks = True,
        infer_table_structure = True,
        starting_page_number = 1,
        strategy = "hi_res")
        return raw_docx_elements
    except Exception:
        log.exception("Doc processing failed")
        return []
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        log.info(f"Tmp File removed: {tmp_path}")


def extract_images_from_docx(file_bytes: str, output_dir:str):
    """Extract images from DOC safely"""
    try:
        doc = DocxDocument(io.BytesIO(file_bytes))
        rels = doc.part.rels
        saved_files = []
        for rel in rels.values():
            if "image" in rel.target_ref:  # Only image relationships
                img_data = rel.target_part.blob
                img_name = os.path.basename(rel.target_ref)
                img_path = os.path.join(output_dir, img_name)
                with open(img_path, "wb") as f:
                    f.write(img_data)
                saved_files.append({"path": img_path,
                                })

        return saved_files 
    except Exception:
        log.exception("Doc image processing failed")
        return []


def extract_docx_elements(file_name,file_bytes,user_id):
    try:
        log.info("Get the folder to store extracted image from docx")
        output_dir = get_doc_image_dir(file_name,user_id) 
        log.info(f"Directory path {output_dir}")

        log.info("Enter processor function to extract raw elements from docx")
        raw_docx_elements = docx_processor(file_bytes)
        log.info(f"Raw_pdf_elements {raw_docx_elements}")

        if not raw_docx_elements:
            log.warning(f"No elements found in {file_name}")
            return [], "No readable text or elements found in the doc."

        log.info(f"[DOC_PARSE] Extracting text elements...")
        try:
           Header, Footer, Title , NarrativeText, Text , ListItem , Img , Tables = extract_text_elements(raw_docx_elements)
        except Exception:
            log.exception("[doc_PARSE] Text extraction failed.")
            return [], "Text extraction failed"

        
        log.info(f"[DOC_PARSE] Extracting images...")
        doc_image_info  = extract_images_from_docx(file_bytes, output_dir)

        Image_summaries = []
    
        if doc_image_info:
            image_paths = [img["path"] for img in doc_image_info]
            Image_summaries = asyncio.run(extract_Image_summaries(image_paths))
            if Image_summaries:
                 Image_summaries = [re.sub(r'[>]+', '', t).strip() for t in Image_summaries if t.strip()]
                 log.info(f"cleaned image summaries extracted successfully. Count: {len(Image_summaries)}")
            else:
                Image_summaries = []
        
     
        log.info(f"[DOC_PARSE] Combining text and image summaries...")
        final = final_doc(Header, Footer, Title , NarrativeText, Text , ListItem , Img , Tables, Image_summaries, file_name)
        documents = final.overall()

        if not documents:
            log.warning(f"DOC parsed but no valid text documents found in {file_name}")
            return [], "No readable text detected."

        log.info(f"Extracted {len(documents)} document chunks from {file_name}")
        return documents, None  # No error message

    except Exception as e:
        log.exception("Creating documents object failed")
        return [], f"DOCx extraction failed: {e}"
  




