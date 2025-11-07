from langchain.docstore.document import Document
from pptx import Presentation
import os
from agentic_rag.backend.utilis import *
from agentic_rag.backend.Image_processing_disk import extract_Image_summaries
#from langchain_unstructured import UnstructuredLoader
#from unstructured.cleaners.core import clean_extra_whitespace
from unstructured.partition.pptx import partition_pptx
import re
import asyncio
from agentic_rag.logger_config import log
import re



def pptx_processor(file_path:str):
    raw_pptx_elements = partition_pptx(
    filename = file_path,
    include_page_breaks = True,
    infer_table_structure = True,
    starting_page_number = 1,
    strategy = "hi_res")
    return raw_pptx_elements


def extract_images_from_pptx(file_path: str, output_dir):
    """Extract embedded images from DOCX and save them to output_dir."""
    
    prs = Presentation(file_path)
    images = []
    for i, slide in enumerate(prs.slides):
        for shape in slide.shapes:
            if shape.shape_type == 13:  # Picture
                img = shape.image
                images.append({
                    "slide": i+1,
                    "image_bytes": img.blob,
                    "ext": img.ext
                })

    if not images:
        return []
    
    img_paths = []

    for idx, img_info in enumerate(images):
        slide_no = img_info["slide"]
        ext = img_info["ext"]
        img_bytes = img_info["image_bytes"]
        img_path = os.path.join(output_dir, f"slide_{slide_no}_{idx}.{ext}")
        with open(img_path, "wb") as f:
            f.write(img_bytes)
        img_paths.append({"path" : img_path})
    return img_paths    


def extract_pptx_elements(file_path,file_name):

    log.info("Get the folder to store extracted image from pptx")
    output_dir = get_doc_image_dir(file_path) #Directory to store image

    log.info("Enter processor function to extract raw elements from pptx")
    raw_pptx_elements = pptx_processor(file_path)
    log.info(f"Raw_pdf_elements {raw_pptx_elements}")

    log.info("Enter processor function to extract text elements from raw elements")
    Header, Footer, Title , NarrativeText, Text , ListItem , Img , Tables = extract_text_elements(raw_pptx_elements)
    
    doc_image_info  = extract_images_from_pptx(file_path, output_dir)

    Image_summaries = []

    if doc_image_info:

        image_paths = [img["path"] for img in doc_image_info]
    
        Image_summaries = asyncio.run(extract_Image_summaries(image_paths))

        Image_summaries = [re.sub(r'[>]+', '', t).strip() for t in Image_summaries if t.strip()]

    final = final_doc(Header, Footer, Title , NarrativeText, Text , ListItem , Img , Tables, Image_summaries, file_name)
    documents = final.overall()
    return documents

