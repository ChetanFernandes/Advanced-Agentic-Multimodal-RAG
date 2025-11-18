from unstructured.partition.pdf import partition_pdf
from langchain.docstore.document import Document
from src.backend.utilis import *
from src.backend.Image_processing_disk import extract_Image_summaries
from src.logger_config import log
import os
import tempfile




def pdf_processor(file_bytes,output_dir):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file_bytes)
            tmp.flush()                  # Ensures the bytes are physically written to disk before using them.
            tmp_path = tmp.name
        log.info(f"Processing temp PDF: {tmp_path}")
    
        raw_pdf_elements = partition_pdf(
        filename=tmp_path,
        strategy="hi_res",                                 # mandatory to use ``hi_res`` strategy
        extract_images_in_pdf=True,                       # mandatory to set as ``True``
        extract_image_block_types=["Image", "Table"],          # optional
        extract_image_block_to_payload=False,                  # optional
        extract_image_block_output_dir=output_dir)
        return raw_pdf_elements
    
    except Exception:
        log.exception(f"PDF processing failed")
        return []
    
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        log.info(f"Tmp File removed: {tmp_path}")

  
def extract_pdf_elements(file_name,file_bytes,user_id):
    """Extract text and images from PDF safely. Return (documents, error_message)."""
    try:
        if not file_bytes:
            log.warning(f"[PDF_PARSE] Empty file: {file_name}")
            return [], "Uploaded PDF is empty."
        

        log.info("Get the folder to store extracted image from PDF")
        output_dir = get_doc_image_dir(file_name,user_id) #Directory to store image
        log.info(f"Directory path {output_dir}")

        log.info("Enter processor function to extract elements from PDF")
        raw_pdf_elements = pdf_processor(file_bytes,output_dir)
        log.info(f"Raw_pdf_elements {raw_pdf_elements}")

        if not raw_pdf_elements:
            log.warning(f"No elements found in {file_name}")
            return [], "No readable text or elements found in the PDF."
        
        log.info(f"[PDF_PARSE] Extracting text elements...")
        try:
            Header, Footer, Title , NarrativeText, Text , ListItem , Img , Tables = extract_text_elements(raw_pdf_elements)
        except Exception:
            log.exception("[PDF_PARSE] Text extraction failed.")
  
        Image_summaries = []
        '''
        images = [f for f in os.listdir(output_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        if images:
            log.info('Pass extracted images for summarization')
            Image_summaries = asyncio.run(extract_Image_summaries(output_dir))
            if Image_summaries:
                 Image_summaries = [re.sub(r'[>]+', '', t).strip() for t in Image_summaries if t.strip()]
                 log.info(f"cleaned image summaries extracted successfully. Count: {len(Image_summaries)}")
            else:
                Image_summaries = []
        '''
        log.info(f"[PDF_PARSE] Combining text and image summaries...")
        final = final_doc(Header, Footer, Title , NarrativeText, Text , ListItem , Img , Tables, Image_summaries, file_name)
        documents = final.overall()

        if not documents:
            log.warning(f"PDF parsed but no valid text documents found in {file_name}")
            return [], "No readable text detected."

        log.info(f"Extracted {len(documents)} document chunks from {file_name}")
        return documents, None  # No error message

    except Exception as e:
        log.exception("Creating documents object failed")
        return [], f"PDF extraction failed: {e}"
   


    
    
 

   



