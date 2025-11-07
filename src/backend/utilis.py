import os
import base64
from pathlib import Path
from langchain_core.documents import Document
import logging as log
import re


def encode_image(image_path):
      with open(image_path, "rb") as image_file:
          return base64.b64encode(image_file.read()).decode('utf-8')
      

def Independent_image_upload():
    """Returns a valid image upload directory."""
    upload_dir = os.path.join("all_images", "Independent_images")
    os.makedirs(upload_dir, exist_ok=True)
    return upload_dir
      

def get_doc_image_dir(file_name): 
    doc_name = Path(file_name).stem # give only file name without etension
    base_output_dir = "all_images"
    output_dir = os.path.join(base_output_dir, doc_name + "_images")
    os.makedirs(output_dir,exist_ok=True)
    return output_dir


def create_document_object_text(extracted_contents_text, file_name):
        return [Document(page_content = item, metadata = {"source": file_name,"type": "combined"}) 
                          for item in extracted_contents_text or []]
  

def create_document_object_table(extracted_contents_table, file_name):
        return [Document(page_content = item, metadata = {"source": file_name,"type": "table" }) 
                          for item in extracted_contents_table or []]
      
def create_document_object_image(extracted_contents_image,file_name):
        return [Document(
                         page_content = item, 
                         metadata = {"source": file_name, "type": "image"}
                         ) 
                for item in extracted_contents_image or []]


def extract_text_elements(raw_pdf_elements):
    Header, Footer, Title , NarrativeText, Text , ListItem , Img , Tables =  [], [], [],  [], [], [], [], []
    for element in raw_pdf_elements:
        if "unstructured.documents.elements.Header" in str(type(element)):
             Header.append(str(element))
        elif "unstructured.documents.elements.Footer" in str(type(element)):
             Footer.append(str(element))
        elif "unstructured.documents.elements.Title" in str(type(element)):
             Title.append(str(element))
        elif "unstructured.documents.elements.NarrativeText" in str(type(element)):
             NarrativeText.append(str(element))
        elif "unstructured.documents.elements.Text" in str(type(element)):
             Text.append(str(element))
        elif "unstructured.documents.elements.ListItem" in str(type(element)):
             ListItem.append(str(element))
        elif "unstructured.documents.elements.Image" in str(type(element)):
             Img.append(str(element))
        elif "unstructured.documents.elements.Table" in str(type(element)):
             Tables.append(str(element))
        elif 'unstructured.documents.elements.CompositeElement' in str(type(element)):
             Text.append(str(element))

    category_counts = {}
    for element in raw_pdf_elements:
         category = str(type(element))
    if category in category_counts:
        category_counts[category] += 1
    else:
        category_counts[category] = 1

    # Unique_categories will have unique elements
    unique_categories = set(category_counts.keys())
    log.info(f"unique_Category_counts{unique_categories}")

    return Header, Footer, Title , NarrativeText, Text , ListItem , Img , Tables


class final_doc:
    def __init__(self, Header, Footer, Title, NarrativeText, Text, ListItem, Img, Tables, image_summaries,file_name):
        self.Header = Header
        self.Footer = Footer
        self.Title = Title
        self.NarrativeText = NarrativeText
        self.Text = Text
        self.ListItem = ListItem
        self.Img = Img
        self.Tables = Tables
        self.image_summaries = image_summaries
        self.file_name = file_name

    def safe_text(self, value):
        """
        Converts lists or nested values into a single clean line of text.
        Example: ["A", "B", "C"] -> "A. B. C"
        """
        if not value:
            return ""
        if isinstance(value, list):
            return ". ".join(str(v).strip() for v in value if v)
        return str(value).strip()

    def overall(self):
        """
        Combine all fields in single-line format under each label.
        """
        overall_text = "\n".join([
            f"Header:\n{self.safe_text(self.Header)}\n",
            f"Footer:\n{self.safe_text(self.Footer)}\n",
            f"Title:\n{self.safe_text(self.Title)}\n",
            f"Narrative_Text:\n{self.safe_text(self.NarrativeText)}\n",
            f"Text:\n{self.safe_text(self.Text)}\n",
            f"List Items:\n{self.safe_text(self.ListItem)}\n",
            f"Image Items:\n{self.safe_text(self.Img)}\n",
            f"Table Items:\n{self.safe_text(self.Tables)}\n",
            f"Image summaries:\n{self.safe_text(self.image_summaries)}\n",
        ])

        documents = [
            Document(
                page_content=overall_text,
                metadata={"source": self.file_name, "type": "combined"}
            )
        ]
        return documents



